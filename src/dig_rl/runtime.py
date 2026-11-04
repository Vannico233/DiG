"""Qwen3.5 text-only use of the official multimodal model; single GPU per process."""
import importlib
import inspect
from pathlib import Path
import time
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .data import extract_boxed, reward
from .io import sha256, version_info
from .objective import rloo_advantages

SYSTEM = "Solve the mathematics problem step by step. Put the final numerical answer inside \\boxed{}."


def optimized_binding(fn):
    """Inspect the implementation actually bound by Transformers, not just callable wrappers."""
    for _ in range(8):
        if not callable(fn):
            return False
        try:
            bindings = inspect.getclosurevars(fn).nonlocals
        except (TypeError, ValueError):
            bindings = {}
        if "is_new_implementation" in bindings:
            return bool(bindings["is_new_implementation"])
        if getattr(fn, "__module__", "").startswith(("fla.", "causal_conv1d")):
            return True
        fn = getattr(fn, "__wrapped__", None)
    return False


def fast_kernel_report():
    result = {"versions": version_info(), "cuda": torch.version.cuda}
    try:
        m = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
        aliases = {"causal_conv1d_fn": "causal_conv1d_fn", "causal_conv1d_update": "causal_conv1d_update",
                   "chunk_gated_delta_rule": "torch_chunk_gated_delta_rule",
                   "fused_recurrent_gated_delta_rule": "torch_recurrent_gated_delta_rule"}
        result["kernels"] = {name: optimized_binding(getattr(m, name, None) or getattr(m, alias, None))
                             for name, alias in aliases.items()}
        result["fast_path"] = all(result["kernels"].values())
        if torch.cuda.is_available():
            from .hopper import needs_tilelang
            if needs_tilelang(torch.cuda.get_device_capability(), result["versions"]["triton"]):
                import os
                from fla.ops.common.backends.tilelang import CommonTileLangBackend
                backend = CommonTileLangBackend()
                ready = (backend.is_available() and backend.is_enabled()
                         and os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") != "1")
                result["hopper_backward_backend"] = "tilelang" if ready else "unavailable"
                if not ready:
                    result.update(fast_path=False, error="Hopper/Triton requires TileLang and usable nvcc; re-run CPU preparation and source activate_offline.sh")
    except Exception as e:
        result.update(fast_path=False, error=f"{type(e).__name__}: {e}")
    return result


def prompt_ids(processor, question):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {"role": "user", "content": [{"type": "text", "text": question}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return processor.tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]


def demo_ids(tokenizer, text):
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    return torch.cat([ids, ids.new_tensor([[tokenizer.eos_token_id]])], dim=1)


def completion_logprobs(model, sequence, prompt_length, chunk_size=64):
    """Score only actual completion tokens, including EOS, never padding/prompt."""
    if not 0 < prompt_length < sequence.shape[1]:
        raise ValueError("Expected nonempty prompt and completion")
    logits = model(input_ids=sequence[:, :-1], attention_mask=torch.ones_like(sequence[:, :-1]), use_cache=False).logits
    logits = logits[:, prompt_length - 1:, :]
    targets = sequence[:, prompt_length:]
    def score(x, y):
        return -F.cross_entropy(x.float().reshape(-1, x.shape[-1]), y.reshape(-1), reduction="none")
    return torch.cat([
        checkpoint(score, logits[:, j:j + chunk_size], targets[:, j:j + chunk_size], use_reentrant=False)
        if torch.is_grad_enabled() else score(logits[:, j:j + chunk_size], targets[:, j:j + chunk_size])
        for j in range(0, targets.shape[1], chunk_size)
    ])


def trim_completion(tokens, eos_ids):
    for j, token in enumerate(tokens.tolist()):
        if token in eos_ids:
            return tokens[:j + 1], False
    return tokens, True


class Engine:
    def __init__(self, model_dir, cfg):
        from transformers import AutoModelForMultimodalLM, AutoProcessor, GenerationConfig
        from peft import LoraConfig, get_peft_model
        if not torch.cuda.is_available():
            raise RuntimeError("Real-model commands require CUDA; run pytest for CPU checks")
        self.cfg, self.device = cfg, torch.device("cuda:0")
        self.runtime = fast_kernel_report()
        if cfg.require_fast_kernels and not self.runtime["fast_path"]:
            raise RuntimeError(f"Qwen3.5 fast kernels unavailable: {self.runtime}. See docs/SERVER.md")
        self.model_dir = Path(model_dir).resolve()
        manifest = self.model_dir / "dig_model.json"
        if not manifest.is_file():
            raise ValueError("Missing dig_model.json; download using python -m dig_rl.prepare")
        self.model_id = sha256(manifest)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        self.processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        self.tokenizer = self.processor.tokenizer
        if self.tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer requires an EOS token")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        base = AutoModelForMultimodalLM.from_pretrained(
            model_dir, local_files_only=True, dtype=torch.bfloat16,
            device_map={"": 0}, attn_implementation="sdpa")
        # Explicit text-layer targets: never train the unused vision tower or LM head.
        targets = [n for n, m in base.named_modules()
                   if isinstance(m, torch.nn.Linear) and ".language_model.layers." in n]
        if not targets:
            raise RuntimeError("Qwen3.5 text layer names changed; inspect before choosing LoRA targets")
        self.model = get_peft_model(base, LoraConfig(
            r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=0.0,
            bias="none", task_type="CAUSAL_LM", target_modules=targets))
        for m in self.model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = 0.0
            if isinstance(getattr(m, "attention_dropout", None), (int, float)):
                m.attention_dropout = 0.0
        for conf in (self.model.config, getattr(self.model.config, "text_config", None)):
            if conf is not None and hasattr(conf, "attention_dropout"):
                conf.attention_dropout = 0.0
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.enable_input_require_grads()
        self.params = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        for _, p in self.params:
            p.data = p.data.float()
        eos = base.generation_config.eos_token_id or self.tokenizer.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos])
        # Fresh config avoids inherited forced/suppressed tokens, penalties, or truncation samplers.
        self.generation = GenerationConfig(
            do_sample=True, temperature=1.0, top_p=1.0, top_k=0, typical_p=1.0,
            repetition_penalty=1.0, no_repeat_ngram_size=0,
            eos_token_id=sorted(self.eos_ids), pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id, max_new_tokens=cfg.horizon, use_cache=True)
        self.cache = {}

    def sync(self):
        torch.cuda.synchronize(self.device)

    def tensors(self, row):
        if row["id"] not in self.cache:
            prompt = prompt_ids(self.processor, row["question"])
            demo = demo_ids(self.tokenizer, row["demo"])
            if prompt.shape[1] > self.cfg.max_prompt_tokens:
                raise ValueError(f"Prompt too long: {row['id']}; reprepare both arms with common limits")
            self.cache[row["id"]] = prompt, demo
        return self.cache[row["id"]]

    def zero_grad(self):
        self.model.zero_grad(set_to_none=True)

    def gradient(self):
        # Negative loss gradient = ascent direction. No gradient sketches.
        flat = torch.cat([(-p.grad if p.grad is not None else torch.zeros_like(p)).detach().reshape(-1)
                          for _, p in self.params]).float()
        if not torch.isfinite(flat).all():
            raise FloatingPointError("Nonfinite gradient; update aborted (no silent clipping)")
        return flat.cpu()

    def demo_backward(self, row, weight=1.0):
        if weight == 0:
            return dict(loss=0.0, seconds=0.0, scored_tokens=0)
        prompt, demo = self.tensors(row)
        if demo.shape[1] > self.cfg.horizon:
            raise ValueError("Demonstration exceeds H; never truncate away the answer")
        sequence = torch.cat([prompt, demo], dim=1).to(self.device)
        self.model.train()
        self.sync(); start = time.monotonic()
        loss = -completion_logprobs(self.model, sequence, prompt.shape[1]).sum() / self.cfg.horizon
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite demonstration loss")
        (loss * weight).backward()
        self.sync()
        return dict(loss=float(loss.detach()), seconds=time.monotonic() - start, scored_tokens=sequence.numel())

    def rollout(self, row, seed, count=None):
        count = self.cfg.group_size if count is None else count
        if count < 1:
            raise ValueError("Positive rollout count required")
        prompt, _ = self.tensors(row)
        prompt = prompt.to(self.device)
        sequences, rewards, lengths, parsed, truncated = [], [], [], 0, 0
        self.model.eval()
        self.sync(); start = time.monotonic()
        with torch.random.fork_rng(devices=[self.device]), torch.no_grad():
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            for offset in range(0, count, self.cfg.rollout_batch_size):
                batch = prompt.repeat(min(count - offset, self.cfg.rollout_batch_size), 1)
                outputs = self.model.generate(input_ids=batch, attention_mask=torch.ones_like(batch),
                                              generation_config=self.generation)
                for sequence in outputs.cpu():
                    completion, hit_limit = trim_completion(sequence[prompt.shape[1]:], self.eos_ids)
                    text = self.tokenizer.decode(completion, skip_special_tokens=True)
                    rewards.append(reward(text, row["answer"]))
                    parsed += extract_boxed(text) is not None
                    truncated += hit_limit
                    lengths.append(len(completion))
                    sequences.append(torch.cat([prompt[0].cpu(), completion]).unsqueeze(0))
        self.sync()
        return sequences, torch.tensor(rewards), dict(
            seconds=time.monotonic() - start, responses=count, generated_tokens=sum(lengths),
            parsed=parsed, truncated=truncated, rewards=rewards)

    def rl_backward(self, row, sequences, rewards, weight=1.0):
        if len(sequences) != self.cfg.group_size or len(rewards) != self.cfg.group_size:
            raise ValueError("Training group size mismatch")
        prompt, _ = self.tensors(row)
        advantages = rloo_advantages(rewards)
        self.model.train()
        self.sync(); start = time.monotonic()
        value, tokens = 0.0, 0
        for sequence, advantage in zip(sequences, advantages):
            if float(advantage) == 0 or weight == 0:
                continue
            logp = completion_logprobs(self.model, sequence.to(self.device), prompt.shape[1])
            loss = -advantage.to(self.device) * logp.sum() / (self.cfg.group_size * self.cfg.horizon)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite RL loss")
            (weight * loss).backward()
            value += float(loss.detach())
            tokens += sequence.numel()
        self.sync()
        return dict(loss=value, seconds=time.monotonic() - start, scored_tokens=tokens)

    def parameter_state(self):
        return {n: p.detach().cpu().clone() for n, p in self.params}

    def load_parameters(self, state):
        if set(state) != {n for n, _ in self.params}:
            raise ValueError("Checkpoint trainable parameter names differ")
        with torch.no_grad():
            for n, p in self.params:
                if p.shape != state[n].shape:
                    raise ValueError(f"Checkpoint shape mismatch: {n}")
                p.copy_(state[n].to(device=p.device, dtype=p.dtype))
