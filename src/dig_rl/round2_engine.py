"""Cross-question batched rollouts, with per-response RNG and visual log-probs."""
from collections import OrderedDict
import random
import time
import torch
import torch.nn.functional as F
from .config import Config
from .runtime import Engine, trim_completion, demo_ids
from .round2_protocol import image_path, score_answer


class IndependentSampler:
    """Inverse-CDF sampling. Request RNG streams do not depend on packing/masks.

    Installed as a logits processor in greedy generate: this processor already
    samples raw softmax, then leaves only the sampled token for greedy selection.
    Floating-point model scores can still differ with batch shape.
    """
    def __init__(self, seeds):
        self.streams = [random.Random(int(s)) for s in seeds]

    def __call__(self, input_ids, scores):
        probabilities = scores.float().softmax(-1)
        if not torch.isfinite(probabilities).all():
            raise FloatingPointError("Invalid sampling distribution")
        cumulative = probabilities.cumsum(-1)
        cumulative[:, -1] = 1.
        # Python's double can round up to 1 when converted to float32. Keep the
        # upper endpoint open; right=True also skips zero-probability prefixes.
        uniforms = scores.new_tensor([[s.random()] for s in self.streams], dtype=torch.float32).clamp_max_(1.-2**-24)
        choices = torch.searchsorted(cumulative.contiguous(), uniforms, right=True)
        return torch.full_like(scores, -torch.inf).scatter_(1, choices, 0.)


def pack_inputs(items, pad_token_id):
    """Left padding only; retain each image tensor and grid in row order."""
    width = max(x["input_ids"].shape[1] for x in items)
    ids, masks = [], []
    for item in items:
        n = item["input_ids"].shape[1]
        ids.append(F.pad(item["input_ids"], (width - n, 0), value=pad_token_id))
        masks.append(F.pad(item["attention_mask"], (width - n, 0), value=0))
    batch = dict(input_ids=torch.cat(ids), attention_mask=torch.cat(masks))
    keys = set().union(*(x.keys() for x in items)) - {"input_ids", "attention_mask"}
    for key in keys:
        if key == "mm_token_type_ids":
            batch[key] = torch.cat([F.pad(x[key], (width-x[key].shape[1], 0), value=0) for x in items])
            continue
        if key not in ("pixel_values", "image_grid_thw"):
            raise ValueError("Unsupported processor output; inspect instead of dropping: " + key)
        if any(key not in item for item in items):
            raise ValueError("Cannot mix image/text-only rows in one generation batch")
        batch[key] = torch.cat([item[key] for item in items], dim=0)
    return batch


def visual_completion_logprobs(model, sequence, prompt_length, media):
    from torch.utils.checkpoint import checkpoint
    media = dict(media)
    if "mm_token_type_ids" in media:
        types = media["mm_token_type_ids"]
        # Answers are text, but the prompt's image locations must be retained.
        media["mm_token_type_ids"] = F.pad(types, (0, sequence.shape[1]-1-types.shape[1]), value=0)
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    kwargs = dict(input_ids=sequence[:, :-1], attention_mask=torch.ones_like(sequence[:, :-1]),
                  **media, use_cache=False)
    # Checkpoint the vocabulary projection together with CE, not just CE after
    # allocating the full [sequence, vocabulary] logits and their gradients.
    if hasattr(base, "lm_head") and hasattr(base, "model"):
        if getattr(base.config, "model_type", None) != "qwen3_5":
            raise ValueError("Chunked visual scoring requires the validated Qwen3.5 architecture")
        values = base.model(**kwargs).last_hidden_state[:, prompt_length - 1:]
        def score(x, y):
            logits = base.lm_head(x).float()
            return -F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none")
    else:
        # Small generic test models retain the public logits interface.
        values = model(**kwargs).logits[:, prompt_length - 1:]
        def score(x, y):
            return -F.cross_entropy(x.float().reshape(-1, x.shape[-1]), y.reshape(-1), reduction="none")
    targets = sequence[:, prompt_length:]
    return torch.cat([checkpoint(score, values[:, j:j+64], targets[:, j:j+64], use_reentrant=False)
                      for j in range(0, targets.shape[1], 64)])


class BatchedEngine(Engine):
    def sampling_processor(self, seeds, cap):
        return IndependentSampler(seeds)

    def __init__(self, model_dir, protocol, data_root):
        cfg = Config(horizon=protocol.max_new_tokens, max_prompt_tokens=protocol.max_input_tokens,
            group_size=protocol.group_size, rollout_batch_size=protocol.group_size,
            candidates=protocol.candidates, seed=protocol.seed, learning_rate=protocol.learning_rate)
        super().__init__(model_dir, cfg)
        from .hopper import configure_round2_kernels
        self.kernel_policy = configure_round2_kernels()
        print("Round-two kernel policy: " + self.kernel_policy, flush=True)
        self.protocol, self.data_root, self.input_cache = protocol, data_root, OrderedDict()
        self.tokenizer.padding_side = "left"

    def memory_check(self, rows):
        """One real full-length backward, with a synthetic answer, before setup.

        Uses the longest calibration input. Does not update weights or contribute
        any reward/gradient statistics to the experiment.
        """
        row = max(rows, key=lambda r: self.encoded(r)["input_ids"].shape[1])
        prompt = self.encoded(row)["input_ids"]
        token = next((int(x) for x in demo_ids(self.tokenizer, row["demo"])[0]
                      if int(x) not in self.eos_ids), None)
        if token is None:
            raise ValueError("No ordinary token for the memory check")
        completion = torch.full((1, self.protocol.max_new_tokens), token, dtype=prompt.dtype)
        sequence = torch.cat([prompt, completion], dim=1).to(self.device)
        self.zero_grad(); self.model.train(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(); self.sync(); start = time.monotonic()
        loss = -visual_completion_logprobs(self.model, sequence, prompt.shape[1], self.media(row)).sum()
        (loss / self.protocol.loss_normalizer).backward()
        self.sync()
        gradients = [p.grad for _, p in self.params if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise FloatingPointError("Full-length memory check produced missing/nonfinite gradients")
        report = dict(passed=True, row_id=row["id"], prompt_tokens=prompt.shape[1],
            completion_tokens=completion.numel(), seconds=time.monotonic()-start,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            kernel_policy=self.kernel_policy, note="Synthetic full-length backward; no parameter update")
        self.zero_grad()
        return report

    def encoded(self, row):
        if row["id"] in self.input_cache:
            self.input_cache.move_to_end(row["id"])
            return self.input_cache[row["id"]]
        from PIL import Image
        images = []
        for path in row["images"]:
            with Image.open(image_path(self.data_root, path)) as img:
                images.append(img.convert("RGB"))
        system = "Solve the problem step by step. Put the final answer inside \\boxed{}."
        if row["kind"] == "choice":
            system += " The final answer must be exactly one option letter."
        content = [{"type": "image", "image": img} for img in images] + [{"type": "text", "text": row["question"]}]
        messages = [{"role": "system", "content": [{"type": "text", "text": system}]},
                    {"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=self.protocol.thinking)
        kwargs = dict(text=[text], return_tensors="pt", padding=True)
        if images:
            kwargs.update(images=images, max_pixels=self.protocol.max_pixels)
        item = dict(self.processor(**kwargs))
        if item["input_ids"].shape[1] > self.protocol.max_input_tokens:
            raise ValueError("Input over budget; no silent image/text truncation: " + row["id"])
        self.input_cache[row["id"]] = item
        if len(self.input_cache) > 32:
            self.input_cache.popitem(last=False)
        return item

    def media(self, row):
        return {k: v.to(self.device) for k, v in self.encoded(row).items()
                if k not in ("input_ids", "attention_mask")}

    def rollout_many(self, requests, count=None, greedy=False, max_tokens=None):
        """Return group records in request order; never backfill unsampled DiG rows."""
        from transformers import GenerationConfig, LogitsProcessorList
        count = self.protocol.group_size if count is None else count
        if count < 1:
            raise ValueError("Positive response count required")
        groups = [dict(sequences=[], rewards=[], texts=[], generated_tokens=0, truncated=0, parsed=0,
                       seconds=0., responses=count, decode_slots=0, generation_batches=0.) for _ in requests]
        flat = [(i, row, (int(seed) + sample * 1000003) % (2**63 - 1))
                for i, (row, seed) in enumerate(requests) for sample in range(count)]
        self.model.eval()
        cap = max_tokens or self.protocol.max_new_tokens
        config = GenerationConfig(do_sample=False, max_new_tokens=cap, use_cache=True,
            eos_token_id=sorted(self.eos_ids), pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id)
        for offset in range(0, len(flat), self.protocol.active_sequences):
            chunk = flat[offset:offset + self.protocol.active_sequences]
            items = [self.encoded(row) for _, row, _ in chunk]
            batch = {k: v.to(self.device) for k, v in pack_inputs(items, self.tokenizer.pad_token_id).items()}
            width = batch["input_ids"].shape[1]
            sampler = None if greedy else self.sampling_processor([x[2] for x in chunk], cap)
            processor = LogitsProcessorList([] if greedy else [sampler])
            self.sync(); started = time.monotonic()
            with torch.no_grad():
                outputs = self.model.generate(**batch, generation_config=config, logits_processor=processor)
            self.sync(); elapsed = time.monotonic() - started
            if sampler is not None and hasattr(sampler, "validate"):
                sampler.validate()
            for (index, row, _), item, output in zip(chunk, items, outputs.cpu()):
                completion, truncated = trim_completion(output[width:], self.eos_ids)
                text = self.tokenizer.decode(completion, skip_special_tokens=True)
                value, parsed = score_answer(text, row)
                group = groups[index]
                group["sequences"].append(torch.cat([item["input_ids"][0], completion]).unsqueeze(0))
                group["texts"].append(text)
                group["rewards"].append(value)
                group["generated_tokens"] += len(completion)
                group["decode_slots"] += outputs.shape[1] - width
                group["generation_batches"] += 1. / len(chunk)
                group["truncated"] += int(truncated)
                group["parsed"] += int(parsed)
                # Allocated batch time for accounting, not an isolated per-question timing.
                group["seconds"] += elapsed / len(chunk)
        return groups

    def rollout(self, row, seed, count=None):
        group = self.rollout_many([(row, seed)], count)[0]
        return group["sequences"], torch.tensor(group["rewards"]), {k:v for k,v in group.items() if k not in ("sequences", "texts")}

    def demo_backward(self, row, weight=1.):
        prompt = self.encoded(row)["input_ids"]
        demo = demo_ids(self.tokenizer, row["demo"])
        if demo.shape[1] > self.protocol.max_new_tokens:
            raise ValueError("Reference solution exceeds limit: " + row["id"])
        sequence = torch.cat([prompt, demo], dim=1).to(self.device)
        self.model.train(); self.sync(); start = time.monotonic()
        loss = -visual_completion_logprobs(self.model, sequence, prompt.shape[1], self.media(row)).sum() / self.protocol.loss_normalizer
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite demonstration loss")
        (loss * weight).backward()
        self.sync()
        return dict(loss=float(loss.detach()), seconds=time.monotonic()-start, scored_tokens=demo.numel())

    def rl_backward(self, row, sequences, rewards, weight=1.):
        from .objective import rloo_advantages
        if len(sequences) != self.protocol.group_size:
            raise ValueError("Group size mismatch")
        advantages = rloo_advantages(rewards)
        self.model.train(); self.sync(); start = time.monotonic()
        total, tokens = 0., 0
        for sequence, advantage in zip(sequences, advantages):
            if float(advantage) == 0 or weight == 0:
                continue
            prompt_length = self.encoded(row)["input_ids"].shape[1]
            logp = visual_completion_logprobs(self.model, sequence.to(self.device), prompt_length, self.media(row))
            loss = -advantage.to(self.device) * logp.sum() / (self.protocol.group_size*self.protocol.loss_normalizer)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite policy loss")
            (loss * weight).backward()
            total += float(loss.detach()); tokens += logp.numel()
            del loss, logp
        self.sync()
        return dict(loss=total, seconds=time.monotonic()-start, scored_tokens=tokens)
