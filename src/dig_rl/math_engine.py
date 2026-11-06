"""Math-only generation with the same raw-softmax, per-response random streams."""
import random
import time
import torch
from .round2_engine import BatchedEngine, visual_completion_logprobs
from .runtime import demo_ids


class BufferedSampler:
    """Move uniforms once; check finite distributions once per completed batch.

    Invalid distributions are remembered on device and fail before any response
    is scored or used for training. Index clamping prevents invalid scatter on
    that error path; it does not change a valid inverse-CDF sample.
    """
    def __init__(self, seeds, cap, device):
        streams = [random.Random(int(s)) for s in seeds]
        self.uniforms = torch.tensor([[[s.random()] for s in streams] for _ in range(cap)],
                                     dtype=torch.float32, device=device).clamp_max_(1.-2**-24)
        self.invalid = torch.zeros((), dtype=torch.bool, device=device)
        self.step = 0

    def __call__(self, input_ids, scores):
        probabilities = scores.float().softmax(-1)
        self.invalid.logical_or_(~torch.isfinite(probabilities).all())
        cumulative = probabilities.cumsum(-1)
        cumulative[:, -1] = 1.
        choices = torch.searchsorted(cumulative.contiguous(), self.uniforms[self.step], right=True)
        choices.clamp_max_(scores.shape[-1]-1)
        self.step += 1
        return torch.full_like(scores, -torch.inf).scatter_(1, choices, 0.)

    def validate(self):
        if bool(self.invalid):
            raise FloatingPointError("Invalid sampling distribution; batch discarded")


class MathEngine(BatchedEngine):
    clear_generation_cache = True
    buffered_sampling = True

    def sampling_processor(self, seeds, cap):
        if not self.buffered_sampling:
            return super().sampling_processor(seeds, cap)
        return BufferedSampler(seeds, cap, self.device)

    def encoded(self, row):
        if row["images"]:
            raise ValueError("Math experiment requires text-only inputs")
        return super().encoded(row)

    def capacity_check(self, row, progress):
        """Synthetic configured-maximum shapes; no scoring or optimizer update.

        Use a fixed-length greedy decode with EOS suppressed until the cap. This
        is a capacity check, NOT a change to training decoding or a speed score.
        Backward also includes the configured maximum prompt and answer lengths.
        """
        from transformers import GenerationConfig
        token = next(int(x) for x in demo_ids(self.tokenizer, row["demo"])[0] if int(x) not in self.eos_ids)
        cfg = self.protocol
        self.zero_grad(); self.sync(); torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        prompt = torch.full((cfg.active_sequences, cfg.max_input_tokens), token, dtype=torch.long, device=self.device)
        generation = GenerationConfig(do_sample=False, max_new_tokens=cfg.max_new_tokens,
            min_new_tokens=cfg.max_new_tokens, use_cache=True, eos_token_id=sorted(self.eos_ids),
            pad_token_id=self.tokenizer.pad_token_id, bos_token_id=self.tokenizer.bos_token_id)
        progress("capacity_generation", active_sequences=cfg.active_sequences,
                 prompt_tokens=cfg.max_input_tokens, completion_tokens=cfg.max_new_tokens)
        self.model.eval(); start = time.monotonic()
        with torch.no_grad():
            outputs = self.model.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt), generation_config=generation)
        self.sync()
        if outputs.shape != (cfg.active_sequences, cfg.max_input_tokens+cfg.max_new_tokens):
            raise RuntimeError("Capacity generation stopped before configured maximum length")
        report = dict(passed=True, device_total_gib=total/2**30, free_before_gib=free/2**30,
            generation_peak_allocated_gib=torch.cuda.max_memory_allocated(self.device)/2**30,
            generation_peak_reserved_gib=torch.cuda.max_memory_reserved(self.device)/2**30,
            generation_seconds=time.monotonic()-start, prompt_tokens=cfg.max_input_tokens,
            completion_tokens=cfg.max_new_tokens, active_sequences=cfg.active_sequences)
        del outputs, prompt
        self.zero_grad(); self.sync(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        sequence = torch.full((1,cfg.max_input_tokens+cfg.max_new_tokens), token, dtype=torch.long, device=self.device)
        progress("capacity_backward", prompt_tokens=cfg.max_input_tokens, completion_tokens=cfg.max_new_tokens)
        self.model.train(); start = time.monotonic()
        loss = -visual_completion_logprobs(self.model,sequence,cfg.max_input_tokens,{}).sum()/cfg.loss_normalizer
        loss.backward(); self.sync()
        gradients = [p.grad for _,p in self.params if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise FloatingPointError("Capacity backward produced missing/nonfinite gradients")
        report["memory"] = dict(passed=True, peak_allocated_gib=torch.cuda.max_memory_allocated(self.device)/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved(self.device)/2**30, seconds=time.monotonic()-start)
        report["note"] = "Synthetic maximum prompt+completion shapes; no score, no update, no throughput comparison"
        del sequence, loss, gradients
        self.zero_grad(); self.sync(); torch.cuda.empty_cache()
        return report
