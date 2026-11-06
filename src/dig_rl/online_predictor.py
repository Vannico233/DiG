"""Predictable, lagged control variates. No current-update coefficient fitting."""
import math
import re
import torch


def block_name(name):
    match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
    return "layer_" + match[1] if match else name


class OnlinePredictor:
    """EMA least squares by decoder layer, fitted only after an update commits.

    The sign convention of both gradients may be loss/descent or reward/ascent.
    Uniform Bernoulli inclusion needs no relative reweighting in this fit. The
    fitted coefficients are detached constants in the NEXT gradient estimator.
    Nonnegative scales, ridge and cap affect variance, not its conditional mean.
    """
    def __init__(self, names, decay=.9, ridge=.01, cap=1.):
        if not names or not 0 <= decay < 1 or not math.isfinite(ridge) or ridge < 0 or not math.isfinite(cap) or cap <= 0:
            raise ValueError("Invalid online predictor configuration")
        self.blocks = [block_name(n) for n in names]
        self.keys = sorted(set(self.blocks))
        self.decay, self.ridge, self.cap = decay, ridge, cap
        self.uu = {k: 0. for k in self.keys}
        self.ug = dict(self.uu)
        self.updates = 0

    def scales(self):
        reference = sum(self.uu.values()) / max(1, len(self.keys))
        values = {k: min(self.cap, max(0., self.ug[k]) /
                         (self.uu[k] + self.ridge * reference))
                  if self.uu[k] + self.ridge * reference > 0 else 0.
                  for k in self.keys}
        return values

    def observe(self, uu, ug, count):
        if count == 0:
            return
        for key in self.keys:
            if not math.isfinite(uu[key]) or uu[key] < 0 or not math.isfinite(ug[key]):
                raise FloatingPointError("Nonfinite online predictor statistics")
        for key in self.keys:
            self.uu[key] = self.decay * self.uu[key] + (1-self.decay) * uu[key]/count
            self.ug[key] = self.decay * self.ug[key] + (1-self.decay) * ug[key]/count
        self.updates += 1

    def state_dict(self):
        return dict(blocks=self.blocks, decay=self.decay, ridge=self.ridge, cap=self.cap,
                    uu=self.uu.copy(), ug=self.ug.copy(), updates=self.updates)

    def load_state_dict(self, state):
        if any(state[k] != getattr(self, k) for k in ("blocks", "decay", "ridge", "cap")):
            raise ValueError("Online predictor protocol changed")
        if set(state["uu"]) != set(self.keys) or set(state["ug"]) != set(self.keys):
            raise ValueError("Online predictor blocks changed")
        if any(not math.isfinite(state["uu"][k]) or state["uu"][k] < 0 or not math.isfinite(state["ug"][k]) for k in self.keys):
            raise ValueError("Invalid online predictor statistics")
        if type(state["updates"]) is not int or state["updates"] < 0:
            raise ValueError("Invalid online predictor update count")
        self.uu, self.ug, self.updates = dict(state["uu"]), dict(state["ug"]), state["updates"]


def update_online(engine, optimizer, rows, sampler, cfg, step, predictor, arm):
    """Same uniform question distribution/G/loss for RLOO, zero and online arms.

    Freeze A before drawing z or seeing any current rollouts:
       h_i = A*u_i + z_i/rho * (g_i - A*u_i).
    Accumulate per-question gradients, then do exactly ONE optimizer step. New
    statistics are published only after optimizer.step, never within this batch.
    """
    import time
    if arm not in ("rloo", "zero", "online"):
        raise ValueError("Unknown arm")
    rho = 1. if arm == "rloo" else cfg.rho
    scales = predictor.scales() if arm == "online" else dict.fromkeys(predictor.keys, 0.)
    draws = sampler.draw(len(rows), cfg.candidates, rho)
    selected = [(rows[i], seed) for i, yes, seed in draws if yes]
    for group in optimizer.param_groups:
        group["lr"] = cfg.learning_rate
    started = time.monotonic()
    groups = iter(engine.rollout_many(selected))
    metrics = dict(generation_seconds=time.monotonic()-started, backward_seconds=0.,
                   generated_tokens=0, responses=0, parsed=0, truncated=0, scored_tokens=0,
                   all_correct_groups=0, all_wrong_groups=0, mixed_groups=0,
                   decode_slots=0, generation_batches=0)
    params = [p for _, p in engine.params]
    accumulator = [torch.zeros_like(p) for p in params]
    touched = [False for _ in params]
    # Dot products stay on device; one compact host transfer after the update.
    stats = {k: torch.zeros(3, dtype=torch.float64, device=params[0].device) for k in predictor.keys}
    residual = torch.zeros(2, dtype=torch.float64, device=params[0].device)
    active = any(scales.values())
    for i, yes, seed in draws:
        row = rows[i]
        u = None
        # Selected rows train the future predictor even when its current scale is zero.
        if arm == "online" and (yes or active):
            optimizer.zero_grad(set_to_none=True)
            record = engine.demo_backward(row, 1.)
            metrics["backward_seconds"] += record["seconds"]
            metrics["scored_tokens"] += record["scored_tokens"]
            u = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
            for index, (p, target, value, block) in enumerate(zip(params, accumulator, u, predictor.blocks)):
                coefficient = scales[block] * (1-int(yes)/rho) / cfg.candidates
                target.add_(value, alpha=coefficient)
                touched[index] |= coefficient != 0 and p.grad is not None
        if yes:
            group = next(groups)
            optimizer.zero_grad(set_to_none=True)
            rewards = torch.tensor(group["rewards"])
            record = engine.rl_backward(row, group["sequences"], rewards, 1.)
            metrics["backward_seconds"] += record["seconds"]
            metrics["scored_tokens"] += record["scored_tokens"]
            for key in ("generated_tokens", "responses", "parsed", "truncated", "decode_slots", "generation_batches"):
                metrics[key] += group.get(key, 0)
            kind = "all_correct_groups" if bool((rewards == 1).all()) else (
                "all_wrong_groups" if bool((rewards == 0).all()) else "mixed_groups")
            metrics[kind] += 1
            for index, (p, target, block) in enumerate(zip(params, accumulator, predictor.blocks)):
                g = p.grad.detach() if p.grad is not None else torch.zeros_like(p)
                target.add_(g, alpha=1./(rho*cfg.candidates))
                touched[index] |= p.grad is not None
                if u is not None:
                    ud, gd = u[index].double(), g.double()
                    uu, ug, gg = ud.square().sum(), (ud*gd).sum(), gd.square().sum()
                    stats[block] += torch.stack((uu, ug, gg))
                    residual[0] += gg
                    residual[1] += (gd-scales[block]*ud).square().sum()
        del u
    norm = sum(g.double().square().sum() for g in accumulator).sqrt()
    if not bool(torch.isfinite(norm)):
        raise FloatingPointError("Nonfinite online estimator; no update committed")
    values = {k: v.tolist() for k,v in stats.items()}
    if any(not math.isfinite(x) for v in values.values() for x in v):
        raise FloatingPointError("Nonfinite predictor statistics; no update committed")
    optimizer.zero_grad(set_to_none=True)
    for p, g, used in zip(params, accumulator, touched):
        # Match the existing RLOO semantics: no backward at all means grad=None,
        # so AdamW must not invent a momentum-only step on skipped parameters.
        p.grad = g if used else None
    optimizer.step()
    if arm == "online":
        predictor.observe({k:v[0] for k,v in values.items()}, {k:v[1] for k,v in values.items()}, len(selected))
    gg, error = residual.tolist()
    metrics.update(step=step+1, selected=len(selected), candidates=cfg.candidates,
        candidate_ids=[rows[i]["id"] for i,_,_ in draws], selected_mask=[bool(z) for _,z,_ in draws],
        gradient_norm=float(norm), learning_rate=cfg.learning_rate, rho=rho,
        scales_used=scales, predictor_statistics=values if arm == "online" else {},
        preupdate_rl_squared_norm=gg, preupdate_residual_squared_norm=error,
        predictor_note="Lagged out-of-fit residual against one noisy RL gradient; not a certificate")
    return metrics
