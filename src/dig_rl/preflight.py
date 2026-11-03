"""Fail before experiments if CUDA, kernels, sampling, or LoRA backward is broken."""
import argparse
from dataclasses import replace
import json
import platform
import shutil
import subprocess
import time
import torch
from .config import Config
from .io import read_rows, verify_data, write_json
from .runtime import Engine, fast_kernel_report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--config", default="configs/qwen35_4b.json")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    from pathlib import Path
    report = dict(platform=platform.platform(), torch=str(torch.__version__), cuda=torch.version.cuda,
                  kernels=fast_kernel_report(), disk_free_gb=shutil.disk_usage(args.model_dir).free / 2**30)
    try:
        import psutil
        report["cpu_available_gb"] = psutil.virtual_memory().available / 2**30
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=20)
        report["nvidia_smi"] = result.stdout.strip()
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA with BF16 support required")
        free, total = torch.cuda.mem_get_info()
        report.update(gpu_free_gb=free / 2**30, gpu_total_gb=total / 2**30)
        cfg = Config.load(args.config)
        verify_data(args.data_dir, cfg)
        engine = Engine(args.model_dir, cfg)
        verify_data(args.data_dir, cfg, engine.model_id)
        row = read_rows(Path(args.data_dir) / "calibration.jsonl")[0]
        engine.zero_grad()
        demo = engine.demo_backward(row)
        u = engine.gradient()
        if not torch.isfinite(u).all() or not float(u.norm()) > 0:
            raise RuntimeError("Demonstration backward did not produce a finite, nonzero LoRA gradient")
        engine.generation.max_new_tokens = 32  # Plumbing check only, not a training/evaluation run.
        sequences, _, generated = engine.rollout(row, cfg.seed, cfg.group_size)
        engine.zero_grad()
        synthetic = torch.zeros(cfg.group_size)
        synthetic[0] = 1  # Explicit synthetic advantages test the backward path even if all answers fail.
        rl = engine.rl_backward(row, sequences, synthetic)
        g = engine.gradient()
        if not torch.isfinite(g).all():
            raise RuntimeError("Nonfinite RL backward")
        report.update(ok=True, trainable_parameters=u.numel(), demo_gradient_norm=float(u.norm()),
            synthetic_rl_gradient_norm=float(g.norm()), demo=demo, short_rollout=generated, rl=rl,
            peak_allocated_gb=torch.cuda.max_memory_allocated() / 2**30,
            note="Synthetic rewards and 32-token generation only test plumbing; no performance claim")
    except Exception as e:
        report.update(ok=False, error=f"{type(e).__name__}: {e}")
        write_json(args.output, report)
        print(json.dumps(report, indent=2), flush=True)
        raise
    write_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
