from packaging.version import Version


def needs_tilelang(capability, triton_version):
    """FLA #640: gated backward is unsafe with these Hopper/Triton versions."""
    return capability[0] == 9 and Version("3.4.0") <= Version(triton_version) < Version("3.7.1")


def bound_l2norm_autotune(module):
    """Select an existing conservative FLA config, without changing its formula.

    Triton 3.6 skips benchmarking (and its disk cache) when configs has length 1.
    Restrict both passes before the first forward: backward used to benchmark 25
    candidates with a live long-sequence autograd graph on the device.
    """
    selected = []
    for name in ("l2norm_fwd_kernel", "l2norm_bwd_kernel"):
        kernel = getattr(module, name)
        configs = [c for c in kernel.configs if c.kwargs == {"BT": 32} and c.num_warps == 4]
        if len(configs) != 1:
            raise RuntimeError("Unsupported FLA L2Norm configuration layout: " + name)
        selected.append((kernel, configs[0]))
    for kernel, config in selected:
        kernel.configs = [config]
        kernel.cache.clear()


def configure_round2_kernels():
    import importlib
    import importlib.metadata
    import torch
    if torch.cuda.get_device_capability()[0] != 9:
        return "unchanged_non_hopper"
    versions = {n: importlib.metadata.version(n) for n in ("fla-core", "triton")}
    if versions != {"fla-core": "0.6.0", "triton": "3.6.0"}:
        raise RuntimeError("Revalidate the bounded L2Norm policy for new FLA/Triton versions")
    bound_l2norm_autotune(importlib.import_module("fla.modules.norm.l2norm"))
    return "fla_0.6.0_l2norm_BT32_warps4"
