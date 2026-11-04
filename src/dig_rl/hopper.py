from packaging.version import Version


def needs_tilelang(capability, triton_version):
    """FLA #640: gated backward is unsafe with these Hopper/Triton versions."""
    return capability[0] == 9 and Version("3.4.0") <= Version(triton_version) < Version("3.7.1")
