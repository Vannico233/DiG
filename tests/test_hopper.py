import pytest

from dig_rl.hopper import needs_tilelang


@pytest.mark.parametrize("version,expected", [("3.3.1", False), ("3.4.0", True), ("3.6.0", True),
                                            ("3.7.0", True), ("3.7.1", False), ("3.8.0", False)])
def test_hopper_backward_version_boundaries(version, expected):
    assert needs_tilelang((9, 0), version) is expected
    assert not needs_tilelang((8, 0), version)
    assert not needs_tilelang((10, 0), version)


def test_l2norm_restricts_both_passes_to_existing_config():
    from types import SimpleNamespace
    from dig_rl.hopper import bound_l2norm_autotune
    def kernel():
        configs = [SimpleNamespace(kwargs={"BT":b}, num_warps=w) for b in (8,32,128) for w in (1,4,16)]
        return SimpleNamespace(configs=configs, cache={"old":"unsafe"})
    module = SimpleNamespace(l2norm_fwd_kernel=kernel(), l2norm_bwd_kernel=kernel())
    bound_l2norm_autotune(module)
    for k in (module.l2norm_fwd_kernel,module.l2norm_bwd_kernel):
        assert len(k.configs) == 1 and k.configs[0].kwargs == {"BT":32} and k.configs[0].num_warps == 4
        assert not k.cache
    bound_l2norm_autotune(module)  # Idempotent.
    module.l2norm_bwd_kernel.configs = []
    with pytest.raises(RuntimeError, match="layout"):
        bound_l2norm_autotune(module)
