import pytest

from dig_rl.hopper import needs_tilelang


@pytest.mark.parametrize("version,expected", [("3.3.1", False), ("3.4.0", True), ("3.6.0", True),
                                            ("3.7.0", True), ("3.7.1", False), ("3.8.0", False)])
def test_hopper_backward_version_boundaries(version, expected):
    assert needs_tilelang((9, 0), version) is expected
    assert not needs_tilelang((8, 0), version)
    assert not needs_tilelang((10, 0), version)
