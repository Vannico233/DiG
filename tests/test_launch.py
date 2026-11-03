import sys
from dig_rl.launch import plan


def test_two_gpu_plan_has_only_requested_stage():
    jobs = plan("train", "assets", "runs/test", "config.json", ["gsm8k", "math_numeric"], ["0", "1"])
    assert len(jobs) == 4
    assert [gpu for _, gpu, _ in jobs] == ["0", "1", "0", "1"]
    assert all(cmd[:3] == [sys.executable, "-m", "dig_rl.train"] for _, _, cmd in jobs)
    assert "--audit-file" not in jobs[0][2] and "--audit-file" in jobs[1][2]
