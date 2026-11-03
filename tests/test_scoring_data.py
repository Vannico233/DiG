from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
from dig_rl.config import Config
from dig_rl.data import normalize, split_rows, reward, numeric
from dig_rl.runtime import completion_logprobs, trim_completion, optimized_binding


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.table = torch.nn.Parameter(torch.randn(7, 7))
    def forward(self, input_ids, attention_mask, use_cache):
        return SimpleNamespace(logits=self.table[input_ids])


def test_scoring_masks_prompt_includes_eos_and_sums_fixed_horizon():
    torch.manual_seed(9)
    model = TinyLM()
    seq = torch.tensor([[0, 1, 2, 3, 4]])
    got = completion_logprobs(model, seq, 2, chunk_size=2)
    expected = torch.log_softmax(model.table, -1)[torch.tensor([1, 2, 3]), torch.tensor([2, 3, 4])]
    torch.testing.assert_close(got, expected)
    (-got.sum() / 16).backward()
    assert model.table.grad[0].abs().sum() == 0
    assert model.table.grad[3].abs().sum() > 0
    grad = model.table.grad.clone()
    model.zero_grad()
    (-expected.sum() / 16).backward()
    torch.testing.assert_close(model.table.grad, grad)


def test_eos_and_padding_same_token():
    seq, truncated = trim_completion(torch.tensor([4, 5, 2, 2, 2]), {2})
    assert seq.tolist() == [4, 5, 2] and not truncated
    seq, truncated = trim_completion(torch.tensor([4, 5]), {2})
    assert truncated and seq.tolist() == [4, 5]


def test_kernel_wrappers_are_not_confused_with_actual_fast_kernels():
    def make_wrapper(enabled):
        is_new_implementation = enabled
        def wrapped(x):
            return x if is_new_implementation else x + 0
        return wrapped
    assert optimized_binding(make_wrapper(True))
    assert not optimized_binding(make_wrapper(False))
    assert not optimized_binding(lambda x: x)
    assert not optimized_binding(None)


@pytest.mark.parametrize("text,answer,expected", [
    (r"work \boxed{1,200}", "1200", 1),
    (r"\boxed{\frac{1}{2}}", ".5", 1),
    (r"\boxed{2} final \boxed{3}", "2", 0),
    (r"<think> \boxed{2}", "2", 0),
    ("answer 2", "2", 0),
    (r"\boxed{nan}", "0", 0),
])
def test_numeric_reward(text, answer, expected):
    assert reward(text, answer) == expected


def test_splits_are_disjoint_and_remove_test_overlap():
    rows = [normalize(dict(question=f"q{i}", answer="work <<1+1=2>>2\n#### 2"), "gsm8k") for i in range(15)]
    cfg = replace(Config(), calibration_size=2, audit_size=4, validation_size=2)
    splits = split_rows(rows + [rows[0]], [rows[-1]], cfg)
    seen = set()
    for part in splits.values():
        ids = {r["id"] for r in part}
        assert not ids & seen
        seen |= ids
    assert len(seen) == 15
    assert "<<" not in rows[0]["demo"]
