"""Optional real-library integration: tiny random Qwen3.5, no downloaded weights or GPU."""
import pytest
import torch

transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")
from dig_rl.runtime import completion_logprobs


def test_tiny_qwen35_lora_generation_scoring_and_signed_backward():
    torch.manual_seed(3)
    config = transformers.Qwen3_5Config(
        text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, head_dim=16,
            linear_num_key_heads=2, linear_num_value_heads=2,
            linear_key_head_dim=8, linear_value_head_dim=8,
            layer_types=["linear_attention", "full_attention"],
            rope_parameters=dict(rope_type="default", rope_theta=10000., partial_rotary_factor=1.,
                                 mrope_section=[2, 3, 3]),
            eos_token_id=2, pad_token_id=0, bos_token_id=1),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2,
            out_hidden_size=32, patch_size=2, spatial_merge_size=2, temporal_patch_size=2),
        image_token_id=62, video_token_id=63)
    base = transformers.AutoModelForMultimodalLM.from_config(config)
    targets = [n for n, m in base.named_modules()
               if isinstance(m, torch.nn.Linear) and ".language_model.layers." in n]
    assert targets
    model = peft.get_peft_model(base, peft.LoraConfig(
        r=2, lora_alpha=4, target_modules=targets, lora_dropout=0., bias="none", task_type="CAUSAL_LM"))
    prompt = torch.tensor([[1, 5, 7]])
    model.eval()
    with torch.no_grad():
        generated = model.generate(prompt, attention_mask=torch.ones_like(prompt),
            generation_config=transformers.GenerationConfig(do_sample=True, top_k=0, top_p=1.,
                temperature=1., max_new_tokens=3, eos_token_id=2, pad_token_id=0,
                return_dict_in_generate=True, output_scores=True))
        logp = completion_logprobs(model, generated.sequences, prompt.shape[1])
        online = torch.stack([torch.log_softmax(score, -1)[0, token]
                              for score, token in zip(generated.scores, generated.sequences[0, 3:])])
        torch.testing.assert_close(logp, online, atol=2e-5, rtol=2e-5)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    # Negative demonstration coefficient must remain differentiable, not clipped to zero.
    loss = -3 * (-completion_logprobs(model, generated.sequences, 3).sum() / 8)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(float(g.square().sum()) for g in grads) > 0
