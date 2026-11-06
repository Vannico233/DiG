import importlib.util
import json
from pathlib import Path
from dataclasses import replace
import pytest
import torch
from dig_rl.math_engine import BufferedSampler
from dig_rl.online_predictor import OnlinePredictor, update_online
from dig_rl.math_train import train_math
from dig_rl.objective import Sampler
from dig_rl.round2_protocol import Protocol
from dig_rl.round2_engine import IndependentSampler
from test_round2 import BatchFake
from test_training import ROWS


def predictor(engine):
    return OnlinePredictor([n for n,_ in engine.params])


def test_buffered_sampler_preserves_every_request_stream_and_detects_nan():
    seeds = [1,51,7]
    old, new = IndependentSampler(seeds), BufferedSampler(seeds,100,"cpu")
    generator = torch.Generator().manual_seed(12)
    for _ in range(100):
        scores = torch.randn(3,37, generator=generator)
        torch.testing.assert_close(old(None,scores),new(None,scores),rtol=0,atol=0)
    new.validate()
    invalid = BufferedSampler([1],1,"cpu")
    invalid(None, torch.full((1,3),float("nan")))
    with pytest.raises(FloatingPointError):
        invalid.validate()


def test_predictor_groups_layers_and_shrinks_without_future_data():
    p = OnlinePredictor(["model.layers.0.q.lora_A", "model.layers.0.q.lora_B", "model.layers.1.k.lora_A"])
    assert p.blocks==["layer_0","layer_0","layer_1"]
    assert all(v==0 for v in p.scales().values())
    frozen = p.scales()
    p.observe(dict(layer_0=2.,layer_1=4.),dict(layer_0=1.,layer_1=-2.),2)
    assert all(v==0 for v in frozen.values())
    assert 0<p.scales()["layer_0"]<.5 and p.scales()["layer_1"]==0
    restored = OnlinePredictor(["model.layers.0.q.lora_A", "model.layers.0.q.lora_B", "model.layers.1.k.lora_A"])
    restored.load_state_dict(p.state_dict())
    assert restored.scales()==p.scales()


def test_first_online_update_is_zero_predictor_ablation_without_extra_rollouts():
    cfg = Protocol("gsm8k", candidates=8, group_size=2, learning_rate=.01)
    outputs = []
    for arm in ("zero","online"):
        e = BatchFake()
        opt = torch.optim.SGD([e.p],lr=cfg.learning_rate)
        pred = predictor(e)
        record = update_online(e,opt,ROWS,Sampler(13),cfg,0,pred,arm)
        outputs.append((e,record,pred))
    torch.testing.assert_close(outputs[0][0].p,outputs[1][0].p,rtol=0,atol=0)
    assert outputs[0][0].calls==outputs[1][0].calls
    assert outputs[1][2].updates==1
    assert all(v==0 for v in outputs[1][1]["scales_used"].values())


def test_online_conditional_mean_is_full_rl_with_a_frozen_nonzero_predictor():
    class Draw:
        def __init__(self,z): self.z=z
        def draw(self,*args): return [(0,self.z,7)]
    cfg = Protocol("gsm8k", candidates=1, group_size=2, learning_rate=.01, rho=.25)
    params = []
    for z in (False,True):
        e = BatchFake(); p = predictor(e)
        p.uu["adapter"]=2.; p.ug["adapter"]=1.
        record = update_online(e,torch.optim.SGD([e.p],lr=.01),ROWS,Draw(z),cfg,0,p,"online")
        assert record["scales_used"]["adapter"] == pytest.approx(1./2.02)
        params.append(e.p.detach())
    e = BatchFake()
    update_online(e,torch.optim.SGD([e.p],lr=.01),ROWS,Draw(True),cfg,0,predictor(e),"rloo")
    torch.testing.assert_close(.75*params[0]+.25*params[1],e.p,rtol=1e-12,atol=1e-12)


def test_new_rloo_matches_existing_batched_estimator():
    from dig_rl.round2_train import update
    cfg = Protocol("gsm8k", candidates=5, group_size=2, learning_rate=.01, optimizer="sgd")
    old,new = BatchFake(),BatchFake()
    update(old,torch.optim.SGD([old.p],lr=.01),ROWS,Sampler(13),cfg,0,0.,1.)
    update_online(new,torch.optim.SGD([new.p],lr=.01),ROWS,Sampler(13),cfg,0,predictor(new),"rloo")
    torch.testing.assert_close(old.p,new.p,rtol=1e-12,atol=1e-12)


def test_no_rl_signal_does_not_invent_an_adam_momentum_step():
    cfg = Protocol("gsm8k", candidates=5, group_size=2)
    e = BatchFake()
    opt = torch.optim.AdamW([e.p],lr=cfg.learning_rate,weight_decay=0.)
    p = predictor(e)
    update_online(e,opt,ROWS,Sampler(13),cfg,0,p,"rloo")
    before = e.p.detach().clone()
    # Production skips backward when every advantage is zero.
    e.rl_backward = lambda *a,**k: dict(seconds=0.,scored_tokens=0)
    update_online(e,opt,ROWS,Sampler(13),cfg,1,p,"rloo")
    torch.testing.assert_close(e.p,before,rtol=0,atol=0)
    assert e.p.grad is None


def test_nonfinite_predictor_cannot_commit_an_update():
    cfg = Protocol("gsm8k", candidates=1, group_size=2)
    e = BatchFake(); p = predictor(e)
    p.ug["adapter"] = float("nan")
    # A corrupted stored scale is not accepted as a silent zero prediction.
    with pytest.raises((FloatingPointError, ValueError)):
        p.load_state_dict(p.state_dict())


@pytest.mark.parametrize("arm",["rloo","zero","online"])
def test_math_resume_preserves_predictor_optimizer_rng_and_parameters(tmp_path,monkeypatch,arm):
    import dig_rl.math_train as module
    cfg = Protocol("gsm8k", candidates=5, group_size=2, learning_rate=.001, max_steps=4, train_seconds=1e6)
    full = BatchFake()
    train_math(full,ROWS,cfg,tmp_path/"full",{},arm,lambda *a,**k:None)
    actual = module.update_online
    def interrupted(*args):
        if args[5]==2: raise RuntimeError("interrupted")
        return actual(*args)
    monkeypatch.setattr(module,"update_online",interrupted)
    with pytest.raises(RuntimeError):
        train_math(BatchFake(),ROWS,cfg,tmp_path/"resume",{},arm,lambda *a,**k:None)
    monkeypatch.setattr(module,"update_online",actual)
    resumed = BatchFake()
    train_math(resumed,ROWS,cfg,tmp_path/"resume",{},arm,lambda *a,**k:None)
    torch.testing.assert_close(full.p,resumed.p,rtol=0,atol=0)
    a,b = [torch.load(tmp_path/x/"checkpoint.pt",weights_only=True) for x in ("full","resume")]
    assert a["predictor"]==b["predictor"] and a["sampler"]==b["sampler"]
    with pytest.raises(ValueError):
        train_math(BatchFake(),ROWS,cfg,tmp_path/"resume",{"changed":1},arm,lambda *a,**k:None)


def script():
    import sys
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/"scripts"))
    spec = importlib.util.spec_from_file_location("math_next",root/"scripts/math_next.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_profile_rejects_oom_and_over_memory_even_when_faster(tmp_path):
    m = script()
    (tmp_path/"profile").mkdir()
    for task in m.task_specs("profile"):
        r = dict(passed=task["size"]!=64, mode=task["mode"],active_sequences=task["size"],
                 tokens_per_second=task["size"],peak_reserved_gib=61 if task["size"]==32 else 20)
        (tmp_path/"profile"/(task["name"]+".json")).write_text(json.dumps(r))
    m.choose_profiles(tmp_path)
    d = json.loads((tmp_path/"profile_decision.json").read_text())
    assert all(r["active_sequences"]==16 for r in d.values())
    for task in m.task_specs("profile"):
        (tmp_path/"profile"/(task["name"]+".json")).write_text(json.dumps(dict(passed=False)))
    with pytest.raises(RuntimeError): m.choose_profiles(tmp_path)
