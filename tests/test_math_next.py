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
                 tokens_per_second=task["size"],peak_reserved_gib=78.,device_total_gib=80.,
                 generation_peak_allocated_gib=72. if task["size"]==32 else 20.,
                 memory=dict(passed=True,peak_allocated_gib=13.))
        (tmp_path/"profile"/(task["name"]+".json")).write_text(json.dumps(r))
    m.choose_profiles(tmp_path)
    d = json.loads((tmp_path/"profile_decision.json").read_text())
    assert all(r["active_sequences"]==16 for r in d.values())
    for task in m.task_specs("profile"):
        (tmp_path/"profile"/(task["name"]+".json")).write_text(json.dumps(dict(passed=False)))
    with pytest.raises(RuntimeError): m.choose_profiles(tmp_path)


def test_reserved_only_is_not_live_memory_and_backward_peak_cannot_replace_generation():
    from dig_rl.math_profile_recovery import memory_assessment
    measured = dict(passed=True,device_total_gib=80.,generation_peak_allocated_gib=20.,
                    peak_reserved_gib=78.5,memory=dict(passed=True,peak_allocated_gib=13.))
    assert memory_assessment(measured)["accepted"]
    historical = dict(measured); del historical["generation_peak_allocated_gib"]
    assert not memory_assessment(historical)["accepted"]
    for live in (72.,float("nan")):
        assert not memory_assessment(dict(measured,generation_peak_allocated_gib=live))["accepted"]
    assert not memory_assessment(dict(measured,memory=dict(passed=True,peak_allocated_gib=73.)))["accepted"]


def test_capacity_falls_back_to_smaller_batch_without_rerunning_throughput(tmp_path,monkeypatch):
    m=script()
    monkeypatch.setattr(m,"read_rows",lambda p:[dict(demo="reference")])
    class Engine:
        protocol=Protocol("math")
        def zero_grad(self): pass
        def capacity_check(self,row,progress):
            if self.protocol.active_sequences==16:
                raise torch.OutOfMemoryError("test OOM")
            return dict(passed=True,device_total_gib=80.,generation_peak_allocated_gib=22.,
                        memory=dict(passed=True,peak_allocated_gib=15.))
    m.verify_capacity(Engine(),tmp_path,tmp_path,"math",lambda *a,**k:None)
    assert json.loads((tmp_path/"capacity/math_decision.json").read_text())["active_sequences"]==8
    assert (tmp_path/"capacity/math_16.json").exists()
    assert not (tmp_path/"capacity/math_4.json").exists()


def test_capacity_check_uses_maximum_shapes_and_never_updates_parameters(monkeypatch):
    from types import SimpleNamespace
    from dig_rl.math_engine import MathEngine
    import dig_rl.math_engine as module
    engine = MathEngine.__new__(MathEngine)
    engine.protocol = SimpleNamespace(active_sequences=2,max_input_tokens=3,max_new_tokens=4,loss_normalizer=2)
    engine.device="cpu"; engine.eos_ids={2}
    engine.tokenizer=SimpleNamespace(pad_token_id=0,bos_token_id=1)
    monkeypatch.setattr(module,"demo_ids",lambda *a:torch.tensor([[3]]))
    p=torch.nn.Parameter(torch.arange(6,dtype=torch.float32))
    class Model:
        def zero_grad(self,**kwargs): p.grad=None
        def eval(self): pass
        def train(self): pass
        def generate(self,**kw):
            assert kw["input_ids"].shape==(2,3)
            assert kw["generation_config"].min_new_tokens==4
            return torch.full((2,7),3,dtype=torch.long)
        def __call__(self,**kw):
            assert kw["input_ids"].shape==(1,6)
            return SimpleNamespace(logits=p[None,None,:].expand(1,6,6))
    engine.model=Model(); engine.params=[("adapter",p)]; engine.sync=lambda:None
    for key in ("empty_cache","reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda,key,lambda *a,**kw:None)
    monkeypatch.setattr(torch.cuda,"mem_get_info",lambda *a:(70*2**30,80*2**30))
    monkeypatch.setattr(torch.cuda,"max_memory_allocated",lambda *a:20*2**30)
    monkeypatch.setattr(torch.cuda,"max_memory_reserved",lambda *a:78*2**30)
    before=p.detach().clone()
    report=engine.capacity_check(dict(demo="x"),lambda *a,**kw:None)
    assert report["passed"] and report["generation_peak_allocated_gib"]==20
    torch.testing.assert_close(p,before,rtol=0,atol=0)
    assert p.grad is None
