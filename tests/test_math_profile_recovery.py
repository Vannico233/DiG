from dataclasses import asdict
import json
from pathlib import Path
import pytest
from dig_rl.io import sha256,write_json
from dig_rl.round2_protocol import Protocol
import dig_rl.math_profile_recovery as recovery


def source_run(tmp_path,monkeypatch):
    source=tmp_path/"old"
    model=tmp_path/"model"
    write_json(model/"dig_model.json",dict(model="test"))
    model_id=sha256(model/"dig_model.json")
    assets=tmp_path/"assets"
    data={}
    configs={}
    for d in ("gsm8k","math"):
        write_json(assets/d/"manifest.json",dict(model_id=model_id))
        data[d]=sha256(assets/d/"manifest.json")
        cfg=Protocol(d,max_new_tokens=4096 if d=="gsm8k" else 8192,candidates=8)
        configs[d]=asdict(cfg)
        for mode,size in (("legacy",16),("buffered",16),("buffered",32),("buffered",64)):
            write_json(source/"profile"/f"{d}_{mode}_{size}.json",dict(passed=True,mode=mode,
                active_sequences=size,max_new_tokens=cfg.max_new_tokens,responses=64,tokens=1000,
                seconds=10.,tokens_per_second=100.,peak_reserved_gib=78.))
    old=dict(runtime=dict(**recovery.LEGACY,versions={"torch":"test"}),output=str(source.resolve()),
             model=str(model),model_id=model_id,assets=str(assets),data=data,configs=configs)
    write_json(source/"experiment.json",old)
    monkeypatch.setattr(recovery,"verify",lambda p:json.loads((p/"manifest.json").read_text()))
    runtime=dict(code="new",script="new",math_script="new",versions={"torch":"test"})
    return source,old,runtime


def test_recovery_preserves_profiles_and_never_marks_them_verified(tmp_path,monkeypatch):
    source,old,runtime=source_run(tmp_path,monkeypatch)
    original=(source/"experiment.json").read_bytes()
    output=tmp_path/"new"
    recovery.migrate_profiles(source,output,runtime)
    new=json.loads((output/"experiment.json").read_text())
    assert new["runtime"]==runtime and new["configs"]==old["configs"]
    assert len(list((output/"historical_profiles").glob("*.json")))==8
    assert not (output/"profile_decision.json").exists()
    assert not list(output.glob("progress_*.json"))
    assert (source/"experiment.json").read_bytes()==original
    with pytest.raises(ValueError): recovery.migrate_profiles(source,output,runtime)


@pytest.mark.parametrize("change",["training","code","library","profile","data"])
def test_recovery_rejects_unsafe_source_and_leaves_no_output(tmp_path,monkeypatch,change):
    source,old,runtime=source_run(tmp_path,monkeypatch)
    if change=="training": write_json(source/"train/math_rloo/summary.json",{})
    elif change=="code": old["runtime"]["code"]="unknown"
    elif change=="library": runtime["versions"]={"torch":"different"}
    elif change=="profile": write_json(source/"profile/math_legacy_16.json",{})
    elif change=="data": write_json(Path(old["assets"])/"math/manifest.json",dict(model_id="different"))
    write_json(source/"experiment.json",old)
    with pytest.raises(ValueError): recovery.migrate_profiles(source,tmp_path/"new",runtime)
    assert not (tmp_path/"new").exists()
