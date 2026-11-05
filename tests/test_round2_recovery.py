from copy import deepcopy
import json
from pathlib import Path
import pytest
from dig_rl import round2_recovery as recovery
from dig_rl.io import write_json, write_jsonl, sha256
from dig_rl.round2_protocol import DATASETS, Protocol, jobs, object_hash


@pytest.fixture
def failed_run(tmp_path, monkeypatch):
    source, assets, model = [tmp_path/p for p in ('old','assets','model')]
    write_json(model/'dig_model.json',dict(model='pinned'))
    configs={d:Protocol(d,calibration_size=4) for d in DATASETS}
    old=dict(output=str(source.resolve()),model=str(model),model_id=sha256(model/'dig_model.json'),assets=str(assets),
        runtime=dict(code=recovery.LEGACY_CODE,script=recovery.LEGACY_SCRIPT,versions={'torch':'2.10.0'}),
        data={},jobs=jobs(configs))
    for dataset in DATASETS:
        write_json(assets/dataset/'manifest.json',dict(model_id=old['model_id']))
        old['data'][dataset]=sha256(assets/dataset/'manifest.json')
        rows=[dict(id=str(i),question='q',answer='a',demo='d') for i in range(4)]
        write_jsonl(assets/dataset/'calibration.jsonl',rows)
    write_json(source/'experiment.json',old)
    for dataset in DATASETS:
        directory=source/'setup'/dataset
        write_json(directory/'identity.json',dict(manifest=object_hash(old),config=configs[dataset].fingerprint()))
        saved=[dict(id=str(i),uu=1.,ug=.2,cross=.1,nu=.1,epsilon=.1,demo_seconds=.2,rl_seconds=.5,
            wall_seconds=2.,responses=16,generated_tokens=10,scored_tokens=6) for i in range(2)]
        write_jsonl(directory/'calibration.jsonl',saved)
    monkeypatch.setattr(recovery,'verify',lambda path:json.loads((path/'manifest.json').read_text()))
    runtime=dict(code='new',script='new',versions=old['runtime']['versions'])
    return source, tmp_path/'new', runtime


def test_migration_preserves_original_and_only_copies_complete_calibration(failed_run):
    source, output, runtime=failed_run
    path=source/'setup/math/calibration.jsonl'
    with path.open('ab') as f:f.write(b'{"id":"broken')
    before={str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    recovery.migrate(source,output,runtime)
    after=json.loads((output/'experiment.json').read_text())
    assert after['runtime']==runtime and after['recovery']['prefixes']['math']['records']==2
    assert len(recovery.records(output/'setup/math/calibration.jsonl'))==2
    for name,data in before.items():assert (source/name).read_bytes()==data
    assert not (output/'train').exists() and not (output/'profile_decision.json').exists()
    assert json.loads((output/'setup/math/identity.json').read_text())['manifest']==object_hash(after)
    with pytest.raises(ValueError,match='fresh'):
        recovery.migrate(source,output,runtime)


@pytest.mark.parametrize('invalid',['version','code','training','prefix','identity','model','statistic'])
def test_migration_rejects_incompatible_or_corrupt_inputs(failed_run,invalid):
    source,output,runtime=failed_run
    old=json.loads((source/'experiment.json').read_text())
    if invalid=='version':runtime['versions']={'torch':'changed'}
    elif invalid=='code':
        old['runtime']['code']='unknown';write_json(source/'experiment.json',old)
    elif invalid=='training':write_json(source/'train/math_rloo/summary.json',{})
    elif invalid=='model':write_json(Path(old['model'])/'dig_model.json',{'changed':True})
    elif invalid=='identity':write_json(source/'setup/math/identity.json',{})
    else:
        path=source/'setup/math/calibration.jsonl';rows=recovery.records(path)
        if invalid=='prefix':rows[0]['id']='wrong'
        else:rows[0]['uu']='not a number'
        write_jsonl(path,rows)
    with pytest.raises(ValueError):recovery.migrate(source,output,runtime)
    assert not output.exists()
