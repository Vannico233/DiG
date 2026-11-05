import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from dig_rl.round2_protocol import Protocol, DATASETS, jobs, boxed_text, score_answer, image_path
from dig_rl.round2_engine import IndependentSampler, pack_inputs, visual_completion_logprobs
from dig_rl.round2_data import normalize, split, verify
from dig_rl.round2_train import train, update, evaluate
from dig_rl.objective import Sampler
from test_training import FakeEngine, ROWS


def test_protocol_and_eight_job_layout():
    configs = {d:Protocol(d) for d in DATASETS}
    plan = jobs(configs)
    assert len(plan) == 8
    for gpu, dataset in zip(range(4), DATASETS):
        pair = [j for j in plan if j['gpu'] == str(gpu)]
        assert {j['arm'] for j in pair} == {'rloo','dig'}
        assert all(j['dataset'] == dataset for j in pair)
        assert pair[0]['config'] == pair[1]['config']
    assert replace(configs['math'], max_new_tokens=8192).loss_normalizer == 2048
    with pytest.raises(ValueError):
        jobs(configs, ['0','0','1','2'])
    with pytest.raises(ValueError):
        Protocol('math', active_sequences=0)


def test_sampling_independent_of_packing_and_request_selection():
    scores = torch.tensor([[.2,-1.,2.], [1.,2.,3.], [-3.,4.,1.]])
    batched = IndependentSampler([10,20,30])
    single = IndependentSampler([20])
    for _ in range(20):
        full = batched(None, scores)
        assert full[1].argmax() == single(None,scores[1:2])[0].argmax()
        assert torch.isfinite(full).sum() == 3


def test_sampler_matches_raw_softmax_distribution():
    scores = torch.tensor([[0.,1.,2.]])
    sampler = IndependentSampler([13])
    counts = torch.zeros(3)
    for _ in range(10000):
        counts[sampler(None,scores)[0].argmax()] += 1
    torch.testing.assert_close(counts/10000, scores.softmax(-1)[0], atol=.015, rtol=0)


def test_sampler_handles_float32_rounding_at_uniform_endpoints():
    sampler=IndependentSampler([1,2])
    sampler.streams=[SimpleNamespace(random=lambda:0.),SimpleNamespace(random=lambda:1.-1e-12)]
    result=sampler(None,torch.tensor([[-torch.inf,0.],[0.,0.]]))
    assert result.argmax(-1).tolist()==[1,1]


def test_left_padding_preserves_image_grid_and_pixels():
    a = dict(input_ids=torch.tensor([[1,2]]), attention_mask=torch.ones(1,2,dtype=torch.long),
             pixel_values=torch.tensor([[3.],[4.]]), image_grid_thw=torch.tensor([[1,2,2]]))
    b = dict(input_ids=torch.tensor([[5]]), attention_mask=torch.ones(1,1,dtype=torch.long),
             pixel_values=torch.tensor([[6.]]), image_grid_thw=torch.tensor([[1,1,1]]))
    result = pack_inputs([a,b],0)
    assert result['input_ids'].tolist() == [[1,2],[0,5]]
    assert result['attention_mask'].tolist() == [[1,1],[0,1]]
    assert result['pixel_values'].flatten().tolist() == [3.,4.,6.]
    assert result['image_grid_thw'].tolist() == [[1,2,2],[1,1,1]]
    with pytest.raises(ValueError):
        pack_inputs([a,{k:v for k,v in b.items() if k!='pixel_values'}],0)


def test_visual_gradient_recompute_passes_images_and_excludes_prompt():
    parameter = torch.nn.Parameter(torch.arange(6.,dtype=torch.float64))
    pixels = torch.ones(1,2)
    grid = torch.ones(1,3,dtype=torch.long)
    class Model:
        def __call__(self, **kwargs):
            assert kwargs['pixel_values'] is pixels and kwargs['image_grid_thw'] is grid
            assert kwargs['input_ids'].tolist() == [[0,1,2]]
            return SimpleNamespace(logits=parameter[None,None,:].expand(1,3,6))
    result = visual_completion_logprobs(Model(),torch.tensor([[0,1,2,3]]),2,
                                        dict(pixel_values=pixels,image_grid_thw=grid))
    torch.testing.assert_close(result,parameter.float().log_softmax(0)[torch.tensor([2,3])])
    (-result.sum()).backward()
    assert parameter.grad is not None and parameter.grad.abs().sum() > 0


def test_boxed_answer_nested_and_choice_grading():
    assert boxed_text(r'\boxed{\frac{1}{2}}') == r'\frac{1}{2}'
    assert boxed_text(r'\boxed{3} then \boxed{4}') == '4'
    assert boxed_text(r'<think>\boxed{4}') is None
    assert score_answer(r'\boxed{B}',dict(answer='B',kind='choice')) == (1.,True)
    assert score_answer(r'\boxed{B or C}',dict(answer='B',kind='choice')) == (0.,False)
    assert score_answer(r'\boxed{3} then \boxed{4}',dict(answer='3',kind='numeric'))[0] == 0


def test_math_equivalence_if_installed(monkeypatch):
    library = pytest.importorskip('math_verify')
    import os
    if os.name == 'nt':
        # The isolated Windows test runner cannot spawn the library timeout worker.
        # Production Linux keeps the package's default bounded parsing/verification.
        from functools import partial
        monkeypatch.setattr(library, 'parse', partial(library.parse, parsing_timeout=None))
        monkeypatch.setattr(library, 'verify', partial(library.verify, timeout_seconds=None))
    assert score_answer(r'\boxed{\sqrt{8}}',dict(answer=r'2\sqrt{2}',kind='math'))[0] == 1
    assert score_answer(r'\boxed{(1,2)}',dict(answer='(2,1)',kind='math'))[0] == 0


def test_vlm_identity_includes_image_and_never_leaks_solution(tmp_path):
    Image = pytest.importorskip('PIL.Image')
    raw = dict(image=Image.new('RGB',(10,10),'red'), question='Which?',choices=['one','two'],answer=1,
               hint='visible hint', solution='SECRET explanation', lecture='SECRET lecture')
    row = normalize(raw,'scienceqa_image',tmp_path)
    assert 'SECRET' not in row['question'] and 'SECRET explanation' in row['demo']
    assert 'SECRET lecture' not in row['demo']
    assert row['answer']=='B' and row['kind']=='choice'
    assert image_path(tmp_path,row['images'][0]).exists()
    other = normalize(dict(raw,image=Image.new('RGB',(10,10),'blue')),'scienceqa_image',tmp_path)
    assert other['id'] != row['id']
    geo = normalize(dict(problem='<image>Find x.',answer=r'2\sqrt{2}',images=[raw['image']]),'geo3k',tmp_path)
    assert geo['proxy']=='answer_only'
    with pytest.raises(ValueError):
        image_path(tmp_path,'../outside.png')


def test_split_is_disjoint_and_does_not_drop_test():
    training = [dict(id=str(i)) for i in range(1000)]
    test = [dict(id=str(i)) for i in range(900,1020)]
    result = split(training,test)
    ids = [r['id'] for rows in result.values() for r in rows]
    assert len(ids)==len(set(ids)) and result['test']==test
    assert len(result['calibration'])==32 and len(result['audit'])==96


class BatchFake(FakeEngine):
    def rollout_many(self, requests, **kwargs):
        out=[]
        for row,seed in requests:
            seq,rewards,stats=self.rollout(row,seed)
            out.append(dict(sequences=seq,rewards=rewards.tolist(),**stats))
        return out


def test_batched_update_matches_original_estimator():
    from dig_rl.train import update as old_update
    from dig_rl.config import Config
    old,new=FakeEngine(),BatchFake()
    cfg=Protocol('gsm8k',group_size=2,candidates=5,optimizer='sgd',learning_rate=.01)
    old_cfg=replace(Config(),group_size=2,rollout_batch_size=2,candidates=5)
    old_update(old,torch.optim.SGD([old.p],lr=.01),ROWS,Sampler(12),old_cfg,0,.7,.5)
    record=update(new,torch.optim.SGD([new.p],lr=.01),ROWS,Sampler(12),cfg,0,.7,.5)
    torch.testing.assert_close(old.p,new.p,rtol=0,atol=0)
    assert old.calls==new.calls
    assert record['all_correct_groups']+record['all_wrong_groups']+record['mixed_groups']==record['selected']


@pytest.mark.parametrize('optimizer',['sgd','adamw'])
def test_resume_keeps_optimizer_sampler_and_completed_steps(tmp_path,monkeypatch,optimizer):
    import dig_rl.round2_train as module
    cfg=Protocol('gsm8k',group_size=2,candidates=5,optimizer=optimizer,max_steps=4,train_seconds=1e6)
    full=BatchFake()
    train(full,ROWS,cfg,tmp_path/'full',{},.7,.5,lambda *a,**k:None)
    actual=module.update
    def interrupted(*args):
        if args[5]==2:
            raise RuntimeError('test interruption')
        return actual(*args)
    monkeypatch.setattr(module,'update',interrupted)
    with pytest.raises(RuntimeError):
        train(BatchFake(),ROWS,cfg,tmp_path/'resume',{},.7,.5,lambda *a,**k:None)
    monkeypatch.setattr(module,'update',actual)
    restored=BatchFake()
    train(restored,ROWS,cfg,tmp_path/'resume',{},.7,.5,lambda *a,**k:None)
    torch.testing.assert_close(restored.p,full.p,rtol=0,atol=0)
    with pytest.raises(ValueError):
        train(BatchFake(),ROWS,cfg,tmp_path/'resume',{'changed':1},.7,.5,lambda *a,**k:None)


def test_cheap_evaluation_is_resumable_and_protocol_bound(tmp_path):
    cfg=Protocol('gsm8k')
    class Engine:
        def rollout_many(self, requests, **kwargs):
            assert kwargs==dict(count=1,greedy=True)
            return [dict(rewards=[1.],sequences=[],texts=['ok'],generated_tokens=2,truncated=0,parsed=1,seconds=0.,responses=1) for r,s in requests]
    result=evaluate(Engine(),ROWS,cfg,tmp_path,{},lambda *a,**k:None)
    assert result['accuracy']==1 and result['prompts']==2
    assert evaluate(Engine(),ROWS,cfg,tmp_path,{},lambda *a,**k:None)==result
    with pytest.raises(ValueError):
        evaluate(Engine(),ROWS,cfg,tmp_path,{'changed':1},lambda *a,**k:None)


def tiny_visual_model():
    transformers = pytest.importorskip('transformers')
    peft = pytest.importorskip('peft')
    torch.manual_seed(3)
    config = transformers.Qwen3_5Config(
        text_config=dict(vocab_size=64,hidden_size=32,intermediate_size=64,
            num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,head_dim=16,
            linear_num_key_heads=2,linear_num_value_heads=2,linear_key_head_dim=8,linear_value_head_dim=8,
            layer_types=['linear_attention','full_attention'],
            rope_parameters=dict(rope_type='default',rope_theta=10000.,partial_rotary_factor=1.,mrope_section=[2,3,3]),
            eos_token_id=2,pad_token_id=0,bos_token_id=1),
        vision_config=dict(depth=1,hidden_size=32,intermediate_size=64,num_heads=2,
            out_hidden_size=32,patch_size=2,spatial_merge_size=2,temporal_patch_size=2),
        image_token_id=62,video_token_id=63,vision_start_token_id=60,vision_end_token_id=61)
    base=transformers.AutoModelForMultimodalLM.from_config(config)
    targets=[n for n,m in base.named_modules() if isinstance(m,torch.nn.Linear) and '.language_model.layers.' in n]
    model=peft.get_peft_model(base,peft.LoraConfig(r=2,lora_alpha=4,target_modules=targets,
                                                lora_dropout=0.,bias='none',task_type='CAUSAL_LM'))
    items=[]
    for tail in ([5],[5,7]):
        ids=torch.tensor([[1,60,62,61,*tail]])
        types=torch.zeros_like(ids); types[0,2]=1
        items.append(dict(input_ids=ids,attention_mask=torch.ones_like(ids),
                          mm_token_type_ids=types,pixel_values=torch.randn(4,24),image_grid_thw=torch.tensor([[1,2,2]])))
    return model, items


def test_tiny_qwen_visual_batch_generation_matches_image_scoring():
    import transformers
    model, items = tiny_visual_model()
    batch=pack_inputs(items,0)
    model.eval()
    with torch.no_grad():
        generated=model.generate(**batch,generation_config=transformers.GenerationConfig(
            do_sample=False,max_new_tokens=3,eos_token_id=2,pad_token_id=0,return_dict_in_generate=True,output_scores=True))
    for i,item in enumerate(items):
        completion=generated.sequences[i,batch['input_ids'].shape[1]:]
        from dig_rl.runtime import trim_completion
        completion,_=trim_completion(completion,{2})
        sequence=torch.cat([item['input_ids'][0],completion]).unsqueeze(0)
        media={k:v for k,v in item.items() if k not in ('input_ids','attention_mask')}
        actual=visual_completion_logprobs(model,sequence,item['input_ids'].shape[1],media)
        expected=torch.stack([score[i].log_softmax(-1)[token] for score,token in zip(generated.scores,completion)])
        torch.testing.assert_close(actual,expected,atol=3e-5,rtol=3e-5)
        (-actual.sum()).backward()
    grads=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum()) for g in grads)>0


@pytest.mark.parametrize('visual',[False,True])
def test_chunked_vocabulary_projection_preserves_all_adapter_gradients(visual):
    model, items = tiny_visual_model()
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()
    item = items[0] if visual else dict(input_ids=torch.tensor([[1,5,6]]))
    prompt = item['input_ids'].shape[1]
    sequence = torch.cat([item['input_ids'],torch.tensor([[5,6]*70])],dim=1)
    media={k:v for k,v in item.items() if k not in ('input_ids','attention_mask')}
    widths=[]
    hook=model.get_base_model().lm_head.register_forward_pre_hook(lambda m,a:widths.append(a[0].shape[1]))
    actual=visual_completion_logprobs(model,sequence,prompt,media)
    # Include a negative DiG coefficient and fixed loss scaling.
    (actual.sum() * -.3/2048).backward()
    actual_grads={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
    hook.remove()
    assert max(widths)<=64 and len(widths)>2
    model.zero_grad(set_to_none=True)
    kwargs=dict(media)
    if 'mm_token_type_ids' in kwargs:
        kwargs['mm_token_type_ids']=torch.nn.functional.pad(kwargs['mm_token_type_ids'],(0,sequence.shape[1]-1-prompt))
    logits=model(input_ids=sequence[:,:-1],attention_mask=torch.ones_like(sequence[:,:-1]),**kwargs,use_cache=False).logits[:,prompt-1:]
    expected=-torch.nn.functional.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),sequence[:,prompt:].reshape(-1),reduction='none')
    (expected.sum() * -.3/2048).backward()
    torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-5)
    for n,p in model.named_parameters():
        if n in actual_grads:
            torch.testing.assert_close(actual_grads[n],p.grad,atol=2e-6,rtol=2e-4)


def test_partial_jsonl_tail_is_repaired_only_by_writer(tmp_path):
    from dig_rl.round2_train import records
    p=tmp_path/'records.jsonl'
    data=b'{"id":"a"}\n{"broken":"\xe4'
    p.write_bytes(data)
    assert records(p)==[dict(id='a')] and p.read_bytes()==data
    assert records(p,repair=True)==[dict(id='a')]
    assert p.read_bytes()==b'{"id":"a"}\n'
    assert list(tmp_path.glob('*.interrupted-*'))[0].read_bytes()==data


def test_scheduler_profile_decision_uses_aggregate_work_and_memory(tmp_path):
    spec=importlib.util.spec_from_file_location('round2_runner',Path(__file__).resolve().parents[1]/'scripts/round2.py')
    runner=importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    from dig_rl.io import write_json
    for dataset in DATASETS:
        write_json(tmp_path/'memorycheck'/f'{dataset}.json',dict(passed=True,peak_reserved_gib=35 if dataset=='math' else 10))
        write_json(tmp_path/'profile/profile1'/f'{dataset}_rloo.json',dict(generated_tokens=1000,seconds=10,active_sequences=16))
        for arm in ('rloo','dig'):
            memory=40 if dataset=='math' else 20
            duration=20 if dataset=='geo3k' else 12
            write_json(tmp_path/'profile/profile2'/f'{dataset}_{arm}.json',dict(generated_tokens=1000,
                started=0,finished=duration,peak_reserved_gib=memory))
    runner.profile_decision(tmp_path,{})
    decision=json.loads((tmp_path/'profile_decision.json').read_text())
    assert decision['gsm8k']['slots']==2
    assert decision['math']['slots']==1  # Combined memory, not individual memory.
    assert decision['geo3k']['slots']==1  # Two slower workers offer no throughput gain.


@pytest.mark.parametrize('slots',[1,2])
def test_scheduler_enforces_per_gpu_slots(tmp_path, monkeypatch, slots):
    spec=importlib.util.spec_from_file_location('round2_scheduler',Path(__file__).resolve().parents[1]/'scripts/round2.py')
    runner=importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    active={str(i):0 for i in range(4)}; peak=dict(active); starts=[]
    class Process:
        def __init__(self, cmd, **kwargs):
            self.gpu=kwargs['env']['CUDA_VISIBLE_DEVICES']; self.polls=0
            active[self.gpu]+=1; peak[self.gpu]=max(peak[self.gpu],active[self.gpu]); starts.append(cmd)
        def poll(self):
            self.polls+=1
            if self.polls==2:
                active[self.gpu]-=1
                return 0
            return None
        def terminate(self):
            raise AssertionError('A completed worker must not be killed')
    monkeypatch.setattr(runner.subprocess,'Popen',Process)
    monkeypatch.setattr(runner.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout=''))
    monkeypatch.setattr(runner.time,'sleep',lambda n:None)
    plan=dict(jobs=jobs({d:Protocol(d) for d in DATASETS}))
    runner.launch(tmp_path,'train',plan,{d:slots for d in DATASETS})
    assert len(starts)==8 and all(v==slots for v in peak.values()) and not any(active.values())
    assert (tmp_path/'phase_timing.jsonl').exists()
