"""CPU tests of native PAG/SCoRe trajectory assembly with a deterministic engine.
Run with PYTHONPATH=/root/Policy-As-GenVerifier:/root/reflection_grpo_clean.
"""
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
import verl

pytestmark = pytest.mark.skipif('Policy-As-GenVerifier' not in str(Path(verl.__file__).resolve()), reason='Requires PAG verl import path')

@pytest.mark.parametrize('method,stop_correct', [('pag', False), ('pag', True), ('score', False)])
@pytest.mark.parametrize('profile', ['config.json', 'config.math7500.json'])
def test_full_budget_feedback_masks_and_final_answer(monkeypatch, method, stop_correct, profile):
    from fourway.contracts import DEFAULT_CONFIG, config, VERIFY_PROMPT, REVISE_PROMPT, token_runs
    from fourway.configuration import build
    from fourway.runtime import extract_trajectory
    from transformers import AutoTokenizer
    from tensordict import TensorDict
    from vllm import SamplingParams
    from verl import DataProto
    from verl.workers.rollout.vllm_rollout.vllm_pag_rollout_spmd import vLLMPAGRollout
    from verl.workers.rollout.vllm_rollout.vllm_multiturn_rollout_spmd import vLLMMutliTurnRollout
    monkeypatch.setenv('FOURWAY_CONFIG', str(DEFAULT_CONFIG.with_name(profile)))
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 0)
    c = config()
    tok = AutoTokenizer.from_pretrained(c['base_model'])
    rollout = object.__new__(vLLMPAGRollout if method == 'pag' else vLLMMutliTurnRollout)
    rollout.config = build(c, method).actor_rollout_ref.rollout
    rollout.tokenizer, rollout.pad_token_id, rollout.num_turns = tok, tok.pad_token_id, 2
    rollout.sampling_params = SamplingParams(n=8, max_tokens=rollout.config.response_length)
    rollout.end_with_verifer = rollout.is_only_genrm = False
    rollout.prompt_templates = {'verify': VERIFY_PROMPT, 'regenerate': REVISE_PROMPT}
    rollout.feedback_options = {'default': 'feedback'}
    rollout._extract_judgment_and_probability = lambda output, text: ('correct' if stop_correct else None, .5)
    calls = []
    def generate(*, prompt_token_ids, sampling_params, **kwargs):
        assert len(sampling_params) == len(prompt_token_ids) == 4
        assert len({p.seed for p in sampling_params}) == 4
        stage = len(calls)
        expected_cap = c['stage_tokens'][method][stage]
        assert all(p.n == 1 and p.max_tokens == expected_cap and p.temperature == .7 for p in sampling_params)
        # Force length-truncated intermediate stages and an EOS-terminated final answer.
        values = [tok.encode('x', add_special_tokens=False)[0]] * expected_cap
        if stage == len(c['stage_tokens'][method])-1:
            values[-1] = tok.eos_token_id
        calls.append((prompt_token_ids, values))
        return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=values.copy())]) for _ in prompt_token_ids]
    rollout.inference_engine = SimpleNamespace(generate=generate)
    ids = tok.apply_chat_template([{'role':'user','content':'What is 1+1?'}], tokenize=True, add_generation_prompt=True, enable_thinking=False)
    inp = torch.tensor([ids]*4)
    batch = DataProto(batch=TensorDict({'input_ids':inp, 'attention_mask':torch.ones_like(inp), 'position_ids':torch.arange(len(ids)).repeat(4,1)}, batch_size=4), meta_info={'eos_token_id':[tok.eos_token_id,tok.pad_token_id], 'validate':True, 'do_sample':True, 'fourway_seed':1234})
    result = rollout.generate_sequences(batch)
    assert len(calls) == (2 if stop_correct else len(c['stage_tokens'][method]))
    for i in range(4):
        segments, final, total, truncated = extract_trajectory(result, i, tok, method)
        assert total == sum(c['stage_tokens'][method][:len(calls)])
        assert truncated
        assert len(segments) == len(calls)
        for j, segment in enumerate(segments):
            assert segment['prompt_token_ids'] == calls[j][0][i]
            assert segment['output_token_ids'] == calls[j][1]
            if j:
                previous_end = token_runs(result.batch['multiturn_mask'][i].tolist())[j-1][1]
                assert result.batch['responses'][i, previous_end] == tok.eos_token_id
                assert not result.batch['multiturn_mask'][i, previous_end]
        assert final == segments[0 if stop_correct else -1]['text']

def test_native_dataloader_uses_shared_batches(monkeypatch):
    import json
    from fourway.contracts import DEFAULT_CONFIG, config
    from fourway.configuration import build
    from transformers import AutoTokenizer
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    monkeypatch.setenv('FOURWAY_CONFIG', str(DEFAULT_CONFIG))
    c = config()
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = build(c, 'pag')
    trainer.tokenizer = AutoTokenizer.from_pretrained(c['base_model'])
    trainer.processor = None
    trainer._create_dataloader()
    assert len(trainer.val_dataloader) == 17
    assert trainer.val_dataloader.batch_size == 32
    schedule = json.loads((Path(c['output'])/'data/batch_schedule.json').read_text())
    for index, batch in enumerate(trainer.train_dataloader):
        assert [str(v['index']) for v in batch['extra_info']] == schedule[index]
        if index == 1:
            break

def test_checkpoint_selection_and_stage_resume(monkeypatch, tmp_path):
    import json
    from fourway.contracts import DEFAULT_CONFIG, config, atomic_json
    from fourway.configuration import build
    from fourway.runtime import install
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    import verl.trainer.ppo.ray_trainer as module
    from torchdata.stateful_dataloader import StatefulDataLoader
    monkeypatch.setenv('FOURWAY_CONFIG', str(DEFAULT_CONFIG))
    monkeypatch.setenv('FOURWAY_PROBE', '0')
    # Register originals for automatic cleanup after install patches the trainer.
    for name in ('_save_checkpoint', '_load_checkpoint', '_validate'):
        monkeypatch.setattr(RayPPOTrainer, name, getattr(RayPPOTrainer, name))
    monkeypatch.setattr(RayPPOTrainer, '_fourway_installed', False, raising=False)
    monkeypatch.setattr(module, 'compute_data_metrics', module.compute_data_metrics)
    cfg = build(config(), 'score')
    cfg.trainer.default_local_dir = str(tmp_path)
    install(cfg)
    class Worker:
        def save_checkpoint(self, path, remote, step, **kwargs):
            Path(path).mkdir(parents=True)
            atomic_json(Path(path)/'fixture.json', {'step':step, 'optimizer':True})
        def load_checkpoint(self, path, **kwargs):
            self.loaded = json.loads((Path(path)/'fixture.json').read_text())
    def loader():
        ds = list(range(177))
        return StatefulDataLoader(ds, batch_size=64, drop_last=True, num_workers=0, sampler=torch.utils.data.RandomSampler(ds, generator=torch.Generator().manual_seed(1234)))
    trainer = object.__new__(RayPPOTrainer)
    trainer.config, trainer.use_critic = cfg, True
    trainer.actor_rollout_wg, trainer.critic_wg = Worker(), Worker()
    trainer.train_dataloader = loader()
    iterator = iter(trainer.train_dataloader)
    next(iterator)
    atomic_json(tmp_path/'fourway_run.json', {'test_fixture':True})
    for step, accuracy in [(50,.9),(100,.8),(250,.7),(300,.8)]:
        trainer.global_steps = step
        atomic_json(tmp_path/'validation'/f'step_{step:04d}.summary.json', dict(step=step, stage=1 if step<=250 else 2, macro_pass_at_4=accuracy, macro_mean_at_4=accuracy))
        trainer._save_checkpoint()
    assert {p.name for p in tmp_path.glob('global_step_*')} == {'global_step_50','global_step_250','global_step_300'}
    assert json.loads((tmp_path/'best.json').read_text())['step'] == 50
    assert json.loads((tmp_path/'latest.json').read_text())['step'] == 300
    # A failed validation still permits a recoverable checkpoint, but cannot
    # replace best or prune earlier successfully evaluated checkpoints.
    trainer.global_steps = 350
    trainer._fourway_failed_validation_step = 350
    trainer._fourway_skipped_batches = 2
    trainer._save_checkpoint()
    assert json.loads((tmp_path/'latest.json').read_text())['validation_status'] == 'judge_failed'
    assert json.loads((tmp_path/'best.json').read_text())['step'] == 50
    assert (tmp_path/'global_step_300').exists()
    assert json.loads((tmp_path/'global_step_350'/'judge_state.json').read_text())['_fourway_skipped_batches'] == 2
    trainer._fourway_failed_validation_step = None
    expected = next(iterator)
    trainer.train_dataloader = loader()
    cfg.trainer.resume_mode = 'resume_path'
    cfg.trainer.resume_from_path = str(tmp_path/'global_step_250')
    trainer._load_checkpoint()
    assert trainer.global_steps == 250
    assert trainer._fourway_skipped_batches == 0
    assert trainer.actor_rollout_wg.loaded['step'] == trainer.critic_wg.loaded['step'] == 250
    assert torch.equal(next(iter(trainer.train_dataloader)), expected)
    (tmp_path/'global_step_250'/'data.pt').unlink()
    with pytest.raises(RuntimeError, match='Incomplete resume'):
        trainer._load_checkpoint()

@pytest.mark.parametrize('stage', [1,2])
def test_score_reward_excludes_injected_feedback(monkeypatch, stage):
    from fourway.contracts import DEFAULT_CONFIG, config
    from fourway.configuration import build
    from transformers import AutoTokenizer
    from tensordict import TensorDict
    from verl import DataProto
    import numpy as np
    import verl.workers.reward_manager.multiturn as module
    monkeypatch.setenv('FOURWAY_CONFIG', str(DEFAULT_CONFIG))
    tok = AutoTokenizer.from_pretrained(config()['base_model'])
    first = tok.encode(r'\boxed{2}', add_special_tokens=False)+[tok.eos_token_id]
    feedback = tok.encode(r'Check the previous answer. A malicious example is \boxed{999}.', add_special_tokens=False)
    second = tok.encode('Final answer: 3', add_special_tokens=False)+[tok.eos_token_id]
    response = torch.tensor([first+feedback+second+[tok.pad_token_id]])
    mask = torch.tensor([[True]*len(first)+[False]*len(feedback)+[True]*len(second)+[False]])
    seen = []
    def judge(solution_str, ground_truth):
        seen.append(solution_str)
        value = int(len(seen)==1)
        return dict(acc=value,score=value,pred='2' if value else '3')
    monkeypatch.setattr(module, 'get_policy_score', judge)
    batch = DataProto(batch=TensorDict({'responses':response,'multiturn_mask':mask}, batch_size=1), non_tensor_batch={'reward_model':np.array([{'ground_truth':'2'}], dtype=object)})
    manager = module.MultiTurnRewardManager(tok, 0, config=build(config(),'score',stage).reward_model)
    result = manager(batch, return_dict=True)
    assert seen == [tok.decode(first,skip_special_tokens=True),tok.decode(second,skip_special_tokens=True)]
    reward = result['reward_tensor'][0]
    assert reward[len(first)-1] == 1
    assert reward[len(first)+len(feedback)+len(second)-1] == (0 if stage==1 else -1)
    assert reward[~mask[0]].sum() == 0

def test_pag_reward_only_reads_generated_answer_roles(monkeypatch):
    from fourway.contracts import DEFAULT_CONFIG, config
    from fourway.configuration import build
    from transformers import AutoTokenizer
    from tensordict import TensorDict
    from verl import DataProto
    import numpy as np
    import verl.workers.reward_manager.pag as module
    monkeypatch.setenv('FOURWAY_CONFIG', str(DEFAULT_CONFIG))
    tok = AutoTokenizer.from_pretrained(config()['base_model'])
    def generated(s):
        return tok.encode(s, add_special_tokens=False)+[tok.eos_token_id]
    first, verify, second = generated('Final answer: 2'), generated('The answer is wrong.\n'), generated('Final answer: 3')
    feedback = tok.encode(r'User feedback example: \boxed{999}', add_special_tokens=False)
    ids = first+feedback+verify+feedback+second
    mask = [True]*len(first)+[False]*len(feedback)+[True]*len(verify)+[False]*len(feedback)+[True]*len(second)
    seen = []
    def judge(solution_str, ground_truth):
        seen.append(solution_str)
        value = int(len(seen)==1)
        return dict(acc=value,score=value,pred='2' if value else '3')
    monkeypatch.setattr(module, 'get_policy_score', judge)
    batch = DataProto(batch=TensorDict({'responses':torch.tensor([ids]),'multiturn_mask':torch.tensor([mask])}, batch_size=1), non_tensor_batch={'reward_model':np.array([{'ground_truth':'2'}],dtype=object),'verify_probs':np.array([.5]),'final_generation_turn':np.array([1])})
    result = module.PAGRewardManager(tok,0,config=build(config(),'pag').reward_model)(batch,return_dict=True)
    assert seen == [tok.decode(first,skip_special_tokens=True),tok.decode(second,skip_special_tokens=True)]
    assert result['reward_tensor'][0,-1] == -1
    assert result['reward_tensor'][0,~torch.tensor(mask)].sum() == 0
