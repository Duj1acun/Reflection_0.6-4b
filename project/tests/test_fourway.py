import json
from pathlib import Path

import pytest

from fourway.contracts import config, judgment, rank_key, summarize, token_runs
from fourway.configuration import build
from fourway.prepare import clean
from fourway.scoring import score_inline

def row(problem, gold="1", source="x"):
    return dict(problem=problem, gold=gold, source_id=source, uid=problem)

def test_clean_preserves_math_operators_and_quarantines_conflicts():
    train = [row("z^4 + z^2 + 1"), row("z^4 - z^2 + 1"), row(" repeated "),
             row("repeated"), row("conflict", "1"), row("conflict", "2"), row("validation")]
    good, excluded = clean(train, [row("validation")])
    assert len(good) == 3
    assert {r["exclusion_reason"] for r in excluded} == {"conflicting_labels", "duplicate_same_label", "validation_exact_overlap"}
    assert good[0]["problem"] != good[1]["problem"]

def test_validation_reports_coverage_separately_from_mean():
    records = []
    for uid, values in [("a", [1,1,1,1]), ("b", [1,0,0,0])]:
        for i,v in enumerate(values):
            records.append(dict(uid=uid, dataset="fixture", sample_index=i, correct=v,
                                score_status="ok", generated_tokens=10, truncated=False))
    s = summarize(records, {"fixture": 2})
    assert s["macro_pass_at_4"] == 1
    assert s["macro_mean_at_4"] == 5/8
    with pytest.raises(ValueError):
        summarize(records[:-1], {"fixture": 2})
    records[-1]["sample_index"] = 0
    with pytest.raises(ValueError):
        summarize(records, {"fixture": 2})

def test_best_ranking_is_macro_then_mean_then_earlier():
    s = dict(macro_pass_at_4=.7, macro_mean_at_4=.5, step=50)
    assert rank_key(s) > rank_key(dict(s, step=100))
    assert rank_key(dict(s, macro_mean_at_4=.6, step=100)) > rank_key(s)

def test_feedback_mask_and_judgment_contract():
    assert token_runs([1,1,0,0,1,0,1,1,0]) == [(0,2),(4,5),(6,8)]
    assert judgment("Checks\nThe answer is wrong.\n\n") == "wrong"
    assert judgment("The answer is correct. Actually, not sure") is None
    assert judgment("The answer is incorrect.") is None

def test_request_rngs_are_independent_and_validation_does_not_consume_training_rng():
    from types import SimpleNamespace
    from fourway.sampling import request_params
    original = SimpleNamespace(seed=None, n=1)
    first = request_params(original, 4, 1234)
    assert len({p.seed for p in first}) == 4
    assert original.seed is None
    request_params(original, 8, 9876, stage=1)
    assert [p.seed for p in request_params(original, 4, 1234)] == [p.seed for p in first]
    assert {p.seed for p in first}.isdisjoint(p.seed for p in request_params(original, 4, 1234, rank=1))

@pytest.mark.parametrize("text,gold,correct", [
    (r"\boxed{-3}", "-3", 1), (r"\boxed{x=5}", "5", 1),
    (r"\boxed{6}", "5", 0), (r"<reflection>So far I have \boxed{5}. However it is wrong.</reflection>", "5", 0),
    (r"<reflection>So far I have 6. However it is wrong.</reflection>\boxed{5}", "5", 1),
    (r"\boxed{(0,1)}", "[0,1]", 0), (r"\boxed{5} final: \boxed{", "5", 0),
])
def test_shared_judge(text, gold, correct):
    r = score_inline(text, gold)
    assert r["score_status"] not in {"exception", "timeout"}
    assert r["correct"] == correct

def test_resolved_configs_have_equal_shared_parameters():
    c = config()
    configs = [build(c,m,stage) for m,stage in [("grpo",1),("reflect",1),("pag",1),("score",1),("score",2)]]
    for x in configs:
        assert x.data.train_batch_size == 64
        assert x.actor_rollout_ref.actor.ppo_mini_batch_size == 64
        assert x.actor_rollout_ref.actor.entropy_coeff == 0
        assert x.actor_rollout_ref.rollout.val_kwargs.n == 4
        assert x.trainer.test_freq == 50
        assert x.actor_rollout_ref.rollout.val_kwargs.temperature == .7
        assert x.trainer.n_gpus_per_node == 2
        assert x.data.truncation == "error"
        assert x.data.seed == 1234
    assert configs[-2].trainer.total_training_steps == 250
    assert configs[-1].trainer.total_training_steps == 500
    assert configs[-1].trainer.resume_from_path.endswith("global_step_250")
    assert configs[-1].trainer.resume_mode == "resume_path"
    assert configs[-2].algorithm.kl_ctrl.kl_coef_turn1 == .1
    assert configs[-1].algorithm.kl_ctrl.kl_coef_turn1 == 0
    assert all(sum(v)==8192 for v in c["stage_tokens"].values())

def test_stateful_sampler_resume_keeps_shared_question_order():
    import torch
    from torchdata.stateful_dataloader import StatefulDataLoader
    def loader():
        data = list(range(177))
        sampler = torch.utils.data.RandomSampler(data, generator=torch.Generator().manual_seed(1234))
        return StatefulDataLoader(data, batch_size=64, sampler=sampler, num_workers=0, drop_last=True)
    a = loader()
    it = iter(a)
    next(it)
    state = a.state_dict()
    expected = next(it)
    b = loader()
    b.load_state_dict(state)
    assert torch.equal(next(iter(b)), expected)

def test_prepared_views_have_same_ids_and_no_leakage():
    import pyarrow.parquet as pq
    c = config()
    root = Path(c["output"])/"data"
    if not (root/"manifest.json").exists():
        pytest.skip("Run prepare first")
    reference = None
    for method in c["stage_tokens"]:
        rows = pq.read_table(root/f"{method}.train.parquet").to_pylist()
        ids = [(r["extra_info"]["index"], r["reward_model"]["ground_truth"]) for r in rows]
        assert len(ids) == 17133
        if reference is None:
            reference = ids
        assert ids == reference
        val = pq.read_table(root/f"{method}.validation.parquet").to_pylist()
        assert not {x[0] for x in ids} & {r["extra_info"]["index"] for r in val}
        assert all(set(r) == {"prompt", "reward_model", "data_source", "ability", "extra_info"} for r in rows)
