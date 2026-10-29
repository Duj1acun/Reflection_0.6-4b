import json

import pytest

from fourway.math_scoring import SCORER_VERSION, score_sample


@pytest.mark.parametrize("pred,gold,correct", [
    ("0.5", r"\frac{1}{2}", 1), (r"\frac{2}{4}", "0.5", 1),
    ("0.33333333", r"\frac{1}{3}", 1), ("0.33", r"\frac{1}{3}", 0),
    ("0.500000009", r"\frac12", 1), ("0.50000002", r"\frac12", 0),
    (r"50\%", "0.5", 1), (r"50\%", "50", 0),
    ("17", r"17\%", 0), ("0.17", r"17\%", 1),
    ("100001", "100000", 0), ("100000.000000001", "100000", 0),
    ("070", "70", 1), ("x=9", "9", 1), ("9+4=13", "13", 1),
    ("x<9", "9", 0), ("x=1,y=9", "9", 0),
    (r"\sqrt{2}/2", r"\frac{1}{\sqrt{2}}", 1),
    ("b+a+c", "a+b+c", 1), ("b+a+c", "a-b+c", 0),
    (r"1-\cos^2 t", r"\sin^2 t", 1),
    (r"\{3,1,2\}", r"\{1,2,3\}", 1),
    ("3,1,2", "1,2,3", 1), ("1", r"\{1,2\}", 0),
    (r"\{1,2,3\}", r"\{1,2\}", 0),
    ("(2,1)", "(1,2)", 0), ("(1,234)", "1234", 0),
    ("[0,1]", "(0,1)", 0), ("[0,1)", "[0,1]", 0),
    (r"(2,12)\cup(12,\infty)", r"(2,12)\cup(12,102)", 0),
    (r"[-2,7]", r"x\in[-2,7]", 1),
    (r"\begin{pmatrix}1&2\\3&4\end{pmatrix}", r"\begin{pmatrix}1&3\\2&4\end{pmatrix}", 0),
    (r"\begin{pmatrix} -\frac{1}{3} \\ \frac{2}{3} \\ \frac{5}{3} \end{pmatrix}", r"\begin{pmatrix} -1/3 \\ 2/3 \\ 5/3 \end{pmatrix}", 1),
    (r"\frac{8}{49} \begin{pmatrix} 2 \\ 6 \\ 3 \end{pmatrix}", r"\begin{pmatrix} 16/49 \\ 48/49 \\ 24/49 \end{pmatrix}", 1),
    (r"\begin{pmatrix}1&-1\\1&1\end{pmatrix}", r"\begin{pmatrix}1&-1\\1&\phantom -1\end{pmatrix}", 1),
    (r"2007+\frac{\pi}{2}", r"2007+\frac\pi 2", 1),
    ("8.24", r"\approx 8.24 \text{ mph}", 1),
    (r"4 \text{ inches per second}", "4", 1),
    (r"4 \, \text{inches per second}", "4", 1),
    (r"\frac{1}{29322216}", r"\dfrac{1}{29,\!322,\!216}", 1),
    ("17", "{17}.", 1), ("7", r"7\text{ hours}.", 1),
    ("0.56", r"0.56 = 56\%", 1),
    ("E", r"\text{(E)},", 1), ("A", r"\text{(E)},", 0),
    ("43", "43_5.", 1), ("43_6", "43_5.", 0),
    ("A5", r"\text{A5}_{11}", 1), ("A5_{12}", r"\text{A5}_{11}", 0),
    ("9", r"9\text{ values}", 1), ("20", r"20\text{ days}", 1),
    ("240", r"240\text{ ways.}", 1), ("428", r"428\text{ square centimeters}", 1),
    ("a+b", "a+b+c", 0), ("2c", "2", 0),
    ("friday", r"\text{Friday}", 1), ("Tuesday", r"\text{Friday}", 0),
    ("D,A", r"\text{A,D}", 1), ("A", r"\text{A,D}", 0),
    ("BCAD", r"\text{CBAD}", 0), ("4", r"\mbox{four}", 1),
    (r"\emptyset", r"\text{empty}", 1), ("0", r"\text{empty}", 0),
    ("5:00", r"05\!:\!00", 1), ("04:00", r"05\!:\!00", 0),
    ("05:00", r"05\!:\!30", 0), ("5:99", "05:00", 0),
    ("12:00 am", "00:00", 1), ("1:00 pm", "13:00", 1),
    ("1:00 am", "13:00", 0), ("12:00 pm", "12:00", 1),
    (r"\frac{19}{60}", "19:60", 1),
    ("3:8", "9:24", 1), ("3:7", "9:24", 0),
    ("1/0", "1", 0), (r"\frac{}{}", "2", 0),
])
def test_math_answer_semantics_and_training_validation_agreement(pred, gold, correct):
    for dataset in (None, "math7500", "math500", "amc23"):
        result = score_sample((r"\boxed{" + pred + "}", gold, dataset))
        assert result["score_status"] not in {"timeout", "exception"}, result
        assert result["correct"] == correct, result


@pytest.mark.parametrize("text,correct", [
    (r"<reflection>The wrong answer is \boxed{5}</reflection>", 0),
    (r"\boxed{5}<reflection>That was wrong.</reflection>", 0),
    (r"<reflection>The wrong answer is 6.</reflection>\boxed{5}", 1),
    (r"\boxed{5} then \boxed{6}", 0),
    (r"\boxed{5} then \boxed{", 0),
    ("We considered 5 but cannot solve it.", 0),
    ("Final answer: $5$", 1),
])
def test_only_final_answer_gets_reward(text, correct):
    assert score_sample((text, "5", "math7500"))["correct"] == correct


def test_new_scorer_is_opt_in_and_legacy_version_is_unchanged(tmp_path, monkeypatch):
    from fourway.scoring import score_inline
    monkeypatch.delenv("FOURWAY_CONFIG", raising=False)
    assert score_inline(r"\boxed{5}", "5")["scorer_version"].startswith("paired-v4")
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"scorer_version": SCORER_VERSION}))
    monkeypatch.setenv("FOURWAY_CONFIG", str(path))
    assert score_inline(r"\boxed{50\%}", "50")["correct"] == 0
    assert score_inline(r"\boxed{5}", "5", "math500")["scorer_version"] == SCORER_VERSION


def test_unknown_parser_failures_remain_errors(monkeypatch):
    import fourway.math_scoring as module
    def failed(value):
        raise RuntimeError("parser infrastructure failure")
    monkeypatch.setattr(module, "parse_answer", failed)
    result = score_sample((r"\boxed{\sqrt{2}}", "3", "math7500"))
    assert result["score_status"] == "exception"
    assert "infrastructure failure" in result["score_error"]


@pytest.mark.parametrize("pred,gold", [
    (r"g(2) = \frac{2^{2^{2008}}}{2^{2^{2009}} - 1}", "2"),
    (r"2^{2^{2008}}", "2"),
    (r"\frac{1}{2^{2^{2008}}}", "0"),
    (r"2 + \frac{1}{2^{2^{2008}}}", "2"),
])
def test_huge_wrong_integer_answers_do_not_enter_simplify(pred, gold, monkeypatch):
    import sympy

    def forbidden(*args, **kwargs):
        raise AssertionError("huge integer arithmetic reached simplify")

    monkeypatch.setattr(sympy, "simplify", forbidden)
    for dataset in (None, "math7500", "math500", "amc23"):
        result = score_sample((r"\boxed{" + pred + "}", gold, dataset))
        assert result["score_status"] == "ok", result
        assert result["correct"] == 0, result


@pytest.mark.parametrize("pred,gold,correct", [
    ("2^3", "8", 1),
    ("(-2)^3", "-8", 1),
    ("2^{-3} * 16", "2", 1),
    # Collision at all three moduli must still undergo exact comparison.
    ("1000000007 * 1000000009 * 998244353 + 2", "2", 0),
    # A denominator divisible by the moduli is inconclusive, not unequal.
    (r"\frac{1000000007 * 1000000009 * 998244353}{1000000007 * 1000000009 * 998244353}", "1", 1),
    (r"\sqrt{4}", "2", 1),
    (r"\sqrt{2}", "2", 0),
])
def test_modular_check_preserves_exact_comparison(pred, gold, correct):
    result = score_sample((r"\boxed{" + pred + "}", gold, "math7500"))
    assert result["score_status"] == "ok", result
    assert result["correct"] == correct, result


def test_modular_check_declines_unsupported_or_unbounded_exponents():
    from fourway.math_scoring import _provably_not_integer, parse_answer

    for pred in (r"\sqrt{4}", r"2^{2^{2^{2008}}}",
                 r"\frac{1}{1000000007 * 1000000009 * 998244353}"):
        assert not _provably_not_integer(parse_answer(pred), 2)


def test_score_timeouts_remain_errors(monkeypatch):
    import fourway.math_scoring as module
    from rgrpo.math_score_light import MathScoreTimeout

    def failed(pred, gold):
        raise MathScoreTimeout()

    monkeypatch.setattr(module, "compare", failed)
    result = score_sample((r"\boxed{2^3}", "8", "math7500"))
    assert result["score_status"] == "timeout"


@pytest.mark.parametrize("pred,gold,correct,status", [
    (r"\frac{2^{2^{2008}}}{2^{2^{2008}}}", "1", 1, "ok"),
    (r"\frac{2^{2^{2008}}}{2^{2^{2008}}} + 1", "2", 1, "ok"),
    (r"2^{2^{2008}}-2^{2^{2008}}+2", "2", 1, "ok"),
    (r"\begin{pmatrix}1&2\\3\end{pmatrix}", "2", 0, "invalid_answer"),
    (r"\begin{pmatrix}1&2\end{pmatrix}+\begin{pmatrix}1\\2\end{pmatrix}", "2", 0, "invalid_answer"),
    (r"\det\begin{pmatrix}1&2\end{pmatrix}", "2", 0, "invalid_answer"),
    ("2", r"\frac{1}{0}", 0, "exception"),
    ("2", r"\begin{pmatrix}1&2\\3\end{pmatrix}", 0, "exception"),
])
def test_runtime_audit_regressions(pred, gold, correct, status):
    result = score_sample((r"\boxed{" + pred + "}", gold, "math7500"))
    assert result["score_status"] == status, result
    assert result["correct"] == correct, result


def test_math_verify_baseexception_timeout_is_reported(monkeypatch):
    import fourway.math_scoring as module
    from math_verify.errors import TimeoutException

    def failed(pred, gold):
        raise TimeoutException("fixture timeout")

    monkeypatch.setattr(module, "compare", failed)
    assert score_sample((r"\boxed{2}", "3", "math7500"))["score_status"] == "timeout"


def test_undefined_logarithm_does_not_fail_the_judge():
    result = score_sample((r"\boxed{\log_1 2}", "2", "math7500"))
    assert result["correct"] == 0
    assert result["score_status"] in {"ok", "invalid_answer"}, result


def test_extraction_failures_are_inside_error_boundary(monkeypatch):
    import fourway.math_scoring as module

    def failed(text, dataset):
        raise RuntimeError("fixture extraction failure")

    monkeypatch.setattr(module, "extract_final", failed)
    assert score_sample(("text", "3", "math7500"))["score_status"] == "exception"


def test_killable_worker_and_policy_reward_use_new_scorer(tmp_path, monkeypatch):
    import fourway.scoring as module
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"scorer_version": SCORER_VERSION, "output": str(tmp_path)}))
    monkeypatch.setenv("FOURWAY_CONFIG", str(path))
    module._reset()
    try:
        assert module.judge(r"\boxed{50\%}", "50", "math500")["correct"] == 0
        result = module.policy_score(r"\boxed{\sqrt{2}/2}", r"\frac{1}{\sqrt{2}}")
        assert result["score"] == 1 and result["scorer_version"] == SCORER_VERSION
        assert module.policy_score(r"\boxed{\frac{}{}}", "2")["score_status"] == "invalid_answer"
        result = module.policy_score(r"\boxed{g(2) = \frac{2^{2^{2008}}}{2^{2^{2009}} - 1}}", "2")
        assert result["score_status"] == "ok" and result["score"] == 0
    finally:
        module._reset()


def test_math_profile_has_equal_answer_caps_and_identical_question_views():
    from pathlib import Path
    import pyarrow.parquet as pq
    from fourway.contracts import config, generated_budget
    from fourway.configuration import build
    root = Path(__file__).resolve().parents[1]
    c = config(root / "experiments/four_methods/config.math7500.json")
    assert c["max_model_len"] == 16384 and c["max_prompt_length"] == 2048
    assert c["stage_tokens"]["grpo"] == [4096]
    assert c["stage_tokens"]["pag"] == [4096, 2048, 4096]
    assert c["stage_tokens"]["score"] == [4096, 4096]
    reference = None
    for method, budgets in c["stage_tokens"].items():
        assert generated_budget(c, method) == sum(budgets)
        x = build(c, method)
        assert x.actor_rollout_ref.rollout.max_model_len == 16384
        assert x.data.max_response_length == max(budgets)
        assert x.trainer.n_gpus_per_node == 2 and x.trainer.test_freq == 50
        assert x.data.val_batch_size == 128
        assert x.actor_rollout_ref.rollout.max_num_seqs == 32
        assert x.actor_rollout_ref.rollout.enforce_eager is False
        assert x.actor_rollout_ref.rollout.gpu_memory_utilization == .50
        assert x.actor_rollout_ref.rollout.max_num_batched_tokens == 8192
        p = Path(c["output"]) / "data" / f"{method}.train.parquet"
        if not p.exists():
            continue
        rows = pq.read_table(p).to_pylist()
        keys = [(r["extra_info"]["index"], r["reward_model"]["ground_truth"]) for r in rows]
        assert len(keys) == 7496 and len({k[0] for k in keys}) == 7496
        assert all(r["data_source"] == "math7500" for r in rows)
        if reference is None:
            reference = keys
        assert keys == reference


def test_math_cleaning_quarantines_missing_labels_and_verified_overlap():
    from fourway.prepare import clean
    def row(i, problem, gold):
        return dict(source_id=str(i), problem=problem, gold=gold, uid=problem)
    retained, removed = clean([row(0, "ok", "1"), row(1, "empty", ""),
                               row(304, "rephrased validation", "11")], [], {"304": "test/algebra/170.json"})
    assert len(retained) == 1
    assert {r["exclusion_reason"] for r in removed} == {"empty_question_or_label", "validation_verified_formatting_overlap"}
