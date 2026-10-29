import json

import pytest

from verl.workers.reward_manager.naive import NaiveRewardManager


def item():
    return {"data_source": "fixture", "response_str": "Final answer: 2",
            "ground_truth": "2", "extra_info": {"index": "fixture-question"}}


def failure(**kwargs):
    raise RuntimeError("fixture judge timed out after retries")


def test_fourway_judge_failure_is_saved_and_propagated(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"output": str(tmp_path)}))
    monkeypatch.setenv("FOURWAY_CONFIG", str(path))
    monkeypatch.setenv("FOURWAY_METHOD", "grpo")
    manager = NaiveRewardManager(None, 0, compute_score=failure)
    with pytest.raises(RuntimeError, match="refusing zero reward") as exc:
        manager._score_items([item()])
    assert isinstance(exc.value.__cause__, RuntimeError)
    records = list((tmp_path / "diagnostics/reward_errors").glob("grpo_*.json"))
    assert len(records) == 1
    evidence = json.loads(records[0].read_text())
    assert evidence["response_text"] == item()["response_str"]
    assert evidence["ground_truth"] == "2"
    assert "fixture judge timed out" in evidence["traceback"]


def test_valid_wrong_answer_remains_zero_and_legacy_behavior_is_preserved(monkeypatch):
    monkeypatch.setenv("FOURWAY_CONFIG", "unused-for-valid-score")
    manager = NaiveRewardManager(None, 0, compute_score=lambda **kwargs: 0.0)
    assert manager._score_items([item()]) == [0.0]
    monkeypatch.delenv("FOURWAY_CONFIG")
    manager.compute_score = failure
    assert manager._score_items([item()]) == [{"score": 0.0, "reward_score_exception": 1.0}]
