import json
from types import SimpleNamespace

import pytest

from fourway.resilience import BatchJudgeGuard, is_judge_error
from fourway.scoring import JudgeError


@pytest.fixture
def setup_guard(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"output": str(tmp_path), "judge_failure_policy": "skip_batch",
                                "judge_max_consecutive_failures": 3}))
    monkeypatch.setenv("FOURWAY_CONFIG", str(path))
    monkeypatch.setenv("FOURWAY_METHOD", "grpo")
    trainer = SimpleNamespace(global_steps=5, config=SimpleNamespace(
        trainer=SimpleNamespace(default_local_dir=str(tmp_path))))
    logs = []
    logger = SimpleNamespace(log=lambda **kwargs: logs.append(kwargs))
    return trainer, logger, tmp_path


def guard(setup):
    trainer, logger, _ = setup
    return BatchJudgeGuard(trainer, logger, {"extra_info": [{"index": "q1"}]}, 0)


def test_skip_whole_batch_without_update_or_step_increment(setup_guard):
    trainer, _, root = setup_guard
    updates = []
    for failed in (True, False):
        with guard(setup_guard) as current:
            if failed:
                raise JudgeError("fixture score timeout")
            current.mark_updating()
            updates.append(trainer.global_steps)
        if current.skipped:
            continue
        trainer.global_steps += 1
    assert updates == [5] and trainer.global_steps == 6
    assert trainer._fourway_skipped_batches == 1
    assert trainer._fourway_consecutive_judge_failures == 0
    records = list((root / "diagnostics").glob("skipped_batch_*.json"))
    record = json.loads(records[0].read_text())
    assert record["question_ids"] == ["q1"] and record["action"] == "skip_batch"
    assert "fixture score timeout" in record["traceback"]


def test_stop_on_third_consecutive_failure_and_reset_on_success(setup_guard):
    trainer, _, _ = setup_guard
    for _ in range(2):
        with guard(setup_guard):
            raise JudgeError("fixture")
    with guard(setup_guard):
        pass
    assert trainer._fourway_consecutive_judge_failures == 0
    for _ in range(2):
        with guard(setup_guard):
            raise JudgeError("fixture")
    with pytest.raises(JudgeError, match="3 consecutive"):
        with guard(setup_guard):
            raise JudgeError("fixture")
    assert trainer.global_steps == 5


def test_other_errors_and_failures_after_updates_are_never_skipped(setup_guard):
    with pytest.raises(RuntimeError, match="CUDA"):
        with guard(setup_guard):
            raise RuntimeError("CUDA error")
    with pytest.raises(JudgeError, match="after update"):
        with guard(setup_guard) as current:
            current.mark_updating()
            raise JudgeError("after update")


def test_wrapped_and_ray_judge_errors_are_recognized(setup_guard):
    from ray.exceptions import RayTaskError
    remote = RayTaskError("reward", "fixture remote traceback", JudgeError("fixture"))
    assert is_judge_error(remote)
    with guard(setup_guard) as current:
        try:
            raise JudgeError("fixture")
        except JudgeError as exc:
            raise RuntimeError("reward manager wrapper") from exc
    assert current.skipped
    assert not is_judge_error(RuntimeError("Unresolved math judge error"))


def test_strict_policy_remains_available(setup_guard, monkeypatch):
    import fourway.resilience as module
    monkeypatch.setattr(module, "config", lambda: {"judge_failure_policy": "raise"})
    with pytest.raises(JudgeError):
        with guard(setup_guard):
            raise JudgeError("fixture")


def test_reward_failure_does_not_replay_batch(setup_guard):
    from verl.trainer.ppo.reward import compute_reward
    calls = []

    def reward(data, return_dict=False):
        calls.append(return_dict)
        raise JudgeError("fixture")

    with pytest.raises(JudgeError):
        compute_reward(None, reward)
    assert calls == [True]


def test_failed_validation_has_no_accuracy_and_stops_after_limit(setup_guard, monkeypatch):
    import fourway.runtime as module
    trainer, _, root = setup_guard
    trainer.config.trainer = {"default_local_dir": str(root), "val_only": False}
    from omegaconf import OmegaConf
    trainer.config.trainer = OmegaConf.create(trainer.config.trainer)

    def failed(self):
        raise JudgeError("fixture validation failure")

    monkeypatch.setattr(module, "_validate_complete", failed)
    for _ in range(2):
        result = module.validate(trainer)
        assert result["val/judge_failed"] == 1
        assert all("pass_at" not in key and "mean_at" not in key for key in result)
    assert (root / "validation/step_0005.failed.json").exists()
    assert not (root / "validation/step_0005.summary.json").exists()
    with pytest.raises(JudgeError, match="3 consecutive failed validations"):
        module.validate(trainer)
    monkeypatch.setattr(module, "_validate_complete", lambda self: {"val/macro_pass_at_4": 1})
    assert module.validate(trainer)["val/macro_pass_at_4"] == 1
    assert trainer._fourway_consecutive_validation_failures == 0
    assert trainer._fourway_failed_validation_step is None


def test_explicit_evaluation_remains_strict(setup_guard, monkeypatch):
    import fourway.runtime as module
    from omegaconf import OmegaConf
    trainer, _, root = setup_guard
    trainer.config.trainer = OmegaConf.create({"default_local_dir": str(root), "val_only": True})

    def failed(self):
        raise JudgeError("fixture")

    monkeypatch.setattr(module, "_validate_complete", failed)
    with pytest.raises(JudgeError):
        module.validate(trainer)


@pytest.mark.parametrize("repo", ["/root/reflection_grpo_clean", "/root/Policy-As-GenVerifier"])
def test_native_training_loops_continue_to_next_batch_before_updates(setup_guard, monkeypatch, repo):
    # Execute each native fit method with CPU stubs. Importing both incompatible
    # verl packages in one interpreter would otherwise exercise only one loop.
    import ast
    import os
    from contextlib import nullcontext
    from pathlib import Path
    from omegaconf import OmegaConf
    import verl.utils.tracking as tracking

    path = Path(repo) / "verl/trainer/ppo/ray_trainer.py"
    if not path.exists():
        pytest.skip("Native repository unavailable")
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RayPPOTrainer")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "fit")
    scope = {"os": os, "_timer": lambda *args: nullcontext(),
             "tqdm": lambda **kwargs: SimpleNamespace(update=lambda n: None, close=lambda: None)}

    class Batch:
        def __init__(self, row):
            self.non_tensor_batch = {}
            self.meta_info = {}
        def pop(self, **kwargs):
            return self

    scope["DataProto"] = SimpleNamespace(from_single_dict=Batch)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    trainer, logger, root = setup_guard
    monkeypatch.setattr(tracking, "Tracking", lambda **kwargs: logger)
    trainer.config = OmegaConf.create({"data": {"seed": 1}, "trainer": {
        "default_local_dir": str(root), "total_epochs": 1, "project_name": "test",
        "experiment_name": "test", "logger": [], "val_before_train": False}})
    trainer.val_reward_fn = None
    trainer.total_training_steps = 500
    trainer._load_checkpoint = lambda: None
    trainer._score_baseline_enabled = lambda: False
    trainer.train_dataloader = [{"extra_info": [{"index": str(i)}]} for i in range(3)]
    seen = []

    class ReachedNextBatch(RuntimeError):
        pass

    def generate(*args, **kwargs):
        seen.append(trainer.global_steps)
        if len(seen) <= 2:
            raise JudgeError("fixture rollout judge failure")
        raise ReachedNextBatch()

    trainer.traj_collector = SimpleNamespace(multi_turn_loop=generate)
    trainer.envs = None
    trainer.actor_rollout_wg = SimpleNamespace(generate_sequences=generate)
    with pytest.raises(ReachedNextBatch):
        scope["fit"](trainer)
    assert seen == [1, 1, 1]
    assert trainer._fourway_skipped_batches == 2


def _stalled_scorer(text, gold, dataset):
    import signal
    import time
    from pathlib import Path
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    Path(text).write_text("started")
    time.sleep(60)


def test_hard_timeout_kills_workers_and_next_judgment_recovers(setup_guard, monkeypatch):
    import time
    import fourway.scoring as module
    from fourway.math_scoring import SCORER_VERSION
    _, _, root = setup_guard
    path = root / "config.json"
    settings = json.loads(path.read_text())
    settings["scorer_version"] = SCORER_VERSION
    path.write_text(json.dumps(settings))
    original = module.score_inline
    module._reset()
    monkeypatch.setattr(module, "score_inline", _stalled_scorer)
    monkeypatch.setattr(module, "_WORKER_TIMEOUT_SECONDS", 1)
    started = time.monotonic()
    try:
        marker = root / "worker_started"
        with pytest.raises(JudgeError):
            module.judge(str(marker), "2")
        assert time.monotonic() - started < 10
        assert marker.exists()
        assert module._pool is None
        assert list((root / "diagnostics/reward_errors").glob("judge_*.json"))
        monkeypatch.setattr(module, "score_inline", original)
        monkeypatch.setattr(module, "_WORKER_TIMEOUT_SECONDS", 15)
        result = module.judge(r"\boxed{2}", "2")
        assert result["correct"] == 1 and result["score_status"] == "ok"
    finally:
        module._reset()
