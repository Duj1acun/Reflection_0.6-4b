"""Regression tests for the SFT prompt and disabled RL format reward gate."""
import ast
from pathlib import Path

import pytest


def test_prompt_matches_both_sft_stages():
    from fourway.contracts import REFLECT_PROMPT
    root = Path(__file__).resolve().parents[1]
    for stage in (1, 2):
        tree = ast.parse((root / f"training/sft/train_phase{stage}.py").read_text())
        node = next(n for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "SYSTEM_PROMPT" for t in n.targets))
        assert REFLECT_PROMPT == ast.literal_eval(node.value)


@pytest.mark.parametrize("body", [
    "", "<reflection>The sign is wrong. Recompute it.</reflection>",
    "<reflection>So far I have a result. However it is wrong.</reflection>",
    "<reflection> </reflection>", "<reflection>unclosed",
])
@pytest.mark.parametrize("correct", [False, True])
def test_disabled_gate_does_not_change_outcome_credit(monkeypatch, body, correct):
    from fourway.configuration import REFLECT_ENV
    from rgrpo import core_trajectory_grpo as reward
    for name in list(reward.os.environ):
        if name.startswith("TRAJ_GRPO_"):
            monkeypatch.delenv(name)
    for name, value in REFLECT_ENV.items():
        monkeypatch.setenv(name, value)
    assert REFLECT_ENV["TRAJ_GRPO_REQUIRE_REFLECTION_FORMAT"] == "0"
    monkeypatch.setattr(reward, "_reflection_format_valid",
                        lambda text: pytest.fail("Disabled format gate was called"))
    monkeypatch.setattr(reward, "compute_math_score", lambda **kwargs: float(correct))
    result = reward.compute_trajectory_grpo_score(
        generated=body + "\\boxed{2}", ground_truth="2",
        extra_info={"num_reflections": int(bool(body)), "n_iters": 2 if body else 1})
    assert result["format_reward"] == 1.0
    assert result["format_gate_zero"] == 0.0
    assert result["score"] == (1.0 if correct else 0.0 if body else -0.25)
