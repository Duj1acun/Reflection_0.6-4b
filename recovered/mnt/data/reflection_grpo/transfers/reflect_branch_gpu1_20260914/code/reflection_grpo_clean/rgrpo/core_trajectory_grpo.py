"""
Trajectory-GRPO reward aligned with the InftyThink+ RL recipe.

This module intentionally keeps the reward at the complete-trajectory level:

    R_task = 1[final answer is correct]
    R_eff  = 1 - ((n_iters - 1) / phi) ** 2
    R      = R_task * R_eff

R_eff is already a quadratic reflection-frequency penalty. An optional extra
linear penalty is available for ablations, but defaults to 0.

The GRPO advantage is then computed by veRL's standard `adv_estimator=grpo`
over the G trajectories sampled for the same prompt. The rollout side is the
existing `reflection_grpo_loop`, which produces one final response per sampled
trajectory and writes `num_reflections`, `n_iters`, `phi`, and
`terminated_by_cap` into `extra_info`.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from .core_hybrid_grpo_v2 import (
    REFL_PATTERN,
    _REFLECTION_CLOSE_RE,
    _REFLECTION_OPEN_RE,
    _reflection_format_valid,
    compute_math_score,
)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    try:
        return bool(int(value))
    except Exception:
        return bool(value)


def _efficiency_reward(n_iters: int, phi: int, free_reflections: Optional[int] = None) -> float:
    """Quadratic efficiency reward with a zero-cost grace for the first reflections.

    `free_reflections` reflections are charged nothing, so a model that reflects
    when it genuinely needs to is not taxed for the necessary correction; only
    *excess* reflections beyond the grace are penalised. With free=1 (default):
    n_iters=1 -> 1.0, n_iters=2 -> 1.0, n_iters=3 -> 1-(1/phi)^2, ...
    Set TRAJ_GRPO_EFF_FREE_REFLECTIONS=0 to recover the original
    1-((n_iters-1)/phi)^2 behaviour.
    """
    phi = max(1, int(phi))
    n_iters = max(1, int(n_iters))
    if free_reflections is None:
        free_reflections = _safe_int(
            os.environ.get("TRAJ_GRPO_EFF_FREE_REFLECTIONS", "1"), 1
        )
    free_reflections = max(0, int(free_reflections))
    chargeable_extra_iters = max(0, (n_iters - 1) - free_reflections)
    reward = 1.0 - (chargeable_extra_iters / phi) ** 2
    return max(0.0, min(1.0, float(reward)))


def _wrong_no_reflect_penalty(n: int, phi: int) -> float:
    """Penalty applied to wrong trajectories that never triggered reflection.

    Uses the same quadratic form as the efficiency reward but negated, evaluated
    at a fixed hypothetical iteration count n (default 2 or 3) rather than the
    actual n_iters (which is always 1 for no-reflection trajectories).

      penalty = -((n - 1) / phi)^2

    With phi=5: n=2 → -0.04, n=3 → -0.16.

    Rationale: a trajectory that failed without attempting reflection should be
    penalised proportionally to how many reflections it "could have tried", using
    the same scale as the efficiency cost, so the two opposing forces are
    dimensionally consistent.
    """
    phi = max(1, int(phi))
    n = max(1, int(n))
    return -(((n - 1) / phi) ** 2)


def compute_trajectory_grpo_score(
    solution_str: str = None,
    ground_truth: str = None,
    generated: str = None,
    data_source: str = None,
    extra_info: Optional[dict] = None,
    phi: Optional[int] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Return an InftyThink+-style scalar reward for one final trajectory.

    Environment knobs:
      TRAJ_GRPO_USE_EFF_REWARD=0 disables the efficiency multiplier.
      TRAJ_GRPO_EFF_FREE_REFLECTIONS sets how many reflections are charged
      nothing by the efficiency reward (default 1, so the first reflection is
      zero-cost; set 0 for the original 1-((n_iters-1)/phi)^2 penalty).
      TRAJ_GRPO_REWARD_PHI overrides the efficiency penalty horizon. This can
      be larger than the rollout max iterations when the local context window
      forces a smaller rollout cap.
      TRAJ_GRPO_REQUIRE_REFLECTION_FORMAT=1 requires any trajectory containing
      reflection markup to use the Hybrid-v2 format:
      '<reflection>So far I have ... However ...</reflection>'.
      TRAJ_GRPO_WRONG_REWARD sets the reward for incorrect+has-reflection
      trajectories; the paper-aligned default is 0.0.
      TRAJ_GRPO_REFLECTION_FREQ_PENALTY optionally sets an extra per-reflection
      linear penalty for correct reflective trajectories. Default 0.0 because
      R_eff already penalizes reflection frequency.
      TRAJ_GRPO_WRONG_REFLECTION_FREQ_PENALTY optionally sets a per-reflection
      penalty for incorrect reflective trajectories. Default 0.0.
      TRAJ_GRPO_WRONG_NO_REFLECT_PENALTY=1 enables the penalty for trajectories
      that answered incorrectly without ever triggering reflection.
      TRAJ_GRPO_WRONG_NO_REFLECT_N sets the hypothetical iteration count n used
      in the penalty formula -((n-1)/phi)^2. Default 2 (penalty=-0.04 at phi=5);
      set to 3 for a stronger signal (penalty=-0.16 at phi=5).
      TRAJ_GRPO_REWARD_STYLE=binary switches to simple outcome reward:
          correct                  -> 1.0
          wrong direct             -> TRAJ_GRPO_BINARY_DIRECT_WRONG_REWARD
          wrong with reflection    -> -((n_iters - 1) / phi)^2
      TRAJ_GRPO_BINARY_DIRECT_WRONG_REWARD sets the wrong direct reward in
      binary mode. Default -0.25.
      TRAJ_GRPO_BINARY_WRONG_REFLECTION_PENALTY=0 disables the wrong-reflection
      quadratic penalty in binary mode and uses 0.0 instead.
      TRAJ_GRPO_REWARD_STYLE=wrong_reflection_penalty switches to outcome-only
      correct rewards and table-based wrong rewards:
          correct direct/reflection -> 1 * format_reward
          wrong direct              -> -C
          wrong with 1/2/3+ refl    -> -A/-B/-C
      TRAJ_GRPO_WRONG_REFL_PENALTY_A/B/C configure A/B/C.
    """

    del data_source, kwargs

    if solution_str is not None and generated is None:
        generated = solution_str
    elif generated is None:
        generated = ""

    extra_info = extra_info if isinstance(extra_info, dict) else {}

    tag_count = len(REFL_PATTERN.findall(generated))
    has_reflection_markup = bool(
        tag_count
        or _REFLECTION_OPEN_RE.search(generated)
        or _REFLECTION_CLOSE_RE.search(generated)
    )
    require_format = _as_bool(os.environ.get("TRAJ_GRPO_REQUIRE_REFLECTION_FORMAT", "1"))
    reflection_format_valid = (
        True
        if (not require_format or not has_reflection_markup)
        else bool(_reflection_format_valid(generated))
    )

    info_num_reflections = _safe_int(extra_info.get("num_reflections", tag_count), tag_count)
    num_reflections = max(tag_count, info_num_reflections)

    env_phi = os.environ.get("TRAJ_GRPO_REWARD_PHI")
    default_phi = phi if phi is not None else extra_info.get("phi", 5)
    phi_source = env_phi if env_phi is not None else extra_info.get("phi", default_phi)
    phi = max(1, _safe_int(phi_source, 5))

    default_n_iters = 1 + num_reflections
    n_iters = _safe_int(extra_info.get("n_iters", default_n_iters), default_n_iters)
    n_iters = max(default_n_iters, n_iters)

    terminated_by_cap = _as_bool(extra_info.get("terminated_by_cap", False))
    if terminated_by_cap:
        n_iters = max(n_iters, phi)
    truncated = _as_bool(extra_info.get("truncated", False))

    task_reward = float(compute_math_score(solution_str=generated, ground_truth=ground_truth))
    is_correct = task_reward > 0.5

    use_eff_reward = _as_bool(os.environ.get("TRAJ_GRPO_USE_EFF_REWARD", "1"))
    eff_reward = _efficiency_reward(n_iters=n_iters, phi=phi) if use_eff_reward else 1.0
    wrong_reward = _safe_float(os.environ.get("TRAJ_GRPO_WRONG_REWARD", "0.0"), 0.0)
    reflection_freq_penalty_coef = max(
        0.0,
        _safe_float(os.environ.get("TRAJ_GRPO_REFLECTION_FREQ_PENALTY", "0.0"), 0.0),
    )
    wrong_reflection_freq_penalty_coef = max(
        0.0,
        _safe_float(os.environ.get("TRAJ_GRPO_WRONG_REFLECTION_FREQ_PENALTY", "0.0"), 0.0),
    )

    use_wrong_no_reflect_penalty = _as_bool(
        os.environ.get("TRAJ_GRPO_WRONG_NO_REFLECT_PENALTY", "0")
    )
    wrong_no_reflect_n = _safe_int(
        os.environ.get("TRAJ_GRPO_WRONG_NO_REFLECT_N", "3"), 3
    )
    wrong_no_reflect_score = _wrong_no_reflect_penalty(n=wrong_no_reflect_n, phi=phi)

    format_reward = 1.0 if reflection_format_valid else 0.0
    reward_style = os.environ.get("TRAJ_GRPO_REWARD_STYLE", "efficiency").strip().lower()
    use_binary_reward_style = reward_style in {
        "binary",
        "binary_outcome",
        "correctness",
        "correctness_binary",
    }
    binary_direct_wrong_reward = _safe_float(
        os.environ.get("TRAJ_GRPO_BINARY_DIRECT_WRONG_REWARD", "-0.25"), -0.25
    )
    binary_wrong_reflection_penalty_enabled = _as_bool(
        os.environ.get("TRAJ_GRPO_BINARY_WRONG_REFLECTION_PENALTY", "1")
    )
    binary_wrong_reflection_reward = (
        _wrong_no_reflect_penalty(n=n_iters, phi=phi)
        if binary_wrong_reflection_penalty_enabled and num_reflections > 0
        else 0.0
    )
    use_wrong_reflection_penalty_style = reward_style in {
        "wrong_reflection_penalty",
        "wrong_refl_penalty",
        "outcome_wrong_reflection_penalty",
    }
    wrong_refl_penalty_a = max(
        0.0,
        _safe_float(os.environ.get("TRAJ_GRPO_WRONG_REFL_PENALTY_A", "0.05"), 0.05),
    )
    wrong_refl_penalty_b = max(
        0.0,
        _safe_float(os.environ.get("TRAJ_GRPO_WRONG_REFL_PENALTY_B", "0.15"), 0.15),
    )
    wrong_refl_penalty_c = max(
        0.0,
        _safe_float(os.environ.get("TRAJ_GRPO_WRONG_REFL_PENALTY_C", "0.25"), 0.25),
    )

    if num_reflections <= 0:
        wrong_reflection_table_penalty = wrong_refl_penalty_c
    elif num_reflections == 1:
        wrong_reflection_table_penalty = wrong_refl_penalty_a
    elif num_reflections == 2:
        wrong_reflection_table_penalty = wrong_refl_penalty_b
    else:
        wrong_reflection_table_penalty = wrong_refl_penalty_c

    correct_reflection_freq_penalty = (
        reflection_freq_penalty_coef * float(num_reflections)
        if is_correct and num_reflections > 0
        else 0.0
    )
    wrong_reflection_freq_penalty = (
        wrong_reflection_freq_penalty_coef * float(num_reflections)
        if (not is_correct) and num_reflections > 0
        else 0.0
    )
    score_before_freq_penalty = (
        float(task_reward * eff_reward * format_reward) if is_correct else float(wrong_reward)
    )

    if use_binary_reward_style:
        if is_correct:
            score_before_freq_penalty = 1.0
        elif num_reflections == 0:
            score_before_freq_penalty = float(binary_direct_wrong_reward)
        else:
            score_before_freq_penalty = float(binary_wrong_reflection_reward)
        score = score_before_freq_penalty
    elif use_wrong_reflection_penalty_style:
        if is_correct:
            score_before_freq_penalty = float(format_reward)
            score = float(format_reward)
        else:
            score_before_freq_penalty = -float(wrong_reflection_table_penalty)
            score = -float(wrong_reflection_table_penalty)
    elif is_correct:
        score = max(0.0, score_before_freq_penalty - correct_reflection_freq_penalty)
    elif num_reflections == 0 and use_wrong_no_reflect_penalty:
        # Wrong with no reflection attempt: apply fixed penalty.
        # Wrong with reflection (num_reflections > 0): keep at wrong_reward (default 0.0)
        # so the model is not additionally penalised for having tried reflection.
        score = wrong_no_reflect_score
    else:
        score = float(wrong_reward - wrong_reflection_freq_penalty)

    return {
        "score": score,
        "score_before_freq_penalty": float(score_before_freq_penalty),
        "task_reward": float(task_reward),
        "eff_reward": float(eff_reward),
        "format_reward": float(format_reward),
        "reflection_freq_penalty": float(correct_reflection_freq_penalty),
        "wrong_reflection_freq_penalty": float(wrong_reflection_freq_penalty),
        "reflection_format_valid": float(reflection_format_valid),
        "has_reflection_markup": float(has_reflection_markup),
        "format_gate_zero": float(format_reward <= 0.5),
        "num_reflections": int(num_reflections),
        "n_reflections": int(num_reflections),
        "has_reflection": float(num_reflections > 0),
        "n_iters": int(n_iters),
        "phi": int(phi),
        "is_correct": float(is_correct),
        "terminated_by_cap": bool(terminated_by_cap),
        "truncated": bool(truncated),
        "wrong_no_reflect_score": float(wrong_no_reflect_score),
        "is_wrong_no_reflect": float(not is_correct and num_reflections == 0),
        "wrong_reflection_table_penalty": float(wrong_reflection_table_penalty),
        "wrong_reflection_full_chain_mask": float(
            use_wrong_reflection_penalty_style and (not is_correct) and num_reflections > 0
        ),
        "binary_direct_wrong_reward": float(binary_direct_wrong_reward),
        "binary_wrong_reflection_reward": float(binary_wrong_reflection_reward),
        "binary_wrong_reflection_penalty_enabled": float(
            binary_wrong_reflection_penalty_enabled
        ),
        "reward_style_binary": float(use_binary_reward_style),
        "reward_style_wrong_reflection_penalty": float(use_wrong_reflection_penalty_style),
    }
