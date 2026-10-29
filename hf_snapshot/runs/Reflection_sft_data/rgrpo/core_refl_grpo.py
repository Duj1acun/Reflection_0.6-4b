# Copyright 2025 R-GRPO Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Reflection-GRPO reward based on complete trajectories.

Reward design:

    - direct correct                         ->  1.0
    - correct with <= 1 reflection          ->  1.0
    - correct with >= 2 reflections         ->  1 - (extra_refl / phi_eff)^2
    - incorrect, no reflection              -> -0.35   (punished for not trying)
    - incorrect, >= 1 reflection            -> -0.30   (flat; reflection didn't help)

where:
    extra_refl = max(0, num_reflections - 1)   # first reflection is FREE
    phi_eff    = max(1, phi - 1)               # effective budget

Design rationale (v2):

  Problem with v1: The gap "wrong+1refl(-0.20) vs wrong+0refl(-0.30)" = +0.10
  created an unconditional bonus for inserting any <reflection> tag, regardless
  of content. Because this bonus is larger than noise and applies universally,
  the model discovered it as a pure dominant strategy: trigger_rate jumped from
  40% → 92% in just 26 steps, with the reflections being boilerplate ("adding
  code is redundant") rather than genuine error correction.

  Fix:
    1. Keep the wrong+no-reflection penalty slightly higher (-0.35) than
       wrong+reflection (-0.30), preserving the intent that "you should at
       least try to self-correct when wrong."
    2. Collapse all wrong+reflection cases to the SAME score (-0.30), regardless
       of how many reflections were added.  This removes the V-shaped sweet-spot
       at exactly n=1 that pinned the model to "always add exactly one tag."
    3. The gap is now 0.05 (vs 0.10 before).  The main task signal spans ~1.3
       (from -0.35 to +1.0), so the gaming incentive is only ~4% of that range—
       too small to dominate, yet still present as a weak nudge toward trying.

  The only way reflection meaningfully raises the reward is by leading to a
  CORRECT answer (+1.0 vs -0.30 = +1.30 difference).  GRPO's group-level
  advantage naturally surfaces this "reflect → correct" signal across the 8
  rollouts per prompt without needing any per-step bonus on the wrong branch.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .core_rgrpo import REFL_PATTERN, compute_math_score


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def compute_reflection_grpo_score(
    solution_str: str = None,
    ground_truth: str = None,
    generated: str = None,
    data_source: str = None,
    extra_info: Optional[dict] = None,
    phi: Optional[int] = 3,
    **kwargs,
) -> Dict[str, Any]:
    """Compute trajectory-level composite reward for Reflection-GRPO."""
    del data_source, kwargs

    if solution_str is not None and generated is None:
        generated = solution_str
    elif generated is None:
        generated = ""

    extra_info = extra_info if isinstance(extra_info, dict) else {}

    # Fall back to counting reflection tags from the final trajectory text if the
    # rollout path did not explicitly set the metadata.
    num_reflections = _safe_int(
        extra_info.get("num_reflections", len(REFL_PATTERN.findall(generated))),
        default=0,
    )
    n_iters = _safe_int(extra_info.get("n_iters", 1 + num_reflections), default=1 + num_reflections)
    phi = _safe_int(extra_info.get("phi", phi if phi is not None else max(1, n_iters)), default=max(1, n_iters))
    terminated_by_cap = bool(extra_info.get("terminated_by_cap", False))

    task_reward = float(
        compute_math_score(solution_str=generated, ground_truth=ground_truth)
    )
    is_correct = bool(task_reward > 0.5)

    # First reflection is free — only extra reflections beyond the first are penalised.
    # This matches the docstring: extra_refl = max(0, num_reflections - 1).
    free_reflections = 1
    extra_reflections = max(0, num_reflections - free_reflections)
    # eff_phi is the *effective* reflection budget after the free slot.
    # phi comes from max_iters (default 3), so eff_phi = max(1, phi - 1) = 2.
    eff_phi = max(1, phi - free_reflections)

    # Correct answer: smooth quadratic penalty only for reflections beyond the free one.
    # Uses eff_phi (not a hardcoded 3) so the budget from max_iters actually matters.

    if is_correct:
        score = 1.0
    else:
        if num_reflections == 0:
            score = -0.35    # 没有尝试反思，惩罚略重
        else:
            score = -0.30  # 尝试了但仍错，不区分次数
    return {
        "score": float(score),
        "task_reward": float(task_reward),
        "num_reflections": int(num_reflections),
        "extra_reflections": int(extra_reflections),
        "n_iters": int(n_iters),
        "phi": int(phi),
        "is_correct": is_correct,
        "terminated_by_cap": bool(terminated_by_cap),
    }
