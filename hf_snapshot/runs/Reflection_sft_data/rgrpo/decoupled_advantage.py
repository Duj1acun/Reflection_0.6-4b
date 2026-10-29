"""Direction-2 + Direction-3 advantage for reflection-continuation GRPO.

This is a *standalone, pure* function. It does NOT modify any verl file. It is
installed at runtime by ``rgrpo.decoupled_patch.apply_patch`` (which is invoked
on import of ``rgrpo.core_trajectory_grpo_decoupled``, the new reward-plugin
entrypoint). The original ``rgrpo.core_trajectory_grpo`` / verl code path is
left byte-for-byte unchanged.

What it computes (per prompt group, grouped by ``index``/uid):

  Direction 2 — per-prompt counterfactual baseline
    baseline = mean reward of the DIRECT sub-group of the same prompt (mu_D).
    advantage_i = (r_i - mu_D) / (sigma_g + eps)        # both D and R use mu_D
    * Direct sub-group stays zero-sum (mu_D == mean(D)) -> direct/efficiency
      ability is not disturbed.
    * Reflect sub-group carries the per-prompt net signal (mu_R - mu_D) =
      "was reflecting better than NOT reflecting on THIS problem".
    * |D|==0 (no direct sample) -> fall back to standard group-mean GRPO.

  Direction 3 — decayed asymmetric floor on the REFLECT sub-group only
    Early training: clamp reflect advantages from below at ``floor0`` (default
    0.0) so dense negative signal from misplaced (easy-problem) reflections
    does not starve the reflection sub-policy before it learns. The floor
    blends out linearly over ``decay_steps`` and only clamps the *negative*
    side (it never lifts a value upward) so failed reflections stay <=0 (no
    cosmetic farming) while genuine recoveries (positive) are untouched.
    Direct samples are never floored.

Numeric conventions mirror verl ``compute_grpo_outcome_advantage``
(core_algos.py): group size 1 -> mean 0 / std 1; std is the unbiased
(sample) std; ``skip_std_for_uniform_outcome`` keeps mean-only advantages for
all-correct / all-wrong groups.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
import torch


def compute_decoupled_reflection_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    is_reflect: np.ndarray,
    *,
    step: int = 0,
    floor0: float = 0.0,
    decay_steps: int = 150,
    norm_adv_by_std_in_grpo: bool = True,
    epsilon: float = 1e-6,
    outcome_scores: Optional[np.ndarray] = None,
    skip_std_for_uniform_outcome: bool = False,
):
    """Return ``(advantages, returns)`` token tensors, same shapes/semantics as
    verl ``compute_grpo_outcome_advantage`` (both equal the broadcast scalar).

    Args mirror the verl function; extra args:
      is_reflect: (bs,) bool/0-1 array, True iff the trajectory reflected.
      step: current training step (drives the Direction-3 floor decay).
      floor0: early lower bound applied to reflect advantages (default 0.0).
      decay_steps: steps over which the floor linearly blends out
        (<=0 disables the floor entirely => pure Direction 2).
    """
    scores = token_level_rewards.sum(dim=-1)
    bsz = scores.shape[0]

    is_reflect_arr = np.asarray(is_reflect).reshape(-1)
    if is_reflect_arr.shape[0] != bsz:
        # Cannot align reflection flags -> behave as plain GRPO (safe).
        is_reflect_arr = np.zeros(bsz, dtype=bool)
    is_reflect_arr = is_reflect_arr.astype(bool)

    outcome_arr = None
    if skip_std_for_uniform_outcome and outcome_scores is not None:
        try:
            tmp = np.asarray(outcome_scores).reshape(-1)
            if tmp.shape[0] == bsz:
                outcome_arr = tmp
        except Exception:
            outcome_arr = None

    id2scores: dict = defaultdict(list)
    id2is_reflect: dict = defaultdict(list)
    id2outcome: dict = defaultdict(list)

    with torch.no_grad():
        for i in range(bsz):
            gid = index[i]
            id2scores[gid].append(scores[i])
            id2is_reflect[gid].append(bool(is_reflect_arr[i]))
            if outcome_arr is not None:
                v = float(outcome_arr[i])
                if not np.isnan(v):
                    id2outcome[gid].append(v > 0.5)

        id2baseline: dict = {}
        id2std: dict = {}
        id2skip_std: dict = {}
        for gid, vals in id2scores.items():
            n = len(vals)
            if n == 1:
                # Mirror verl: single-sample group -> mean 0, std 1.
                id2baseline[gid] = torch.tensor(0.0)
                id2std[gid] = torch.tensor(1.0)
                id2skip_std[gid] = False
                continue

            vt = torch.tensor(vals, dtype=torch.float32)
            group_mean = vt.mean()
            group_std = vt.std()  # unbiased (sample) std, matches verl

            refl_mask = torch.tensor(id2is_reflect[gid], dtype=torch.bool)
            direct_mask = ~refl_mask
            if direct_mask.any():
                # Direction 2: baseline = same-prompt DIRECT sub-group mean.
                baseline = vt[direct_mask].mean()
            else:
                # No direct counterfactual -> standard group-mean GRPO.
                baseline = group_mean

            id2baseline[gid] = baseline
            id2std[gid] = group_std

            outcomes = id2outcome.get(gid, [])
            id2skip_std[gid] = (
                skip_std_for_uniform_outcome
                and len(outcomes) == n
                and n > 1
                and (all(outcomes) or not any(outcomes))
            )

        # Direction 3: decayed floor weight (1 early -> 0 after decay_steps).
        if decay_steps and decay_steps > 0:
            w = 1.0 - (float(step) / float(decay_steps))
            floor_w = float(min(1.0, max(0.0, w)))
        else:
            floor_w = 0.0

        adv = scores.clone()
        for i in range(bsz):
            gid = index[i]
            baseline = id2baseline[gid]
            a = scores[i] - baseline
            if norm_adv_by_std_in_grpo and not id2skip_std.get(gid, False):
                a = a / (id2std[gid] + epsilon)

            if is_reflect_arr[i] and floor_w > 0.0:
                # Blend fully-floored (early) -> unfloored (after decay).
                # max() only clamps the negative side (never lifts upward),
                # so genuine recoveries (positive) pass through unchanged and
                # failed reflections never become positive (no cosmetic farm).
                a_floored = torch.maximum(a, torch.tensor(float(floor0), dtype=a.dtype))
                a = floor_w * a_floored + (1.0 - floor_w) * a

            adv[i] = a

        adv = adv.unsqueeze(-1) * response_mask

    return adv, adv
