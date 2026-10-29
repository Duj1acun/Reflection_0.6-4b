"""Direction-2 + difficulty-gated reflect-floor advantage (DG variant).

Standalone copy of ``rgrpo.decoupled_advantage`` PLUS one added behaviour,
selected by the new ``difficulty_gated_floor`` arg. It does NOT modify any
verl file and does NOT touch the original ``decoupled_advantage`` module, so
runs that import the legacy path are byte-for-byte unaffected. Installed at
runtime by ``rgrpo.decoupled_patch_dg.apply_patch`` (invoked on import of
``rgrpo.core_trajectory_grpo_decoupled_dg``).

What it computes (per prompt group, grouped by ``index``/uid):

  Direction 2 — per-prompt counterfactual baseline (UNCHANGED)
    baseline = mean reward of the DIRECT sub-group of the same prompt (mu_D).
    advantage_i = (r_i - mu_D) / (sigma_g + eps)        # both D and R use mu_D
    * Direct sub-group stays zero-sum -> direct/efficiency ability untouched.
    * Reflect sub-group carries the per-prompt net signal (mu_R - mu_D).
    * |D|==0 (no direct sample) -> fall back to standard group-mean GRPO.

  Floor on the REFLECT sub-group only — two modes:

    Legacy (difficulty_gated_floor=False, default): the original Direction-3
    *time-decayed* asymmetric floor. Weight 1 early -> 0 after decay_steps.
    Byte-equivalent to ``decoupled_advantage.compute_decoupled_reflection_advantage``.

    Difficulty-gated (difficulty_gated_floor=True): the floor weight is the
    PER-PROMPT difficulty ``fw_gid = clip(mu_D, 0, 1)`` instead of a function
    of the training step. Easy prompt (direct sub-group ~all-correct,
    mu_D~=1) -> floor fully on, so a failed reflection's ~ -1/sigma is clamped
    to ``floor0`` (default -0.25): a residual NEGATIVE is preserved (the
    "don't reflect on easy" signal is muted, NOT zeroed -> no white-knight)
    while no longer drowning the whole reflect gradient. Hard prompt
    (mu_D~=0) -> floor off, so a genuine rescue keeps its full +1/sigma and a
    failed hard reflection keeps full negative (no free positive). Groups
    with no direct sub-group -> floor off (fw=0): never floor a pure-reflect
    group (white-knight guard). decay_steps is ignored in this mode; the gate
    is per-prompt so it never globally white-knights and needs no decay.
    The asymmetric ``max(a, floor0)`` (only clamps the negative side, never
    lifts a positive) is preserved in both modes -> no cosmetic farming.

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
    difficulty_gated_floor: bool = False,
):
    """Return ``(advantages, returns)`` token tensors, same shapes/semantics as
    verl ``compute_grpo_outcome_advantage`` (both equal the broadcast scalar).

    Args mirror the verl function; extra args:
      is_reflect: (bs,) bool/0-1 array, True iff the trajectory reflected.
      step: current training step (drives the legacy time-floor decay).
      floor0: lower bound applied to reflect advantages (default 0.0).
      decay_steps: legacy time-floor decay horizon (<=0 disables it). Ignored
        when difficulty_gated_floor is True.
      difficulty_gated_floor: if True, the floor weight is the per-prompt
        difficulty clip(mu_D, 0, 1) instead of the time-decayed weight; only
        groups that have a direct sub-group are floored.
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
        id2has_direct: dict = {}
        for gid, vals in id2scores.items():
            n = len(vals)
            if n == 1:
                # Mirror verl: single-sample group -> mean 0, std 1.
                id2baseline[gid] = torch.tensor(0.0)
                id2std[gid] = torch.tensor(1.0)
                id2skip_std[gid] = False
                id2has_direct[gid] = False
                continue

            vt = torch.tensor(vals, dtype=torch.float32)
            group_mean = vt.mean()
            group_std = vt.std()  # unbiased (sample) std, matches verl

            refl_mask = torch.tensor(id2is_reflect[gid], dtype=torch.bool)
            direct_mask = ~refl_mask
            if direct_mask.any():
                # Direction 2: baseline = same-prompt DIRECT sub-group mean.
                baseline = vt[direct_mask].mean()
                id2has_direct[gid] = True
            else:
                # No direct counterfactual -> standard group-mean GRPO.
                baseline = group_mean
                id2has_direct[gid] = False

            id2baseline[gid] = baseline
            id2std[gid] = group_std

            outcomes = id2outcome.get(gid, [])
            id2skip_std[gid] = (
                skip_std_for_uniform_outcome
                and len(outcomes) == n
                and n > 1
                and (all(outcomes) or not any(outcomes))
            )

        # Legacy Direction-3 time-decayed floor weight (1 early -> 0 after
        # decay_steps). Used only when difficulty_gated_floor is False.
        if decay_steps and decay_steps > 0:
            w = 1.0 - (float(step) / float(decay_steps))
            floor_w_time = float(min(1.0, max(0.0, w)))
        else:
            floor_w_time = 0.0

        # Difficulty-gated floor weight, per prompt: clip(mu_D, 0, 1) for
        # groups that have a direct sub-group, else 0.0 (never floor a
        # pure-reflect / no-direct group -> white-knight guard).
        id2floor_w_dg: dict = {}
        if difficulty_gated_floor:
            for gid in id2scores:
                if id2has_direct.get(gid, False):
                    mu_d = float(id2baseline[gid])
                    id2floor_w_dg[gid] = float(min(1.0, max(0.0, mu_d)))
                else:
                    id2floor_w_dg[gid] = 0.0

        adv = scores.clone()
        for i in range(bsz):
            gid = index[i]
            baseline = id2baseline[gid]
            a = scores[i] - baseline
            if norm_adv_by_std_in_grpo and not id2skip_std.get(gid, False):
                a = a / (id2std[gid] + epsilon)

            if is_reflect_arr[i]:
                if difficulty_gated_floor:
                    floor_w = id2floor_w_dg.get(gid, 0.0)
                else:
                    floor_w = floor_w_time
                if floor_w > 0.0:
                    # Asymmetric blend: max() only clamps the negative side
                    # (never lifts a positive), so genuine recoveries pass
                    # through unchanged and failed reflections never become
                    # positive (no cosmetic farm). Weight source differs by
                    # mode (time decay vs per-prompt difficulty); the clamp
                    # math is identical.
                    a_floored = torch.maximum(
                        a, torch.tensor(float(floor0), dtype=a.dtype)
                    )
                    a = floor_w * a_floored + (1.0 - floor_w) * a

            adv[i] = a

        adv = adv.unsqueeze(-1) * response_mask

    return adv, adv
