"""Non-invasive runtime patch installing the decoupled-baseline +
DIFFICULTY-GATED reflect-floor advantage into verl's ``compute_advantage``.

DG sibling of ``rgrpo.decoupled_patch``. Identical safety contract; differs
only in: (1) it calls the difficulty-gated advantage in
``rgrpo.decoupled_advantage_dg``; (2) it reads
``TRAJ_GRPO_REFLECT_FLOOR_DIFFICULTY_GATED`` and forwards it; (3) it marks
the wrapper with a distinct attribute so it never collides with the legacy
patch. The legacy reward path (``rgrpo.core_trajectory_grpo`` /
``rgrpo.core_trajectory_grpo_decoupled``) never imports this module, so those
runs are entirely unaffected.

Why a runtime patch (not an edit to verl): the existing training pipeline
must stay byte-for-byte unchanged. ``apply_patch()`` is called on import of
``rgrpo.core_trajectory_grpo_decoupled_dg`` -- which the verl reward manager
imports (via ``custom_reward_function.path``) *inside the Ray TaskRunner
actor*, before ``trainer.fit()``. So the patch lands in the exact process
that later calls ``compute_advantage``.

Activation is additionally gated by env ``TRAJ_GRPO_DECOUPLED_BASELINE``:
when not truthy the wrapper delegates verbatim to the original verl
function, so the patch is a strict no-op unless the new run script enables
it. Any error in the decoupled path falls back to the original
implementation -- training never breaks because of this module.

Env knobs:
  TRAJ_GRPO_DECOUPLED_BASELINE             1 to activate (default 0 / off)
  TRAJ_GRPO_REFLECT_ADV_FLOOR              floor0 lower bound (default 0.0)
  TRAJ_GRPO_REFLECT_ADV_FLOOR_DECAY_STEPS  legacy time-floor horizon (ignored
                                           when difficulty-gated is on)
  TRAJ_GRPO_REFLECT_FLOOR_DIFFICULTY_GATED 1 => per-prompt difficulty floor
                                           (default 0 => legacy time floor)
"""

from __future__ import annotations

import os
import sys
import traceback

_PATCHED = False
_STATE = {"step": 0}


def _truthy(v) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _adv_is_grpo(adv_estimator) -> bool:
    name = getattr(adv_estimator, "value", adv_estimator)
    return str(name).strip().lower() == "grpo"


def _make_patched(orig_compute_advantage):
    def patched_compute_advantage(data, adv_estimator, *args, **kwargs):
        # Gate: env off, or not GRPO -> original behaviour, verbatim.
        if not _truthy(os.environ.get("TRAJ_GRPO_DECOUPLED_BASELINE", "0")):
            return orig_compute_advantage(data, adv_estimator, *args, **kwargs)
        if not _adv_is_grpo(adv_estimator):
            return orig_compute_advantage(data, adv_estimator, *args, **kwargs)

        try:
            import numpy as np
            from rgrpo.decoupled_advantage_dg import (
                compute_decoupled_reflection_advantage,
            )

            multi_turn = bool(kwargs.get("multi_turn", False))
            norm_adv_by_std_in_grpo = bool(
                kwargs.get("norm_adv_by_std_in_grpo", True)
            )
            grpo_skip_std_for_uniform_outcome = bool(
                kwargs.get("grpo_skip_std_for_uniform_outcome", False)
            )

            # Mirror verl GRPO branch mask selection exactly.
            grpo_calculation_mask = data.batch["response_mask"]
            if multi_turn:
                response_length = grpo_calculation_mask.size(1)
                grpo_calculation_mask = data.batch["loss_mask"][
                    :, -response_length:
                ]

            ntb = data.non_tensor_batch
            index = ntb["uid"]

            # Per-sample reflection flag (reward dict keys, same plumbing as
            # the existing is_correct usage in ray_trainer GRPO branch).
            refl = ntb.get("has_reflection", None)
            if refl is None:
                refl = ntb.get("num_reflections", None)
            if refl is None:
                refl = ntb.get("n_reflections", None)
            if refl is None:
                # No reflection signal -> nothing to decouple; defer.
                return orig_compute_advantage(
                    data, adv_estimator, *args, **kwargs
                )
            is_reflect = np.asarray(refl).reshape(-1).astype(float) > 0.5

            outcome_scores = None
            if grpo_skip_std_for_uniform_outcome:
                outcome_scores = ntb.get("is_correct", None)
                if outcome_scores is None:
                    outcome_scores = ntb.get("task_reward", None)

            floor0 = float(
                os.environ.get("TRAJ_GRPO_REFLECT_ADV_FLOOR", "0.0")
            )
            decay_steps = int(
                os.environ.get(
                    "TRAJ_GRPO_REFLECT_ADV_FLOOR_DECAY_STEPS", "150"
                )
            )
            difficulty_gated_floor = _truthy(
                os.environ.get(
                    "TRAJ_GRPO_REFLECT_FLOOR_DIFFICULTY_GATED", "0"
                )
            )

            _STATE["step"] += 1  # one compute_advantage call per training step

            advantages, returns = compute_decoupled_reflection_advantage(
                token_level_rewards=data.batch["token_level_rewards"],
                response_mask=grpo_calculation_mask,
                index=index,
                is_reflect=is_reflect,
                step=_STATE["step"],
                floor0=floor0,
                decay_steps=decay_steps,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                outcome_scores=outcome_scores,
                skip_std_for_uniform_outcome=grpo_skip_std_for_uniform_outcome,
                difficulty_gated_floor=difficulty_gated_floor,
            )
            data.batch["advantages"] = advantages
            data.batch["returns"] = returns
            if _STATE["step"] <= 3 or _STATE["step"] % 25 == 0:
                print(
                    f"[decoupled-adv-dg] step~{_STATE['step']} active "
                    f"floor0={floor0} dg={difficulty_gated_floor} "
                    f"decay_steps={decay_steps} "
                    f"reflect_frac={float(is_reflect.mean()):.3f}",
                    file=sys.stderr,
                    flush=True,
                )
            return data
        except Exception:  # never break training because of this patch
            print(
                "[decoupled-adv-dg] ERROR -> falling back to original "
                "compute_advantage:\n" + traceback.format_exc(),
                file=sys.stderr,
                flush=True,
            )
            return orig_compute_advantage(data, adv_estimator, *args, **kwargs)

    patched_compute_advantage._decoupled_dg_patched = True  # type: ignore[attr-defined]
    return patched_compute_advantage


def apply_patch() -> bool:
    """Idempotently wrap ``verl.trainer.ppo.ray_trainer.compute_advantage``.

    Returns True if the patch is in place (or already was). Safe to call
    multiple times and from multiple processes. Uses a distinct marker
    attribute so it never double-wraps or collides with the legacy
    ``rgrpo.decoupled_patch``.
    """
    global _PATCHED
    try:
        import verl.trainer.ppo.ray_trainer as rt
    except Exception:
        print(
            "[decoupled-adv-dg] could not import verl ray_trainer; "
            "patch skipped",
            file=sys.stderr,
            flush=True,
        )
        return False

    cur = getattr(rt, "compute_advantage", None)
    if cur is not None and getattr(cur, "_decoupled_dg_patched", False):
        _PATCHED = True
        return True

    rt.compute_advantage = _make_patched(cur)
    _PATCHED = True
    print(
        "[decoupled-adv-dg] patched verl compute_advantage "
        "(gated by TRAJ_GRPO_DECOUPLED_BASELINE; "
        "difficulty-gated via TRAJ_GRPO_REFLECT_FLOOR_DIFFICULTY_GATED)",
        file=sys.stderr,
        flush=True,
    )
    return True
