"""Reward-plugin entrypoint for the difficulty-gated-floor (Direction 2 + DG)
run.

DG sibling of ``rgrpo.core_trajectory_grpo_decoupled``. It exists so the
existing ``rgrpo.core_trajectory_grpo`` AND ``rgrpo.core_trajectory_grpo_
decoupled`` stay byte-for-byte unchanged. It:

  1. re-exports the UNCHANGED reward ``compute_trajectory_grpo_score`` (same
     scalar reward; Direction 2 / the difficulty-gated floor only change the
     *advantage*, not the reward);
  2. on import, installs the decoupled-baseline + difficulty-gated-floor
     advantage patch via ``rgrpo.decoupled_patch_dg.apply_patch``.

The verl reward manager imports ``custom_reward_function.path`` inside the
Ray TaskRunner actor (``verl/trainer/ppo/reward.py`` ->
``importlib.import_module``) before ``trainer.fit()``. Pointing the NEW run
script (``run_trajectory_grpo_decoupled_dg_math.sh``) at this module
therefore installs the DG patch in the exact process that later calls
``compute_advantage``. The old run scripts point at
``rgrpo.core_trajectory_grpo`` / ``rgrpo.core_trajectory_grpo_decoupled``
and never import this, so previous/concurrent training is completely
unaffected.
"""

from rgrpo.core_trajectory_grpo import (  # noqa: F401  re-export, reward UNCHANGED
    compute_trajectory_grpo_score,
)
from rgrpo.decoupled_patch_dg import apply_patch

# Side-effecting import: install the advantage patch in this (actor) process.
apply_patch()

__all__ = ["compute_trajectory_grpo_score"]
