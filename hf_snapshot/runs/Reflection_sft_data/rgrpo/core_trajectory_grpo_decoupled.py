"""Reward-plugin entrypoint for the decoupled-baseline (Direction 2 + 3) run.

This module exists so the existing ``rgrpo.core_trajectory_grpo`` stays
byte-for-byte unchanged. It:

  1. re-exports the UNCHANGED reward ``compute_trajectory_grpo_score`` (same
     scalar reward; Direction 2/3 only change the *advantage*, not the reward);
  2. on import, installs the decoupled-baseline + reflect-floor advantage patch
     via ``rgrpo.decoupled_patch.apply_patch``.

The verl reward manager imports ``custom_reward_function.path`` inside the Ray
TaskRunner actor (``verl/trainer/ppo/reward.py`` -> ``importlib.import_module``)
before ``trainer.fit()``. Pointing the NEW run script at this module therefore
installs the patch in the exact process that later calls ``compute_advantage``.
The OLD run script points at ``rgrpo.core_trajectory_grpo`` and never imports
this, so previous training is completely unaffected.
"""

from rgrpo.core_trajectory_grpo import (  # noqa: F401  re-export, reward UNCHANGED
    compute_trajectory_grpo_score,
)
from rgrpo.decoupled_patch import apply_patch

# Side-effecting import: install the advantage patch in this (actor) process.
apply_patch()

__all__ = ["compute_trajectory_grpo_score"]
