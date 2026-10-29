"""REFLECT+ paper implementation.

Import concrete modules explicitly. Importing :mod:`rgrpo` must not install
the veRL advantage patch as a side effect; the paper training entrypoint
installs it through `rgrpo.core_trajectory_grpo_decoupled_dg`.
"""

__all__: list[str] = []
