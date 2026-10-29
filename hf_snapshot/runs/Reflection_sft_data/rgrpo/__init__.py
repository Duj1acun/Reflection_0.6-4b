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
R-GRPO: Reflective Reasoning GRPO.
"""

import os as _os

if _os.environ.get("RGRPO_LIGHT_IMPORT") == "1":
    __all__ = []
else:
    from .core_rgrpo import (
        compute_rgrpo_outcome_advantage,
        build_training_examples,
        DirectRollout,
        ReflRollout,
        TrainingExample,
    )
    from .core_refl_grpo import compute_reflection_grpo_score
    from .core_hybrid_grpo import (
        compute_hybrid_outcome_advantage,
        compute_hybrid_math_score,
    )

    from .reflection import (
        ReflectionRolloutManager,
        ReflectionRolloutConfig,
        integrate_reflection_rollout,
    )

    __all__ = [
        # Core functions
        "compute_rgrpo_outcome_advantage",
        "build_training_examples",
        "compute_reflection_grpo_score",
        # Data classes
        "DirectRollout",
        "ReflRollout",
        "TrainingExample",
        # Hybrid-GRPO
        "compute_hybrid_outcome_advantage",
        "compute_hybrid_math_score",
        # Reflection rollout
        "ReflectionRolloutManager",
        "ReflectionRolloutConfig",
        "integrate_reflection_rollout",
    ]

    # Hybrid-GRPO v2 (conditional baseline + dual-pool normalization).
    # Importing v2 monkey-patches core_hybrid_grpo so verl's
    # adv_estimator=hybrid_grpo transparently dispatches to v2.
    if _os.environ.get("RGRPO_HYBRID_V2") == "1":
        from . import core_hybrid_grpo_v2  # noqa: F401  triggers monkey-patch
