import numpy as np
import torch

from rgrpo.core_trajectory_grpo import (
    _efficiency_reward,
    _reflection_format_valid,
    _wrong_no_reflect_penalty,
)
from rgrpo.decoupled_advantage_dg import compute_decoupled_reflection_advantage
from rgrpo.reflection import REFLECTION_SYSTEM_PROMPT


def test_paper_reward_constants():
    assert _wrong_no_reflect_penalty(n=3, phi=4) == -0.25
    assert _efficiency_reward(n_iters=1, phi=4, free_reflections=1) == 1.0
    assert _efficiency_reward(n_iters=2, phi=4, free_reflections=1) == 1.0
    assert _efficiency_reward(n_iters=3, phi=4, free_reflections=1) == 0.9375


def test_reflection_format_gate_and_prompt():
    good = "<reflection>So far I have: x=1. However, that sign is wrong.</reflection>"
    bad = "<reflection>I should retry.</reflection>"
    assert _reflection_format_valid(good)
    assert not _reflection_format_valid(bad)
    assert REFLECTION_SYSTEM_PROMPT.endswith("then continue reasoning based on the reflection.")
    assert "Always present" not in REFLECTION_SYSTEM_PROMPT


def test_difficulty_floor_only_changes_reflection_samples():
    rewards = torch.tensor([[1.0], [1.0], [0.0], [0.0], [0.0], [1.0]])
    mask = torch.ones_like(rewards)
    index = np.array(["easy", "easy", "easy", "hard", "hard", "hard"], dtype=object)
    is_reflect = np.array([False, False, True, False, False, True])

    advantages, returns = compute_decoupled_reflection_advantage(
        rewards,
        mask,
        index,
        is_reflect,
        floor0=-0.5,
        norm_adv_by_std_in_grpo=False,
        difficulty_gated_floor=True,
    )

    expected = torch.tensor([[0.0], [0.0], [-0.5], [0.0], [0.0], [1.0]])
    assert torch.allclose(advantages, expected)
    assert torch.equal(advantages, returns)


def test_no_direct_group_falls_back_to_group_mean():
    rewards = torch.tensor([[0.0], [1.0]])
    mask = torch.ones_like(rewards)
    advantages, _ = compute_decoupled_reflection_advantage(
        rewards,
        mask,
        np.array(["p", "p"], dtype=object),
        np.array([True, True]),
        norm_adv_by_std_in_grpo=False,
        difficulty_gated_floor=True,
    )
    assert torch.allclose(advantages, torch.tensor([[-0.5], [0.5]]))


def test_uniform_outcome_skips_std_normalization():
    rewards = torch.tensor([[0.0], [-0.25]])
    mask = torch.ones_like(rewards)
    advantages, _ = compute_decoupled_reflection_advantage(
        rewards,
        mask,
        np.array(["p", "p"], dtype=object),
        np.array([False, False]),
        outcome_scores=np.array([0.0, 0.0]),
        skip_std_for_uniform_outcome=True,
        norm_adv_by_std_in_grpo=True,
    )
    assert torch.allclose(advantages, torch.tensor([[0.125], [-0.125]]))
