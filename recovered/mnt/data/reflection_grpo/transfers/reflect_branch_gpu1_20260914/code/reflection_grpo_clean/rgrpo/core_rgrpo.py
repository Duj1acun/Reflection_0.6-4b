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
R-GRPO: Reflective Reasoning GRPO core algorithms.

Reward structure:
  No-reflection rollouts  → standard GRPO within-group advantage (all response tokens)
  Reflection rollouts     → three token types:
    pre_refl tokens       → delta reward: refl_acc - dir_acc (can be negative)
    refl tokens           → refl_acc + recovery_bonus - length_penalty - freq_penalty
    continuation tokens   → within-K GRPO advantage (from K fresh continuation samples)
  Multi-turn (up to 3)    → recursively apply per reflection round

Key modifications from original R-GRPO (per edit.md):
1. _pre_refl_reward: Removed max(0, ...) - allows negative rewards
2. _refl_reward: Uses dir_acc instead of trigger_freq for per-problem ability awareness
3. Z-score normalization: Both refl and pre rewards are z-score normalized
4. Joint trajectory example: pre_refl + refl_tag + best_continuation in single forward pass
5. Loss aggregation: Use 'seq-mean-token-mean' mode (each example's loss is averaged
   over tokens first, then averaged across examples)

Configuration recommendation for R-GRPO:
  config.actor_rollout_ref.actor.loss_agg_mode = "seq-mean-token-mean"
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch

# ── Hyperparameters ────────────────────────────────────────────────────────────
GAMMA = 0.5  # recovery bonus weight
LEN_PENALTY_COEF = 0.0  # reflection length penalty disabled
CONT_LEN_PENALTY_COEF = 0.0  # continuation length penalty disabled
FREQ_PENALTY_COEF = 0.0  # frequency penalty disabled
EPS = 1e-8
MAX_REFLECTIONS = 3
K_CONTINUATIONS = 4

# Reflection tag pattern
REFL_PATTERN = re.compile(r"<reflection>[\s\S]+?</reflection>", re.DOTALL)


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class RoundData:
    """One vLLM generate call's result."""
    prompt: str
    output_text: str
    token_ids: List[int]
    log_probs: List[float]


@dataclass
class DirectRollout:
    """Rollout with no reflection."""
    prompt: str
    output_text: str
    token_ids: List[int]
    log_probs: List[float]
    accuracy: float


@dataclass
class ReflRollout:
    """
    Rollout containing one or more reflections.

    Per reflection round r:
      rounds[r]              : RoundData for round r
      pre_refl_token_ids[r]  : token ids of pre-reflection segment
      pre_refl_log_probs[r]  : corresponding log probs
      refl_token_ids[r]      : token ids of <reflection>...</reflection> segment
      refl_log_probs[r]      : corresponding log probs
      refl_tags[r]           : full text of <reflection>...</reflection>
      cont_rollouts[r]       : K=4 RoundData objects (continuations after refl r)
      cont_accuracies[r]     : accuracy of each continuation
    """
    rounds: List[RoundData]
    pre_refl_token_ids: List[List[int]]
    pre_refl_log_probs: List[List[float]]
    refl_token_ids: List[List[int]]
    refl_log_probs: List[List[float]]
    refl_tags: List[str]
    final_accuracy: float
    cont_rollouts: List[List[RoundData]] = field(default_factory=list)
    cont_accuracies: List[List[float]] = field(default_factory=list)


@dataclass
class TrainingExample:
    """
    One training sequence (prompt + response) with token-level advantages.

    input_ids:        full token ids = prompt_ids + response_ids
    prompt_len:       number of prompt tokens (no gradient on these)
    token_advantages: per-token scalar advantage for response tokens only
                      len(token_advantages) == len(input_ids) - prompt_len
    old_log_probs:    per-token old log prob (for KL penalty)
                      same length as token_advantages
    example_type:     "direct_nr" | "refl_orig" | "refl_joint" | "continuation"
    refl_round:       which reflection round this came from (for logging)
    """
    input_ids: List[int]
    prompt_len: int
    token_advantages: List[float]
    old_log_probs: List[float]
    example_type: str = "direct_nr"
    refl_round: int = 0


# ── Reward Functions ──────────────────────────────────────────────────────────

def _grpo_advantage(rewards: List[float]) -> List[float]:
    """(r - mean) / (std + eps) normalization (z-score)."""
    if not rewards:
        return []
    if len(rewards) == 1:
        return [0.0]
    mean_r = sum(rewards) / len(rewards)
    var_r = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
    std_r = math.sqrt(var_r)
    return [(r - mean_r) / (std_r + EPS) for r in rewards]


def _refl_reward(
    refl_acc: float,
    dir_acc: float,
    n_refl_tokens: int,
) -> float:
    """
    Compute reward for reflection tokens.
    
    Modified from original:
    - Uses dir_acc instead of trigger_freq for frequency penalty
    - This provides per-problem ability awareness
    """
    base = refl_acc
    bonus = GAMMA * max(0.0, refl_acc - dir_acc)
    len_pen = LEN_PENALTY_COEF * n_refl_tokens
    freq_pen = FREQ_PENALTY_COEF * dir_acc  # per-problem ability
    return base + bonus - len_pen - freq_pen


def _pre_refl_reward(refl_acc: float, dir_acc: float) -> float:
    """
    Compute reward for pre-reflection tokens.
    
    Modified from original:
    - Removed max(0, ...) truncation
    - Allows negative rewards when reflection hurts performance
    - Model learns "don't trigger reflection for problems you can solve directly"
    """
    return refl_acc - dir_acc


def _cont_reward(cont_acc: float, n_cont_tokens: int) -> float:
    """
    Compute reward for continuation tokens (after reflection).
    
    Adds length penalty to encourage concise continuations.
    
    Args:
        cont_acc: Accuracy of this continuation
        n_cont_tokens: Number of tokens in the continuation
    
    Returns:
        Reward with length penalty applied
    """
    len_pen = CONT_LEN_PENALTY_COEF * n_cont_tokens
    return cont_acc - len_pen


# ── Reflection Parsing ─────────────────────────────────────────────────────────

def find_first_reflection(text: str) -> Optional[re.Match]:
    """Return the re.Match for the first complete <reflection>...</reflection> in text."""
    return REFL_PATTERN.search(text)


def find_token_boundary(
    tokenizer,
    prompt: str,
    response: str,
    char_pos: int,
) -> int:
    """
    Find the token index in the response token sequence corresponding to
    character position `char_pos` in `response`.
    
    Uses cumulative tokenization of (prompt + response[:char_pos]) minus
    len(prompt tokens) to find the boundary.
    """
    if char_pos <= 0:
        return 0
    
    prefix_text = prompt + response[:char_pos]
    prompt_text = prompt

    prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    return max(0, len(prefix_ids) - len(prompt_ids))


def split_response_at_first_reflection(
    tokenizer,
    prompt: str,
    response: str,
    token_ids: List[int],
    log_probs: List[float],
) -> Tuple[
    Optional[List[int]],   # pre_refl token ids
    Optional[List[float]], # pre_refl log probs
    Optional[List[int]],   # refl token ids
    Optional[List[float]], # refl log probs
    Optional[str],         # reflection tag text
]:
    """
    Split a response (and its per-token log probs) at the first
    <reflection>...</reflection> boundary.

    Returns (pre_ids, pre_lps, refl_ids, refl_lps, refl_tag_text).
    If no reflection found, all return values are None.
    """
    match = find_first_reflection(response)
    if match is None:
        return None, None, None, None, None

    pre_refl_end_char = match.start()
    refl_end_char = match.end()
    refl_tag_text = match.group(0)

    pre_token_end = find_token_boundary(tokenizer, prompt, response, pre_refl_end_char)
    refl_token_end = find_token_boundary(tokenizer, prompt, response, refl_end_char)

    # Clamp to actual token sequence length
    n = len(token_ids)
    pre_token_end = min(pre_token_end, n)
    refl_token_end = min(refl_token_end, n)

    pre_ids = token_ids[:pre_token_end]
    pre_lps = log_probs[:pre_token_end] if log_probs else []
    refl_ids = token_ids[pre_token_end:refl_token_end]
    refl_lps = log_probs[pre_token_end:refl_token_end] if log_probs else []

    return pre_ids, pre_lps, refl_ids, refl_lps, refl_tag_text


# ── Main Builder ──────────────────────────────────────────────────────────────

def build_training_examples(
    tokenizer,
    header: str,
    direct_rollouts: List[DirectRollout],
    refl_rollouts: List[ReflRollout],
) -> Tuple[List[TrainingExample], dict]:
    """
    Convert one problem's G rollouts into TrainingExamples with token advantages.

    Returns (examples, stats_dict).
    stats_dict keys:
      dir_acc, mean_refl_acc, trigger_freq,
      mean_refl_reward, mean_pre_reward, mean_cont_advantage,
      total_reflection_count, refl_count_per_round  (list of 3)
    """
    G = len(direct_rollouts) + len(refl_rollouts)
    examples: List[TrainingExample] = []

    # ── Direct rollouts ────────────────────────────────────────────────────────
    dir_accs = [d.accuracy for d in direct_rollouts]
    
    # If no direct rollouts, estimate dir_acc from reflection rollouts' final accuracy
    # This prevents division issues and provides a reasonable baseline
    if dir_accs:
        dir_acc = sum(dir_accs) / len(dir_accs)
    elif refl_rollouts:
        # Use final accuracy of reflection rollouts as baseline
        dir_acc = sum(r.final_accuracy for r in refl_rollouts) / len(refl_rollouts)
    else:
        dir_acc = 0.0
    
    nr_adv = _grpo_advantage(dir_accs)

    for rollout, adv in zip(direct_rollouts, nr_adv):
        prompt_ids = tokenizer(rollout.prompt, add_special_tokens=False)["input_ids"]
        resp_ids = rollout.token_ids
        if not resp_ids:
            continue
        resp_len = len(resp_ids)
        examples.append(TrainingExample(
            input_ids=prompt_ids + resp_ids,
            prompt_len=len(prompt_ids),
            token_advantages=[adv] * resp_len,
            old_log_probs=rollout.log_probs[:resp_len] if rollout.log_probs else [0.0] * resp_len,
            example_type="direct_nr",
        ))

    # ── Reflection rollouts (Two-pass: collect then normalize) ─────────────────
    trigger_freq = len(refl_rollouts) / G if G > 0 else 0.0

    # First pass: collect all raw rewards
    raw_refl_rewards: List[float] = []
    raw_pre_rewards: List[float] = []
    refl_meta = []  # Store metadata for second pass

    for rollout in refl_rollouts:
        for r in range(len(rollout.refl_tags)):
            cont_accs = rollout.cont_accuracies[r] if r < len(rollout.cont_accuracies) else []
            refl_acc = (sum(cont_accs) / len(cont_accs)) if cont_accs else rollout.final_accuracy
            n_refl_tok = len(rollout.refl_token_ids[r]) if r < len(rollout.refl_token_ids) else 0
            
            raw_refl_rewards.append(_refl_reward(refl_acc, dir_acc, n_refl_tok))
            raw_pre_rewards.append(_pre_refl_reward(refl_acc, dir_acc))
            refl_meta.append((rollout, r, cont_accs, refl_acc))

    # Z-score normalization
    normed_refl = _grpo_advantage(raw_refl_rewards)
    normed_pre = _grpo_advantage(raw_pre_rewards)

    all_refl_rewards: List[float] = []
    all_pre_rewards: List[float] = []
    all_cont_advs: List[float] = []
    refl_count_per_round = [0, 0, 0]

    # Second pass: build training examples with normalized rewards
    for idx, (rollout, r, cont_accs, refl_acc) in enumerate(refl_meta):
        r_refl = normed_refl[idx] if idx < len(normed_refl) else 0.0
        r_pre = normed_pre[idx] if idx < len(normed_pre) else 0.0
        
        all_refl_rewards.append(r_refl)
        all_pre_rewards.append(r_pre)
        if r < 3:
            refl_count_per_round[r] += 1

        # Get token data
        pre_ids = rollout.pre_refl_token_ids[r] if r < len(rollout.pre_refl_token_ids) else []
        refl_ids = rollout.refl_token_ids[r] if r < len(rollout.refl_token_ids) else []
        pre_lps = rollout.pre_refl_log_probs[r] if r < len(rollout.pre_refl_log_probs) else []
        refl_lps = rollout.refl_log_probs[r] if r < len(rollout.refl_log_probs) else []

        # ── Joint trajectory example (MODIFICATION 4) ──────────────────────────
        # pre_refl + refl_tag + best_continuation as single example
        if r < len(rollout.cont_rollouts) and rollout.cont_rollouts[r] and cont_accs:
            # Ensure cont_accs and cont_rollouts have matching lengths
            n_conts = min(len(rollout.cont_rollouts[r]), len(cont_accs))
            if n_conts == 0:
                continue
            
            # Compute continuation rewards with length penalty
            cont_rewards = []
            for i in range(n_conts):
                cont_rd = rollout.cont_rollouts[r][i]
                n_tokens = len(cont_rd.token_ids) if cont_rd.token_ids else 0
                cont_rewards.append(_cont_reward(cont_accs[i], n_tokens))
            
            # Apply GRPO advantage on length-penalized rewards
            cont_adv_scalars = _grpo_advantage(cont_rewards)

            # Find best continuation by reward (not just accuracy)
            best_idx = max(range(len(cont_rewards)), key=lambda i: cont_rewards[i])
            best_cont = rollout.cont_rollouts[r][best_idx]
            best_c_adv = cont_adv_scalars[best_idx] if best_idx < len(cont_adv_scalars) else 0.0

            # Build joint example
            joint_ids = pre_ids + refl_ids + (best_cont.token_ids if best_cont.token_ids else [])
            joint_lps = pre_lps + refl_lps + (best_cont.log_probs[:len(best_cont.token_ids)] if best_cont.log_probs and best_cont.token_ids else [])
            joint_advs = (
                [r_pre] * len(pre_ids) +
                [r_refl] * len(refl_ids) +
                [best_c_adv] * len(best_cont.token_ids if best_cont.token_ids else [])
            )

            if joint_ids:
                round_prompt_ids = tokenizer(
                    rollout.rounds[r].prompt, add_special_tokens=False
                )["input_ids"] if r < len(rollout.rounds) else []
                
                # Ensure log_probs match token count
                if len(joint_lps) < len(joint_ids):
                    joint_lps = joint_lps + [0.0] * (len(joint_ids) - len(joint_lps))
                
                examples.append(TrainingExample(
                    input_ids=round_prompt_ids + joint_ids,
                    prompt_len=len(round_prompt_ids),
                    token_advantages=joint_advs,
                    old_log_probs=joint_lps[:len(joint_advs)],
                    example_type="refl_joint",
                    refl_round=r,
                ))

            # K=4 independent continuation examples (preserved)
            cont_prompt = header + rollout.refl_tags[r].rstrip() + "\n"
            cont_prompt_ids = tokenizer(cont_prompt, add_special_tokens=False)["input_ids"]

            # Use n_conts to ensure we don't go out of bounds
            for i in range(n_conts):
                cont_rd = rollout.cont_rollouts[r][i]
                c_adv = cont_adv_scalars[i]
                if not cont_rd.token_ids:
                    continue
                resp_len = len(cont_rd.token_ids)
                cont_lps = cont_rd.log_probs[:resp_len] if cont_rd.log_probs else [0.0] * resp_len
                examples.append(TrainingExample(
                    input_ids=cont_prompt_ids + cont_rd.token_ids,
                    prompt_len=len(cont_prompt_ids),
                    token_advantages=[c_adv] * resp_len,
                    old_log_probs=cont_lps,
                    example_type="continuation",
                    refl_round=r,
                ))
                all_cont_advs.append(c_adv)

    # Compute statistics
    refl_acc_list = []
    for rollout in refl_rollouts:
        for r in range(len(rollout.refl_tags)):
            cont_accs = rollout.cont_accuracies[r] if r < len(rollout.cont_accuracies) else []
            refl_acc_list.append(
                (sum(cont_accs) / len(cont_accs)) if cont_accs else rollout.final_accuracy
            )

    stats = {
        "dir_acc": dir_acc,
        "mean_refl_acc": (sum(refl_acc_list) / len(refl_acc_list)) if refl_acc_list else 0.0,
        "trigger_freq": trigger_freq,
        "mean_refl_reward": (sum(all_refl_rewards) / len(all_refl_rewards)) if all_refl_rewards else 0.0,
        "mean_pre_reward": (sum(all_pre_rewards) / len(all_pre_rewards)) if all_pre_rewards else 0.0,
        "mean_cont_advantage": (sum(all_cont_advs) / len(all_cont_advs)) if all_cont_advs else 0.0,
        "total_reflection_count": sum(refl_count_per_round),
        "refl_count_per_round": refl_count_per_round,
    }
    return examples, stats


# ── verl Integration: Advantage Computation ────────────────────────────────────

def compute_rgrpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    reflection_info: Optional[Dict[str, Any]] = None,
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for R-GRPO, handling both direct and reflection rollouts.
    
    For standard trajectories (no reflection), uses GRPO-style advantage.
    For reflection trajectories, applies token-level advantages based on segment type.
    
    Args:
        token_level_rewards: shape (bs, response_length)
        response_mask: shape (bs, response_length)
        index: unique problem identifier for each sample
        traj_index: trajectory index within problem
        reflection_info: Optional dict containing reflection-specific data:
            - 'has_reflection': bool array indicating which samples have reflection
            - 'segment_types': array of segment type labels ('pre_refl', 'refl', 'cont', 'direct')
            - 'segment_boundaries': list of (start, end) tuples for each segment
            - 'segment_rewards': pre-computed rewards for each segment
        epsilon: small value for numerical stability
    
    Returns:
        advantages: shape (bs, response_length)
        returns: shape (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)
    batch_size, response_length = token_level_rewards.shape
    device = token_level_rewards.device
    
    # If no reflection info, fall back to standard GRPO
    if reflection_info is None:
        return _compute_standard_grpo_advantage(
            token_level_rewards, response_mask, index, traj_index, epsilon
        )
    
    # Initialize output tensors
    advantages = torch.zeros_like(token_level_rewards)
    returns = torch.zeros_like(token_level_rewards)
    
    # Group by problem index
    id2indices = defaultdict(list)
    for i, idx in enumerate(index):
        id2indices[idx].append(i)
    
    has_reflection = reflection_info.get('has_reflection', np.zeros(batch_size, dtype=bool))
    segment_rewards = reflection_info.get('segment_rewards', {})
    
    for problem_idx, sample_indices in id2indices.items():
        # Separate direct and reflection samples
        direct_indices = [i for i in sample_indices if not has_reflection[i]]
        refl_indices = [i for i in sample_indices if has_reflection[i]]
        
        # Compute direct rollout advantage (standard GRPO)
        if direct_indices:
            direct_scores = scores[direct_indices]
            direct_mean = direct_scores.mean()
            direct_std = direct_scores.std(unbiased=False) + epsilon
            direct_advantages = (direct_scores - direct_mean) / direct_std
            
            for j, sample_idx in enumerate(direct_indices):
                mask = response_mask[sample_idx]
                advantages[sample_idx] = direct_advantages[j] * mask
                returns[sample_idx] = token_level_rewards[sample_idx].sum() * mask
        
        # Compute reflection rollout advantages
        if refl_indices:
            # Get pre-computed segment rewards for this problem
            problem_segment_rewards = segment_rewards.get(problem_idx, {})
            
            for sample_idx in refl_indices:
                mask = response_mask[sample_idx]
                sample_key = f"{problem_idx}_{sample_idx}"
                
                if sample_key in problem_segment_rewards:
                    # Use pre-computed per-token advantages
                    token_advs = problem_segment_rewards[sample_key]
                    adv_tensor = torch.tensor(token_advs, device=device, dtype=advantages.dtype)
                    if len(adv_tensor) < response_length:
                        adv_tensor = torch.nn.functional.pad(
                            adv_tensor, (0, response_length - len(adv_tensor))
                        )
                    advantages[sample_idx] = adv_tensor[:response_length] * mask
                else:
                    # Fallback to standard advantage
                    group_scores = scores[sample_indices]
                    group_std = group_scores.std(unbiased=False) + epsilon
                    advantages[sample_idx] = (scores[sample_idx] - group_scores.mean()) / group_std * mask
                
                returns[sample_idx] = token_level_rewards[sample_idx].sum() * mask
    
    return advantages, returns


def _compute_standard_grpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Standard GRPO advantage computation (fallback)."""
    scores = token_level_rewards.sum(dim=-1)
    batch_size, response_length = token_level_rewards.shape
    device = token_level_rewards.device
    
    advantages = torch.zeros_like(token_level_rewards)
    returns = torch.zeros_like(token_level_rewards)
    
    # Group by problem index
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    
    for i, idx in enumerate(index):
        id2score[idx].append(scores[i].item())
    
    for idx, score_list in id2score.items():
        id2mean[idx] = np.mean(score_list)
        id2std[idx] = np.std(score_list) + epsilon
    
    for i in range(batch_size):
        idx = index[i]
        adv = (scores[i] - id2mean[idx]) / id2std[idx]
        mask = response_mask[i]
        advantages[i] = adv * mask
        returns[i] = scores[i] * mask
    
    return advantages, returns


# ── Utility Functions ──────────────────────────────────────────────────────────

def extract_boxed(text: str) -> Optional[str]:
    """Extract content of the LAST \\boxed{...} in text. Returns None if not found."""
    if "\\boxed " in text:
        val = text.split("\\boxed ")[-1].split("$")[0].strip()
        if val:
            return val
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return None
    i, depth, right = idx + len("\\boxed{") - 1, 0, None
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                right = i
                break
        i += 1
    return text[idx + len("\\boxed{"):right] if right is not None else None


def _last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last \\boxed{...} expression from string."""
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    return None if right_brace_idx is None else string[idx: right_brace_idx + 1]


def _remove_boxed(s: str) -> str:
    """Remove \\boxed{} or \\boxed  wrapper."""
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left):]
    left = "\\boxed{"
    assert s[: len(left)] == left
    assert s[-1] == "}"
    return s[len(left): -1]


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:
                    return string
                a, b = substr[0], substr[1]
                if b != "{":
                    post = substr[2:]
                    new_str += "{" + a + "}{" + b + "}" + post
                else:
                    new_str += "{" + a + "}" + b + (substr[2:] if len(substr) > 2 else "")
    return new_str


def _fix_a_slash_b(string: str) -> str:
    parts = string.split("/")
    if len(parts) != 2:
        return string
    try:
        a, b = int(parts[0]), int(parts[1])
        assert string == "{}/{}".format(a, b)
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except Exception:
        return string


def _remove_right_units(string: str) -> str:
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    return string


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            new_string += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def _strip_string(string: str) -> str:
    """Normalize a mathematical expression string for comparison."""
    string = string.replace("\n", "").replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "").replace("\%", "")  # noqa: W605
    string = string.replace(" .", " 0.").replace("{.", "{0.")
    if not string:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


def _is_equiv(str1: Optional[str], str2: Optional[str]) -> bool:
    """Check mathematical equivalence after normalization."""
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False
    try:
        return _strip_string(str1) == _strip_string(str2)
    except Exception:
        return str1 == str2


def compute_math_score(solution_str: str = None, ground_truth: str = None, generated: str = None, data_source: str = None, extra_info: dict = None, **kwargs) -> float:
    """Compute accuracy score for math problems.
    Accepts both positional and keyword args for compatibility with NaiveRewardManager.
    Uses robust boxed-answer extraction + mathematical normalization (same as grpo/core_grpo.py).
    """
    # Support both calling conventions
    if solution_str is not None and generated is None:
        generated = solution_str
    elif generated is None:
        generated = ""
    if ground_truth is None:
        ground_truth = ""
    if isinstance(ground_truth, dict):
        ground_truth = ground_truth.get("answer", "")
    try:
        answer = extract_boxed(generated)
        if answer is not None and _is_equiv(answer, ground_truth):
            return 1.0
    except Exception:
        pass
    return 0.0
