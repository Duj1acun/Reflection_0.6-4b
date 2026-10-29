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
Hybrid-GRPO: 统一奖励设计，结合 R-GRPO 的续推采样结构与分段优势。

奖励设计（同一问题组 = direct 样本 + reflection event）：

  MAX_REFLECTIONS = 3,  PENALTY_BASE = 3
  penalty(n) = (n / 3)^2    n=1→0.1111, n=2→0.4444, n=3→1.0

  直接样本（无反思）：
    回答正确  → +1.0
    回答错误  → -(3/3)^2 = -1.0  （= -penalty(MAX_REFLECTIONS））

  续推 joint 样本 = prefix + continuation，其中：
    prefix = 反思前文本 + 反思文本 + 更早轮次累积到当前叶子的前缀
    cont   = 最后一次叶子续推文本

    refl_reward = 1.0  若该反思事件的 k 条续推中有任意一条正确，否则 0.0
    cont_reward = 1.0  若本条续推回答正确，否则 0.0
    event_raw   = refl_reward - penalty(n_refl)

  两层 GRPO 归一化：
    1) 组级主池：对 [direct_raws ..., event_raws ...] 做 z-score，
       得到 direct_adv / event_adv
    2) event 内局部池：对同一反思 event 下的 [cont_reward_1, ..., cont_reward_k]
       做 z-score，得到 local_cont_adv
       若同一 event 的 continuation 全错或全对，则整个 event 视为 no-signal，
       prefix / continuation 均不回传优势

  token 赋值：
    direct token 使用 direct_adv
    prefix token 使用 event_adv
    continuation token 使用 local_cont_adv

指标设计：
  训练指标（每步）：
    hybrid/trigger_rate              反思触发率（有续推样本的问题比例）
    hybrid/n_direct_samples          直接样本数
    hybrid/n_cont_samples            续推样本数
    hybrid/n_refl_events             反思事件数（独立的反思 group 数）
    hybrid/direct_acc                直接样本正确率
    hybrid/cont_acc                  续推样本正确率
    hybrid/refl_any_correct_rate     反思事件中至少有一条续推正确的比例（refl_reward 的均值）
    hybrid/reward/direct_mean        直接样本原始奖励均值
    hybrid/reward/event_mean         反思事件级原始奖励均值（event_raw）
    hybrid/reward/prefix_raw_mean    prefix 段原始奖励均值（refl - penalty）
    hybrid/reward/cont_mean          续推 joint 样本总原始奖励均值（prefix + cont）
    hybrid/reward/refl_component     续推样本的 refl_reward 分量均值
    hybrid/reward/cont_component     续推样本的 cont_reward 分量均值
    hybrid/reward/penalty_mean       续推样本的惩罚项均值
    hybrid/n_refl_depth/mean         续推样本的平均反思轮数
    hybrid/adv/event_mean            主池中 reflection event 优势均值
    hybrid/adv/cont_local_abs_mean   continuation 局部优势绝对值均值
    hybrid/adv/cont_local_pos_rate   continuation 局部优势为正的比例
    hybrid/adv/cont_local_neg_rate   continuation 局部优势为负的比例
    hybrid/adv/cont_local_zero_rate  continuation 局部优势约为 0 的比例
    hybrid/local_signal/zero_event_rate  event 内 continuation 奖励无差异的比例
    hybrid/prefix_mode/split_rate    能可靠切分 prefix/cont 的续推样本比例
    hybrid/prefix_mode/whole_rate    切分点无效、整条只用 event_adv 的比例
    hybrid/prefix_mode/cont_only_rate prefix_len=0，仅 continuation 段参与优化的比例

  验证指标（来自 compute_hybrid_math_score 返回的字段，由 veRL 自动聚合）：
    val/*/test_score                 = accuracy（0/1 均值）
    val/*/is_correct                 正确率
    val/*/has_reflection             验证时触发反思的比例
    val/*/n_reflections              平均反思次数
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from .core_rgrpo import compute_math_score, REFL_PATTERN

# ── 超参数 ──────────────────────────────────────────────────────────────────────
MAX_REFLECTIONS = 3
PENALTY_BASE = 3      # penalty(n) = (n / PENALTY_BASE)^2
EPS = 1e-8

# 直接答对/答错的奖励。
_DIRECT_CORRECT_REWARD = 1.0
# 直接答错的奖励 = -(反思 MAX_REFLECTIONS 次的惩罚)
_DIRECT_WRONG_REWARD = -((MAX_REFLECTIONS / PENALTY_BASE) ** 2)  # -(3/3)^2 = -1.0


# ── 辅助函数 ────────────────────────────────────────────────────────────────────

def _penalty(n_refl: int) -> float:
    """反思次数惩罚项：penalty(n) = (n / 3)^2
    n=1 → 0.1111,  n=2 → 0.4444,  n=3 → 1.0
    """
    return (min(n_refl, MAX_REFLECTIONS) / PENALTY_BASE) ** 2


def _grpo_norm(rewards: list) -> list:
    """对奖励列表做 z-score 归一化。"""
    if not rewards:
        return []
    if len(rewards) == 1:
        return [0.0]
    mean_r = sum(rewards) / len(rewards)
    var_r = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
    std_r = math.sqrt(var_r)
    return [(r - mean_r) / (std_r + EPS) for r in rewards]


def _safe_mean(lst: list) -> float:
    return float(sum(lst) / len(lst)) if lst else float("nan")


def _is_no_signal_event(rewards: list) -> bool:
    """All-wrong / all-correct events provide no relative continuation signal."""
    if not rewards:
        return True
    if len(rewards) == 1:
        return True
    first = rewards[0]
    return all(abs(r - first) <= EPS for r in rewards)


def _normalize_prefix_len(prefix_len: Any, valid_len: int) -> Optional[int]:
    """Validate continuation split point within the visible response span."""
    if valid_len <= 0:
        return None
    try:
        prefix_len = int(prefix_len)
    except (TypeError, ValueError):
        return None
    if prefix_len < 0 or prefix_len >= valid_len:
        return None
    return prefix_len


# ── 主函数：组级混合奖励优势计算 ───────────────────────────────────────────────

def compute_hybrid_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    hybrid_info: Optional[Dict[str, Any]] = None,
    epsilon: float = 1e-6,
    metrics_out: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Hybrid-GRPO 优势计算。

    参数
    ----
    token_level_rewards : shape (bs, response_length)
    response_mask       : shape (bs, response_length)
    index               : np.ndarray, shape (bs,) — 问题 uid
    hybrid_info         : dict，包含来自 non_tensor_batch 的 hybrid 字段
    epsilon             : 数值稳定项
    metrics_out         : 若传入可变 dict，会填充训练统计指标

    返回
    ----
    advantages : shape (bs, response_length)
    returns    : shape (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)          # (bs,) 每样本标量分数
    batch_size = token_level_rewards.shape[0]

    advantages = torch.zeros_like(token_level_rewards)
    returns = torch.zeros_like(token_level_rewards)

    # 如果没有 hybrid_info，退化为标准 GRPO
    if hybrid_info is None:
        return _fallback_grpo(token_level_rewards, response_mask, index, epsilon)

    refl_group_ids = hybrid_info.get("hybrid_refl_group_id", None)
    n_refl_arr     = hybrid_info.get("hybrid_n_refl",       np.zeros(batch_size, dtype=np.float32))
    roles          = hybrid_info.get("hybrid_sample_role",  None)
    prefix_len_arr = hybrid_info.get("hybrid_prefix_len",   None)

    # ── 全局统计累积器 ────────────────────────────────────────────────────────
    stat_direct_correct:   list = []   # 1/0 per direct sample
    stat_direct_rewards:   list = []   # raw reward per direct sample
    stat_cont_correct:     list = []   # 1/0 per continuation sample
    stat_prefix_raw:       list = []   # prefix_raw per continuation sample
    stat_cont_rewards:     list = []   # total raw reward per continuation sample
    stat_refl_rewards:     list = []   # refl_reward (0/1) per continuation
    stat_cont_only:        list = []   # cont_reward (0/1) per continuation
    stat_penalties:        list = []   # penalty per continuation
    stat_n_refl:           list = []   # n_refl depth per continuation
    stat_event_rewards:    list = []   # event_raw per reflection event
    stat_event_advs:       list = []   # event_adv per reflection event
    stat_cont_local_advs:  list = []   # local continuation adv per continuation
    stat_zero_local_events:list = []   # 1 if local continuation rewards are all equal within event
    refl_event_keys:       set  = set()  # unique (uid, rg_id) events
    refl_event_correct:    set  = set()  # events where any cont was correct
    n_split_mode = 0
    n_whole_mode = 0
    n_cont_only_mode = 0

    # 按 uid 分组
    uid_to_indices: Dict[Any, list] = defaultdict(list)
    for i, uid in enumerate(index):
        uid_to_indices[uid].append(i)

    for uid, sample_indices in uid_to_indices.items():
        # ── 1. 分类样本 ──────────────────────────────────────────────────────
        direct_indices: list = []
        cont_indices:   list = []  # (sample_idx, refl_group_id, n_refl, prefix_len)

        for i in sample_indices:
            role = str(roles[i]) if roles is not None else (
                "direct" if (refl_group_ids is None or refl_group_ids[i] == -1) else "continuation"
            )
            if role == "direct":
                direct_indices.append(i)
            elif role == "continuation":
                rg_id  = int(refl_group_ids[i]) if refl_group_ids is not None else -1
                n_refl = int(n_refl_arr[i])
                raw_prefix_len = None
                if prefix_len_arr is not None:
                    try:
                        raw_prefix_len = prefix_len_arr[i]
                    except (IndexError, TypeError, KeyError):
                        raw_prefix_len = None
                cont_indices.append((i, rg_id, n_refl, raw_prefix_len))

        # ── 2. 计算直接样本奖励 ───────────────────────────────────────────────
        direct_rewards: Dict[int, float] = {}
        for i in direct_indices:
            is_correct = scores[i].item() > 0.5
            r = _DIRECT_CORRECT_REWARD if is_correct else _DIRECT_WRONG_REWARD
            direct_rewards[i] = r
            stat_direct_correct.append(float(is_correct))
            stat_direct_rewards.append(r)

        # ── 3. 计算 reflection event 奖励与 continuation 局部优势 ─────────────
        group_any_correct: Dict[int, bool] = defaultdict(bool)
        event_to_samples: Dict[int, list] = defaultdict(list)
        for (i, rg_id, _, _) in cont_indices:
            key = (uid, rg_id)
            refl_event_keys.add(key)
            event_to_samples[rg_id].append(i)
            if scores[i].item() > 0.5:
                group_any_correct[rg_id] = True
                refl_event_correct.add(key)

        global_keys = []
        global_raws = []
        segment_modes: Dict[int, str] = {}
        split_prefix_len: Dict[int, int] = {}
        direct_adv_map: Dict[int, float] = {}
        event_adv_map: Dict[int, float] = {}
        cont_local_adv: Dict[int, float] = {}

        for i in direct_indices:
            global_keys.append(("direct", i))
            global_raws.append(direct_rewards[i])

        for rg_id, sample_ids in event_to_samples.items():
            n_refl = next(n_refl for (i, cur_rg_id, n_refl, _) in cont_indices if cur_rg_id == rg_id)
            refl_reward = 1.0 if group_any_correct[rg_id] else 0.0
            pen = _penalty(n_refl)
            if refl_reward > 0.5:
                event_raw = refl_reward - pen   # 反思成功：保留原有的递减奖励
            else:
                event_raw = -_penalty(MAX_REFLECTIONS)  # 反思失败：与直接错误相同的重罚
            stat_event_rewards.append(event_raw)

            local_raws = []
            for (i, cur_rg_id, _, raw_prefix_len) in cont_indices:
                if cur_rg_id != rg_id:
                    continue

                cont_reward = 1.0 if scores[i].item() > 0.5 else 0.0
                total_raw = min(event_raw + cont_reward, 2.0)

                stat_cont_correct.append(cont_reward)
                stat_prefix_raw.append(event_raw)
                stat_cont_rewards.append(total_raw)
                stat_refl_rewards.append(refl_reward)
                stat_cont_only.append(cont_reward)
                stat_penalties.append(pen)
                stat_n_refl.append(float(n_refl))

                valid_len = int(response_mask[i].sum().item())
                prefix_len = _normalize_prefix_len(raw_prefix_len, valid_len)
                if prefix_len is None:
                    segment_modes[i] = "whole"
                    n_whole_mode += 1
                elif prefix_len == 0:
                    segment_modes[i] = "cont_only"
                    n_cont_only_mode += 1
                else:
                    segment_modes[i] = "split"
                    split_prefix_len[i] = prefix_len
                    n_split_mode += 1

                local_raws.append(cont_reward)

            no_signal_event = _is_no_signal_event(local_raws)
            if not no_signal_event:
                global_keys.append(("event", rg_id))
                global_raws.append(event_raw)

            local_normed = [0.0] * len(local_raws) if no_signal_event else _grpo_norm(local_raws)
            stat_zero_local_events.append(float(no_signal_event))
            for sample_idx, adv in zip(sample_ids, local_normed):
                cont_local_adv[sample_idx] = adv
                stat_cont_local_advs.append(adv)

        # ── 4. 组级主池归一化：direct vs reflection event ───────────────────
        normed = _grpo_norm(global_raws) if global_keys else []
        for seg_key, adv in zip(global_keys, normed):
            if seg_key[0] == "direct":
                direct_adv_map[seg_key[1]] = adv
            else:
                event_adv_map[seg_key[1]] = adv
                stat_event_advs.append(adv)

        # ── 5. 写入优势张量 ──────────────────────────────────────────────────
        for i in direct_indices:
            mask = response_mask[i]
            direct_adv = direct_adv_map.get(i, 0.0)
            advantages[i] = direct_adv * mask
            returns[i]    = scores[i] * mask

        for (i, rg_id, _, _) in cont_indices:
            mask = response_mask[i]
            mode = segment_modes.get(i, "whole")
            event_adv = event_adv_map.get(rg_id, 0.0)
            local_adv = cont_local_adv.get(i, 0.0)
            cont_adv = local_adv

            if mode == "split":
                prefix_len = min(split_prefix_len[i], int(mask.sum().item()))
                adv_tensor = torch.zeros_like(mask, dtype=advantages.dtype)
                adv_tensor[:prefix_len] = event_adv
                adv_tensor[prefix_len:] = cont_adv
                advantages[i] = adv_tensor * mask
            elif mode == "cont_only":
                advantages[i] = cont_adv * mask
            else:
                # Without a reliable split point, keep the shared event signal only.
                advantages[i] = event_adv * mask
            returns[i] = scores[i] * mask

    # ── 6. 填充指标字典 ───────────────────────────────────────────────────────
    if metrics_out is not None:
        n_d = len(stat_direct_correct)
        n_c = len(stat_cont_correct)
        n_e = len(refl_event_keys)
        total = n_d + n_c

        metrics_out["hybrid/n_direct_samples"]      = n_d
        metrics_out["hybrid/n_cont_samples"]        = n_c
        metrics_out["hybrid/n_refl_events"]         = n_e
        metrics_out["hybrid/trigger_rate"]          = n_c / total if total > 0 else 0.0

        metrics_out["hybrid/direct_acc"]            = _safe_mean(stat_direct_correct)
        metrics_out["hybrid/cont_acc"]              = _safe_mean(stat_cont_correct)

        metrics_out["hybrid/refl_any_correct_rate"] = (
            len(refl_event_correct) / n_e if n_e > 0 else float("nan")
        )

        metrics_out["hybrid/reward/direct_mean"]    = _safe_mean(stat_direct_rewards)
        metrics_out["hybrid/reward/event_mean"]     = _safe_mean(stat_event_rewards)
        metrics_out["hybrid/reward/prefix_raw_mean"] = _safe_mean(stat_prefix_raw)
        metrics_out["hybrid/reward/cont_mean"]      = _safe_mean(stat_cont_rewards)
        metrics_out["hybrid/reward/refl_component"] = _safe_mean(stat_refl_rewards)
        metrics_out["hybrid/reward/cont_component"] = _safe_mean(stat_cont_only)
        metrics_out["hybrid/reward/penalty_mean"]   = _safe_mean(stat_penalties)
        metrics_out["hybrid/n_refl_depth/mean"]     = _safe_mean(stat_n_refl)
        metrics_out["hybrid/adv/event_mean"]        = _safe_mean(stat_event_advs)
        metrics_out["hybrid/adv/cont_local_abs_mean"] = (
            _safe_mean([abs(x) for x in stat_cont_local_advs])
        )
        metrics_out["hybrid/adv/cont_local_pos_rate"] = (
            _safe_mean([float(x > EPS) for x in stat_cont_local_advs])
        )
        metrics_out["hybrid/adv/cont_local_neg_rate"] = (
            _safe_mean([float(x < -EPS) for x in stat_cont_local_advs])
        )
        metrics_out["hybrid/adv/cont_local_zero_rate"] = (
            _safe_mean([float(abs(x) <= EPS) for x in stat_cont_local_advs])
        )
        metrics_out["hybrid/local_signal/zero_event_rate"] = _safe_mean(stat_zero_local_events)
        metrics_out["hybrid/prefix_mode/split_rate"] = n_split_mode / n_c if n_c > 0 else float("nan")
        metrics_out["hybrid/prefix_mode/whole_rate"] = n_whole_mode / n_c if n_c > 0 else float("nan")
        metrics_out["hybrid/prefix_mode/cont_only_rate"] = (
            n_cont_only_mode / n_c if n_c > 0 else float("nan")
        )

    return advantages, returns


def _fallback_grpo(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """标准 GRPO 退化实现，用于 hybrid_info 缺失时。"""
    scores = token_level_rewards.sum(dim=-1)
    advantages = torch.zeros_like(token_level_rewards)
    returns    = torch.zeros_like(token_level_rewards)

    uid_to_scores: Dict[Any, list] = defaultdict(list)
    for i, uid in enumerate(index):
        uid_to_scores[uid].append(scores[i].item())

    uid_mean = {uid: float(np.mean(v)) for uid, v in uid_to_scores.items()}
    uid_std  = {uid: float(np.std(v)) + epsilon for uid, v in uid_to_scores.items()}

    for i, uid in enumerate(index):
        adv = (scores[i].item() - uid_mean[uid]) / uid_std[uid]
        advantages[i] = adv * response_mask[i]
        returns[i]    = scores[i] * response_mask[i]

    return advantages, returns


# ── 兼容 NaiveRewardManager 的每样本奖励函数 ──────────────────────────────────

def compute_hybrid_math_score(
    solution_str: str = None,
    ground_truth: str = None,
    generated: str = None,
    data_source: str = None,
    extra_info: Optional[dict] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    veRL NaiveRewardManager 兼容的每样本奖励函数。

    训练时：返回 0/1 正确性分数（组级混合奖励在 compute_hybrid_outcome_advantage 中计算）。
    验证时：额外返回 has_reflection / n_reflections / is_correct 字段，
           veRL 框架会自动对这些字段求均值，得到验证集的反思率、正确率等指标。
    """
    del data_source, kwargs

    if solution_str is not None and generated is None:
        generated = solution_str
    elif generated is None:
        generated = ""

    extra_info = extra_info if isinstance(extra_info, dict) else {}

    # 若已预计算，直接返回
    if "hybrid_reward" in extra_info:
        return {"score": float(extra_info["hybrid_reward"])}

    score = float(compute_math_score(solution_str=generated, ground_truth=ground_truth))
    is_correct = score > 0.5

    n_reflections = len(REFL_PATTERN.findall(generated))
    has_reflection = n_reflections > 0

    return {
        "score":          score,                   # 主评分（0/1），驱动 test_score
        "is_correct":     float(is_correct),       # val/*/is_correct
        "has_reflection": float(has_reflection),   # val/*/has_reflection（反思率）
        "n_reflections":  float(n_reflections),    # val/*/n_reflections（平均反思次数）
    }
