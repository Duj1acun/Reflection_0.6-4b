# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import re
import uuid
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Dict, Optional, Type

import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.model import compute_position_id_with_mask
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.rollout.async_server import AsyncLLMServerManager

# Import algorithm-specific modules
try:
    from gigpo import core_gigpo
except ImportError:
    core_gigpo = None
try:
    from rewardflow import core_rewardflow
except ImportError:
    core_rewardflow = None
try:
    from rgrpo import core_rgrpo
except ImportError:
    core_rgrpo = None
try:
    from rgrpo import core_hybrid_grpo
except ImportError:
    core_hybrid_grpo = None

from agent_system.multi_turn_rollout import TrajectoryCollector, adjust_batch

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    GRPO_PASSK = "grpo_passk"
    GiGPO = 'gigpo'
    RewardFlow = 'rewardflow'
    R_GRPO = 'r_grpo'        # Reflective GRPO (per-token segment advantages)
    HYBRID_GRPO = 'hybrid_grpo'  # Hybrid-GRPO (segment advantage on joint continuation trajectories)


def _get_validation_float_array(val_reward_extra_info, reward_size, *keys):
    """Read the first validation field that can be converted to a float array."""
    for key in keys:
        values = val_reward_extra_info.get(key)
        if values is None or len(values) != reward_size:
            continue
        arr = np.asarray(values)
        if arr.size == 0:
            continue
        if arr.dtype == np.bool_:
            return arr.astype(np.float32)
        try:
            return arr.astype(np.float32)
        except (TypeError, ValueError):
            continue
    return None


def _compute_validation_metric_dict(
    reward_tensor,
    data_sources,
    tool_callings,
    traj_uids,
    val_reward_extra_info,
    success_rate,
):
    """Aggregate validation metrics with backward-compatible reward extra-info keys."""
    # evaluate test_score based on data source
    data_source_reward = {}
    for i in range(reward_tensor.shape[0]):
        data_source = data_sources[i]
        if data_source not in data_source_reward:
            data_source_reward[data_source] = []
        data_source_reward[data_source].append(reward_tensor[i].item())

    # evaluate tool call based on data source
    # the values in tool_callings represent the tool call count for each trajectory; however, since the batch is expanded by step, we only need to take one value for each unique trajectories.
    data_source_tool_calling = {}
    unique_traj_uid, unique_idx = np.unique(traj_uids, return_index=True)
    unique_data_sources = data_sources[unique_idx]
    unique_tool_callings = tool_callings[unique_idx]

    for i in range(unique_tool_callings.shape[0]):
        data_source = unique_data_sources[i]
        if data_source not in data_source_tool_calling:
            data_source_tool_calling[data_source] = []
        data_source_tool_calling[data_source].append(unique_tool_callings[i].item())

    reward_size = reward_tensor.shape[0]
    task_reward_arr = _get_validation_float_array(val_reward_extra_info, reward_size, "task_reward")
    eff_reward_arr = _get_validation_float_array(val_reward_extra_info, reward_size, "eff_reward")
    num_reflections_arr = _get_validation_float_array(val_reward_extra_info, reward_size, "num_reflections", "n_reflections")
    n_iters_arr = _get_validation_float_array(val_reward_extra_info, reward_size, "n_iters")
    is_correct_arr = _get_validation_float_array(val_reward_extra_info, reward_size, "is_correct", "task_reward")
    has_reflection_arr = _get_validation_float_array(val_reward_extra_info, reward_size, "has_reflection")

    accuracy_arr = task_reward_arr if task_reward_arr is not None else is_correct_arr
    reflection_flag_arr = has_reflection_arr
    if reflection_flag_arr is None and num_reflections_arr is not None:
        reflection_flag_arr = (num_reflections_arr > 0).astype(np.float32)

    metric_dict = {}
    for data_source, rewards in data_source_reward.items():
        metric_dict[f'val/{data_source}/test_score'] = np.mean(rewards)
        data_mask = data_sources == data_source

        if accuracy_arr is not None:
            ds_acc = accuracy_arr[data_mask]
            if ds_acc.size > 0:
                metric_dict[f'val/{data_source}/accuracy'] = float(ds_acc.mean())
                metric_dict[f'val/{data_source}/reward/task'] = float(ds_acc.mean())

        if is_correct_arr is not None:
            ds_correct = is_correct_arr[data_mask]
            if ds_correct.size > 0:
                metric_dict[f'val/{data_source}/is_correct'] = float(ds_correct.mean())

        if eff_reward_arr is not None:
            ds_eff = eff_reward_arr[data_mask]
            if ds_eff.size > 0:
                metric_dict[f'val/{data_source}/reward/eff'] = float(ds_eff.mean())

        if n_iters_arr is not None:
            ds_iters = n_iters_arr[data_mask]
            if ds_iters.size > 0:
                metric_dict[f'val/{data_source}/n_iters'] = float(ds_iters.mean())

        if has_reflection_arr is not None:
            ds_has_reflection = has_reflection_arr[data_mask]
            if ds_has_reflection.size > 0:
                metric_dict[f'val/{data_source}/has_reflection'] = float(ds_has_reflection.mean())

        if num_reflections_arr is not None:
            ds_num_reflections = num_reflections_arr[data_mask]
            if ds_num_reflections.size > 0:
                metric_dict[f'val/{data_source}/n_reflections'] = float(ds_num_reflections.mean())
                metric_dict[f'val/{data_source}/reflection_count'] = float(ds_num_reflections.mean())

        if reflection_flag_arr is not None:
            ds_refl_flags = reflection_flag_arr[data_mask]
            if ds_refl_flags.size > 0:
                refl_mask = ds_refl_flags > 0.5
                direct_mask = ~refl_mask
                metric_dict[f'val/{data_source}/reflection_rate'] = float(refl_mask.astype(np.float32).mean())
                metric_dict[f'val/{data_source}/direct_solve_rate'] = float(direct_mask.astype(np.float32).mean())
                metric_dict[f'val/{data_source}/n_reflect_samples'] = int(refl_mask.sum())
                metric_dict[f'val/{data_source}/n_direct_samples'] = int(direct_mask.sum())

                if num_reflections_arr is not None:
                    ds_num_reflections = num_reflections_arr[data_mask]
                    metric_dict[f'val/{data_source}/reflection/depth_1_count'] = int(np.sum(ds_num_reflections == 1))
                    metric_dict[f'val/{data_source}/reflection/depth_2_count'] = int(np.sum(ds_num_reflections == 2))
                    metric_dict[f'val/{data_source}/reflection/depth_3plus_count'] = int(np.sum(ds_num_reflections >= 3))
                    if refl_mask.any():
                        metric_dict[f'val/{data_source}/reflection/mean_depth_given_reflection'] = float(
                            ds_num_reflections[refl_mask].mean()
                        )

                if accuracy_arr is not None:
                    ds_acc = accuracy_arr[data_mask]
                    metric_dict[f'val/{data_source}/accuracy/direct'] = float(
                        ds_acc[direct_mask].mean()
                    ) if direct_mask.any() else 0.0
                    metric_dict[f'val/{data_source}/accuracy/reflection'] = float(
                        ds_acc[refl_mask].mean()
                    ) if refl_mask.any() else 0.0

    for data_source, tool_calls in data_source_tool_calling.items():
        metric_dict[f'val/{data_source}/tool_call_count/mean'] = np.mean(tool_calls)
        metric_dict[f'val/{data_source}/tool_call_count/max'] = np.max(tool_calls)
        metric_dict[f'val/{data_source}/tool_call_count/min'] = np.min(tool_calls)

    for k, v in success_rate.items():
        metric_dict[f'val/{k}'] = v

    return metric_dict


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}" + "cannot be satisfied in this ray cluster")


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    if multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics

def apply_invalid_action_penalty(data: DataProto, invalid_action_penalty_coef=float):
    reward_tensor = data.batch['token_level_scores']
    if 'step_rewards' in data.batch.keys():
        step_rewards = data.batch['step_rewards']
    for i in range(len(data)):
        data_item = data[i]  # DataProtoItem

        prompt_ids = data_item.batch['prompts']

        prompt_length = prompt_ids.shape[-1]

        valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()

        action_valids = data_item.non_tensor_batch['is_action_valid'].astype(np.float32)
        action_invalids = torch.tensor(1 - action_valids, dtype=torch.float32, device=prompt_ids.device).squeeze(0)
        # invalid action penalty
        # assert reward_tensor[i, valid_response_length - 1] != 0.0, f'i={i}'
        reward_tensor[i, valid_response_length - 1] -= invalid_action_penalty_coef * action_invalids

        if 'step_rewards' in data.batch.keys():
            step_rewards[i] -= invalid_action_penalty_coef * action_invalids
    
    valid_action_ratio = np.mean(data.non_tensor_batch['is_action_valid'].astype(np.float32)).item()
    metrics = {'episode/valid_action_ratio': valid_action_ratio}
    return data, metrics

def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, step_advantage_w=1.0, gigpo_mode="mean_std_norm", gigpo_enable_similarity=False, gigpo_similarity_thresh=0.95, grpo_skip_std_for_uniform_outcome=False, **kwargs):
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator: The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in GRPO. Defaults to True.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if kwargs.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                kwargs.get("pf_ppo_reweight_method", "pow"),
                kwargs.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # TODO: test on more adv estimator type
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            response_length = grpo_calculation_mask.size(1)  # Get length from the initial response mask
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]  # This mask is the one intended for GRPO
        grpo_outcome_scores = None
        if grpo_skip_std_for_uniform_outcome:
            grpo_outcome_scores = data.non_tensor_batch.get("is_correct", None)
            if grpo_outcome_scores is None:
                grpo_outcome_scores = data.non_tensor_batch.get("task_reward", None)
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            outcome_scores=grpo_outcome_scores,
            skip_std_for_uniform_outcome=grpo_skip_std_for_uniform_outcome,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO_PASSK:
        advantages, returns = core_algos.compute_grpo_passk_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],
            response_mask=data.batch["response_mask"],
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GiGPO:
        if core_gigpo is None:
            raise ImportError("gigpo module not found. Please ensure gigpo package is installed.")
        advantages, returns = core_gigpo.compute_gigpo_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'], # for episode group reward computing
            step_rewards=data.batch['step_rewards'], # for step group reward computing
            response_mask=data.batch['response_mask'],
            anchor_obs=data.non_tensor_batch['anchor_obs'],
            index=data.non_tensor_batch['uid'],
            traj_index=data.non_tensor_batch['traj_uid'],
            step_advantage_w=step_advantage_w,
            mode=gigpo_mode,
            enable_similarity=gigpo_enable_similarity,
            similarity_thresh=gigpo_similarity_thresh,
            )
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.RewardFlow:
        if core_rewardflow is None:
            raise ImportError("rewardflow module not found. Please ensure rewardflow package is installed.")
        anchor_obs = data.non_tensor_batch['anchor_obs']
        advantages, returns = core_rewardflow.compute_rewardflow_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'],
            step_rewards=data.batch['step_rewards'],
            response_mask=data.batch['response_mask'],
            anchor_obs=anchor_obs,
            index=data.non_tensor_batch['uid'],
            traj_index=data.non_tensor_batch['traj_uid'],
        )
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.HYBRID_GRPO:
        if core_hybrid_grpo is None:
            raise ImportError("rgrpo module not found. Please ensure rgrpo package is installed.")
        hybrid_info = {
            "hybrid_refl_group_id": data.non_tensor_batch.get("hybrid_refl_group_id", None),
            "hybrid_n_refl":        data.non_tensor_batch.get("hybrid_n_refl",        None),
            "hybrid_sample_role":   data.non_tensor_batch.get("hybrid_sample_role",   None),
            "hybrid_prefix_len":    data.non_tensor_batch.get("hybrid_prefix_len",    None),
            "reflection_format_valid": data.non_tensor_batch.get(
                "reflection_format_valid", None
            ),
            "hybrid_cf_continue_acc": data.non_tensor_batch.get(
                "hybrid_cf_continue_acc", None
            ),
            "hybrid_prefix_contrast_delta": data.non_tensor_batch.get(
                "hybrid_prefix_contrast_delta", None
            ),
        }
        # metrics_out is filled inside compute_hybrid_outcome_advantage
        _hybrid_metrics: dict = {}
        advantages, returns = core_hybrid_grpo.compute_hybrid_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            hybrid_info=hybrid_info,
            metrics_out=_hybrid_metrics,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"]    = returns
        data.meta_info["hybrid_metrics"] = _hybrid_metrics
    elif adv_estimator == AdvantageEstimator.R_GRPO:
        if core_rgrpo is None:
            raise ImportError("rgrpo module not found. Please ensure rgrpo package is installed.")
        # Reconstruct reflection_info from per-sample arrays stored in non_tensor_batch.
        # Using per-sample storage (rather than a single dict) ensures adjust_batch
        # copy/delete operations correctly propagate reflection data.
        has_refl     = data.non_tensor_batch.get('has_reflection', None)
        token_adv_arr = data.non_tensor_batch.get('rgrpo_token_adv', None)

        if has_refl is not None and token_adv_arr is not None and has_refl.any():
            uids_arr = data.non_tensor_batch['uid']
            # Build nested dict: {uid: {f"{uid}_{i}": token_advs}}
            # so that compute_rgrpo_outcome_advantage can look up by uid first,
            # then by sample key — matching the get(problem_idx, {}) query pattern.
            segment_rewards = {}
            for i in range(len(has_refl)):
                if has_refl[i] and token_adv_arr[i] is not None:
                    uid = uids_arr[i]
                    if uid not in segment_rewards:
                        segment_rewards[uid] = {}
                    segment_rewards[uid][f"{uid}_{i}"] = token_adv_arr[i]
            reflection_info = {
                'has_reflection': has_refl,
                'segment_rewards': segment_rewards,
            }
        else:
            reflection_info = None

        advantages, returns = core_rgrpo.compute_rgrpo_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'],
            response_mask=data.batch['response_mask'],
            index=data.non_tensor_batch['uid'],
            traj_index=data.non_tensor_batch['traj_uid'],
            reflection_info=reflection_info,
        )
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    """Context manager for timing code execution.

    This utility function measures the execution time of code within its context
    and accumulates the timing information in the provided dictionary.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
        traj_collector: TrajectoryCollector = None,
        envs=None,
        val_envs=None,
    ):
        """Initialize distributed PPO trainer with Ray backend."""

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.envs = envs
        self.val_envs = val_envs
        self.traj_collector = traj_collector

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get('lora_rank', 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
            AdvantageEstimator.GiGPO,
            AdvantageEstimator.RewardFlow,
            AdvantageEstimator.R_GRPO,
            AdvantageEstimator.HYBRID_GRPO,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'" + "is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            # assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            print("WARNING: val_batch_size is deprecated." + " Validation datasets are sent to inference engines as a whole batch," + " which will schedule the memory themselves.")

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None, "tool_config_path must be set when enabling multi_turn with tool, due to no role-playing support"
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], "only GRPO is tested for multi-turn with tool"

        if config.algorithm.get("reflection_grpo", {}).get("enable", False):
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], (
                "reflection_grpo requires algorithm.adv_estimator=grpo"
            )
            assert config.algorithm.get("reflection_grpo", {}).get("max_iters", 0) >= 1, (
                "reflection_grpo.max_iters must be >= 1"
            )

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
        if val_dataset is None:
            val_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer, self.processor)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _score_baseline_validate(self):
        reward_score_lst = []
        val_reward_extra_info = defaultdict(list)
        data_source_lst = []
        tool_calling_list = []
        traj_uid_list = []
        success_rate_dict = {}

        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        import time

        max_prompt_length = self._score_baseline_max_prompt_length()
        val_meta_info = {
            "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
            "validate": True,
        }

        for entry, test_data in enumerate(self.val_dataloader):
            start_time = time.time()
            print(f"SCoRe validation set {entry} / {len(self.val_dataloader)} started")
            try:
                test_batch = DataProto.from_single_dict(test_data)
                test_batch = test_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                    interleave=True,
                )
                if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                    return {}
                if "raw_prompt" not in test_batch.non_tensor_batch:
                    raise RuntimeError("SCoRe validation requires data.return_raw_chat=True.")

                input_ids = test_batch.batch["input_ids"]
                input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
                sample_inputs.extend(input_texts)

                source_non_tensor = {key: val for key, val in test_batch.non_tensor_batch.items()}
                raw_messages = source_non_tensor["raw_prompt"]
                messages_list = [
                    messages.tolist() if isinstance(messages, np.ndarray) else deepcopy(list(messages))
                    for messages in raw_messages
                ]

                with _timer("score_val_gen_attempt1", {}):
                    attempt1_raw = self._score_baseline_generate(
                        messages_list,
                        self.actor_rollout_wg,
                        max_prompt_length,
                        meta_info_overrides=val_meta_info,
                    )
                attempt1 = self._score_baseline_attach_metadata(attempt1_raw, source_non_tensor)
                attempt1_texts = self.tokenizer.batch_decode(
                    attempt1.batch["responses"],
                    skip_special_tokens=True,
                )
                reward1, info1 = self._score_baseline_format_and_match(attempt1_texts, attempt1)

                attempt2_messages = self._score_baseline_build_attempt2_messages(raw_messages, attempt1_texts)
                with _timer("score_val_gen_attempt2", {}):
                    attempt2_raw = self._score_baseline_generate(
                        attempt2_messages,
                        self.actor_rollout_wg,
                        max_prompt_length,
                        meta_info_overrides=val_meta_info,
                    )
                attempt2 = self._score_baseline_attach_metadata(attempt2_raw, source_non_tensor)
                attempt2_texts = self.tokenizer.batch_decode(
                    attempt2.batch["responses"],
                    skip_special_tokens=True,
                )
                reward2, info2 = self._score_baseline_format_and_match(attempt2_texts, attempt2)

                reward_score_lst.append(reward2.cpu())
                sample_outputs.extend(attempt2_texts)
                sample_scores.extend(reward2.tolist())

                data_source_lst.append(
                    attempt2.non_tensor_batch.get(
                        "data_source",
                        np.array(["unknown"] * len(attempt2), dtype=object),
                    )
                )
                tool_calling_list.append(attempt2.non_tensor_batch["tool_callings"])
                traj_uid_list.append(attempt2.non_tensor_batch["traj_uid"])

                delta = (reward2 - reward1).tolist()
                val_reward_extra_info["output_attempt1"].extend(attempt1_texts)
                val_reward_extra_info["output_attempt2"].extend(attempt2_texts)
                val_reward_extra_info["score_attempt1"].extend(reward1.tolist())
                val_reward_extra_info["score_attempt2"].extend(reward2.tolist())
                val_reward_extra_info["delta_attempt2_minus_attempt1"].extend(delta)
                val_reward_extra_info["format_think_attempt1"].extend(info1["format_think"])
                val_reward_extra_info["format_think_attempt2"].extend(info2["format_think"])
                val_reward_extra_info["format_answer_attempt1"].extend(info1["format_answer"])
                val_reward_extra_info["format_answer_attempt2"].extend(info2["format_answer"])
                val_reward_extra_info["match_attempt1"].extend(info1["answer_match"])
                val_reward_extra_info["match_attempt2"].extend(info2["answer_match"])
                val_reward_extra_info["extracted_answer_attempt1"].extend(info1["extracted_answer"])
                val_reward_extra_info["extracted_answer_attempt2"].extend(info2["extracted_answer"])
                val_reward_extra_info["target_answer"].extend(info2["target_answer"])
                val_reward_extra_info["task_reward"].extend(info2["answer_match"])
                val_reward_extra_info["is_correct"].extend(info2["answer_match"])

                for key in test_batch.non_tensor_batch.keys():
                    if "success_rate" in key:
                        if key not in success_rate_dict:
                            success_rate_dict[key] = []
                        success_rate_dict[key].append(test_batch.non_tensor_batch[key][0])

                end_time = time.time()
                print(f"SCoRe validation set {entry} finished, used time: {end_time - start_time:.2f} seconds")
            except Exception as e:
                import traceback
                print(f"Error in SCoRe validation entry {entry}: {e}")
                traceback.print_exc()
                continue

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        if not reward_score_lst:
            print("Warning: all SCoRe validation entries failed, skipping validation metrics.")
            return {}
        reward_tensor = torch.cat(reward_score_lst, dim=0).cpu()
        data_sources = np.concatenate(data_source_lst, axis=0)
        tool_callings = np.concatenate(tool_calling_list, axis=0)
        traj_uids = np.concatenate(traj_uid_list, axis=0)
        success_rate = {key: np.mean(value) for key, value in success_rate_dict.items()}

        val_dump_path = os.environ.get(
            "VAL_DUMP_DIR", "artifacts/validation"
        )
        val_dump_infos = {
            "data_source": data_sources.tolist(),
            "traj_uid": traj_uids.tolist(),
            "tool_calling": tool_callings.tolist(),
        }
        for key, value in val_reward_extra_info.items():
            if len(value) == reward_tensor.shape[0]:
                val_dump_infos[key] = list(value)
        self._dump_generations(
            inputs=sample_inputs,
            outputs=sample_outputs,
            scores=reward_tensor.tolist(),
            reward_extra_infos_dict=val_dump_infos,
            dump_path=val_dump_path,
        )

        metric_dict = _compute_validation_metric_dict(
            reward_tensor=reward_tensor,
            data_sources=data_sources,
            tool_callings=tool_callings,
            traj_uids=traj_uids,
            val_reward_extra_info=val_reward_extra_info,
            success_rate=success_rate,
        )
        metric_dict["val/score_baseline/score_attempt1"] = float(np.mean(val_reward_extra_info["score_attempt1"]))
        metric_dict["val/score_baseline/score_attempt2"] = float(np.mean(val_reward_extra_info["score_attempt2"]))
        metric_dict["val/score_baseline/delta_attempt2_minus_attempt1"] = float(
            np.mean(val_reward_extra_info["delta_attempt2_minus_attempt1"])
        )
        metric_dict["val/score_baseline/match_attempt1"] = float(np.mean(val_reward_extra_info["match_attempt1"]))
        metric_dict["val/score_baseline/match_attempt2"] = float(np.mean(val_reward_extra_info["match_attempt2"]))
        return metric_dict

    def _validate(self):
        if self._score_baseline_enabled():
            return self._score_baseline_validate()

        reward_score_lst = []
        val_reward_extra_info = defaultdict(list)
        data_source_lst = []
        tool_calling_list = []
        traj_uid_list = []
        success_rate_dict = {}

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        import time

        validation_eos_token_id = self.tokenizer.eos_token_id
        if isinstance(validation_eos_token_id, int):
            validation_eos_token_ids = [validation_eos_token_id]
        elif isinstance(validation_eos_token_id, (list, tuple)):
            validation_eos_token_ids = [token_id for token_id in validation_eos_token_id if isinstance(token_id, int)]
        else:
            validation_eos_token_ids = []

        im_end_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        unk_token_id = self.tokenizer.unk_token_id
        if (
            isinstance(im_end_token_id, int)
            and im_end_token_id >= 0
            and (unk_token_id is None or im_end_token_id != unk_token_id)
            and im_end_token_id not in validation_eos_token_ids
        ):
            validation_eos_token_ids.append(im_end_token_id)

        if validation_eos_token_ids:
            validation_eos_token_id = validation_eos_token_ids[0] if len(validation_eos_token_ids) == 1 else validation_eos_token_ids

        for entry, test_data in enumerate(self.val_dataloader):
            start_time = time.time()
            print(f"Validation set {entry} / {len(self.val_dataloader)} started")
            try:
                test_batch = DataProto.from_single_dict(test_data)

                # repeat test batch
                test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

                # we only do validation on rule-based rm
                if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                    return {}

                # Store original inputs
                input_ids = test_batch.batch["input_ids"]
                # TODO: Can we keep special tokens except for padding tokens?
                input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
                sample_inputs.extend(input_texts)

                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                if "multi_modal_data" in test_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in test_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in test_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "env_kwargs" in test_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("env_kwargs")
                test_gen_batch = test_batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                test_gen_batch.meta_info = {
                    "eos_token_id": validation_eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "recompute_log_prob": False,
                    "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                    "validate": True,
                }
                print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

                # # pad to be divisible by dp_size
                # test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
                # test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)

                # # unpad
                # test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

                ################ agent-environment loop ###############
                test_output_gen_batch, rewardflow_time = self.traj_collector.multi_turn_loop(
                                                        gen_batch=test_gen_batch,
                                                        actor_rollout_wg=self.actor_rollout_wg,
                                                        envs=self.val_envs,
                                                        is_train=False,
                                                        )
                print('validation generation end')
                # Store generated outputs
                output_ids = test_output_gen_batch.batch["responses"]
                output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
                sample_outputs.extend(output_texts)

                # Copy metadata from original test_batch (e.g. reward_model, data_source)
                # into the generated output, which carries responses but not the original metadata
                for _key, _val in test_batch.non_tensor_batch.items():
                    if _key not in test_output_gen_batch.non_tensor_batch:
                        test_output_gen_batch.non_tensor_batch[_key] = _val
                test_batch = test_output_gen_batch

                # evaluate using reward_function
                result = self.val_reward_fn(test_batch, return_dict=True)
                reward_tensor = result["reward_tensor"]
                reward_extra_info = result.get("reward_extra_info", {})
                sample_reward = reward_tensor.sum(-1).cpu()
                scores = sample_reward.tolist()
                sample_scores.extend(scores)

                if reward_extra_info:
                    for k, v in reward_extra_info.items():
                        val_reward_extra_info[k].extend(list(v))

                reward_score_lst.append(sample_reward)
                data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
                tool_calling_list.append(test_output_gen_batch.non_tensor_batch['tool_callings'])
                traj_uid_list.append(test_output_gen_batch.non_tensor_batch['traj_uid'])
                for k in test_batch.non_tensor_batch.keys():
                    if 'success_rate' in k:
                        if k not in success_rate_dict:
                            success_rate_dict[k] = []
                        success_rate_dict[k].append(test_batch.non_tensor_batch[k][0])
                        # all success_rate should be the same
                        for i in range(1, len(test_batch.non_tensor_batch[k])):
                            assert test_batch.non_tensor_batch[k][0] == test_batch.non_tensor_batch[k][i], f'not all success_rate are the same, 0: {test_batch.non_tensor_batch[k][0]}, {i}: {test_batch.non_tensor_batch[k][i]}'
                end_time = time.time()
                print(f"Validation set {entry} finished, used time: {end_time - start_time:.2f} seconds")
            except Exception as e:
                import traceback
                print(f"Error in validation entry {entry}: {e}")
                traceback.print_exc()
                continue

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        if not reward_score_lst:
            print("Warning: all validation entries failed, skipping validation metrics.")
            return {}
        reward_tensor = torch.cat(reward_score_lst, dim=0).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        tool_callings = np.concatenate(tool_calling_list, axis=0)
        traj_uids = np.concatenate(traj_uid_list, axis=0)
        success_rate = {k: np.mean(v) for k, v in success_rate_dict.items()}

        assert reward_tensor.shape[0] == len(data_sources), (
            f"Validation aggregation mismatch: rewards={reward_tensor.shape[0]}, "
            f"data_sources={len(data_sources)}"
        )
        assert reward_tensor.shape[0] == len(tool_callings), (
            f"Validation aggregation mismatch: rewards={reward_tensor.shape[0]}, "
            f"tool_callings={len(tool_callings)}"
        )
        assert reward_tensor.shape[0] == len(traj_uids), (
            f"Validation aggregation mismatch: rewards={reward_tensor.shape[0]}, "
            f"traj_uids={len(traj_uids)}"
        )

        val_dump_path = os.environ.get(
            "VAL_DUMP_DIR", "artifacts/validation"
        )
        val_dump_infos = {
            "data_source": data_sources.tolist(),
            "traj_uid": traj_uids.tolist(),
            "tool_calling": tool_callings.tolist(),
        }
        for k, v in val_reward_extra_info.items():
            if len(v) == reward_tensor.shape[0]:
                val_dump_infos[k] = list(v)
        self._dump_generations(
            inputs=sample_inputs,
            outputs=sample_outputs,
            scores=reward_tensor.tolist(),
            reward_extra_infos_dict=val_dump_infos,
            dump_path=val_dump_path,
        )

        return _compute_validation_metric_dict(
            reward_tensor=reward_tensor,
            data_sources=data_sources,
            tool_callings=tool_callings,
            traj_uids=traj_uids,
            val_reward_extra_info=val_reward_extra_info,
            success_rate=success_rate,
        )

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=self.device_name, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config.actor_rollout_ref,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print("Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead")
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1

        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _score_baseline_enabled(self) -> bool:
        return bool(self.config.algorithm.get("score_baseline", {}).get("enable", False))

    @staticmethod
    def _object_array(values) -> np.ndarray:
        arr = np.empty(len(values), dtype=object)
        for idx, value in enumerate(values):
            arr[idx] = value
        return arr

    @staticmethod
    def _score_baseline_normalize_answer(text: str) -> str:
        return str(text).strip().lower()

    @staticmethod
    def _score_baseline_extract_boxed(text: str) -> str | None:
        idx = text.find(r"\boxed{")
        if idx == -1:
            return None
        start = idx + len(r"\boxed{")
        depth = 1
        for pos in range(start, len(text)):
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:pos]
        return None

    @classmethod
    def _score_baseline_peel_math_wrappers(cls, text: str) -> str:
        dollar_wrapper = re.compile(r"^\$+|\$+$")
        display_brackets = re.compile(r"^\\\[(.*)\\\]$", flags=re.DOTALL)
        prev = None
        text = str(text).strip()
        while text != prev:
            prev = text
            text = dollar_wrapper.sub("", text).strip()
            match = display_brackets.match(text)
            if match:
                text = match.group(1).strip()
                continue
            if text.startswith(r"\boxed{"):
                boxed = cls._score_baseline_extract_boxed(text)
                if boxed is not None:
                    text = boxed.strip()
        return text

    @classmethod
    def _score_baseline_math_final_answer(cls, text: str) -> str:
        text = "" if text is None else str(text)
        tag_matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL)
        if tag_matches:
            return cls._score_baseline_peel_math_wrappers(tag_matches[-1].rstrip("."))

        boxed = cls._score_baseline_extract_boxed(text)
        if boxed is not None:
            return cls._score_baseline_peel_math_wrappers(boxed.rstrip("."))

        marker = re.search(r"final answer is:?\s*", text, flags=re.IGNORECASE)
        if marker:
            rest = text[marker.end():]
            match = re.match(r"\$([^$\n]+)\$", rest)
            if match:
                return cls._score_baseline_peel_math_wrappers(match.group(1).rstrip("."))
            match = re.match(r"([^\n]+?)(?=\.\s|\.$|\n|$)", rest)
            if match:
                return cls._score_baseline_peel_math_wrappers(match.group(1).rstrip("."))
        return ""

    @classmethod
    def _score_baseline_target_answer(cls, target) -> str:
        if isinstance(target, dict):
            target = target.get("ground_truth", target.get("answer", target.get("target", "")))
        text = "" if target is None else str(target)
        extracted = cls._score_baseline_math_final_answer(text)
        if extracted:
            return extracted
        return cls._score_baseline_peel_math_wrappers(text.rstrip("."))

    def _score_baseline_targets(self, data: DataProto) -> list[str]:
        targets = []
        non_tensors = data.non_tensor_batch
        for idx in range(len(data)):
            target = None
            if "reward_model" in non_tensors:
                reward_model = non_tensors["reward_model"][idx]
                if isinstance(reward_model, dict):
                    target = reward_model.get("ground_truth", reward_model.get("answer"))
            if target is None and "answer" in non_tensors:
                target = non_tensors["answer"][idx]
            if target is None and "solution" in non_tensors:
                target = non_tensors["solution"][idx]
            targets.append(self._score_baseline_target_answer(target))
        return targets

    def _score_baseline_format_and_match(
        self,
        predictions: list[str],
        data: DataProto,
    ) -> tuple[torch.Tensor, dict[str, list]]:
        targets = self._score_baseline_targets(data)
        scores = []
        info = {
            "score_reward": [],
            "format_think": [],
            "format_answer": [],
            "answer_match": [],
            "extracted_answer": [],
            "target_answer": [],
        }
        think_pair = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
        answer_pair = re.compile(r"<answer>.*?</answer>", flags=re.DOTALL)
        for pred, target in zip(predictions, targets):
            pred = "" if pred is None else str(pred)
            extracted = self._score_baseline_math_final_answer(pred)
            has_think = float(len(think_pair.findall(pred)) >= 1)
            has_one_answer = float(len(answer_pair.findall(pred)) == 1)
            answer_match = float(
                bool(self._score_baseline_normalize_answer(target))
                and self._score_baseline_normalize_answer(extracted)
                == self._score_baseline_normalize_answer(target)
            )
            score = 0.25 * has_think + 0.25 * has_one_answer + 0.5 * answer_match
            scores.append(score)
            info["score_reward"].append(float(score))
            info["format_think"].append(has_think)
            info["format_answer"].append(has_one_answer)
            info["answer_match"].append(answer_match)
            info["extracted_answer"].append(extracted)
            info["target_answer"].append(target)
        return torch.tensor(scores, dtype=torch.float32), info

    @staticmethod
    def _score_baseline_thinking_to_history(text: str) -> str:
        return text

    def _score_baseline_truncate_ids(self, ids: list[int], max_length: int) -> list[int]:
        if len(ids) <= max_length:
            return list(ids)
        truncation = self.config.data.get("truncation", "left")
        if truncation == "left":
            return list(ids[-max_length:])
        if truncation == "right":
            return list(ids[:max_length])
        if truncation == "middle":
            left_half = max_length // 2
            right_half = max_length - left_half
            return list(ids[:left_half] + ids[-right_half:])
        raise RuntimeError(
            f"SCoRe baseline prompt length {len(ids)} exceeds {max_length} "
            f"with truncation={truncation!r}"
        )

    def _score_baseline_max_prompt_length(self) -> int:
        score_cfg = self.config.algorithm.get("score_baseline", {})
        explicit = score_cfg.get("max_prompt_length_attempt2", None)
        if explicit is not None:
            return int(explicit)
        max_model_len = self.config.actor_rollout_ref.rollout.get("max_model_len", None)
        if max_model_len is not None:
            return max(
                int(self.config.data.max_prompt_length),
                int(max_model_len) - int(self.config.data.max_response_length),
            )
        return int(self.config.data.max_prompt_length) + int(self.config.data.max_response_length)

    def _score_baseline_build_prompt_batch(
        self,
        messages_list: list[list[dict]],
        max_prompt_length: int,
        meta_info_overrides: dict | None = None,
    ) -> DataProto:
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        encoded_prompts: list[list[int]] = []
        raw_prompt_ids: list[np.ndarray] = []
        for messages in messages_list:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            ids = self._score_baseline_truncate_ids(ids, max_prompt_length)
            encoded_prompts.append(ids)
            raw_prompt_ids.append(np.asarray(ids, dtype=np.int64))

        input_rows = []
        attention_rows = []
        for ids in encoded_prompts:
            pad_len = max_prompt_length - len(ids)
            input_rows.append([pad_token_id] * pad_len + ids)
            attention_rows.append([0] * pad_len + [1] * len(ids))

        input_ids = torch.tensor(input_rows, dtype=torch.long)
        attention_mask = torch.tensor(attention_rows, dtype=torch.long)
        position_ids = compute_position_id_with_mask(attention_mask)
        meta_info = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": pad_token_id,
            "n": 1,
            "do_sample": True,
            "temperature": self.config.actor_rollout_ref.rollout.temperature,
            "top_p": self.config.actor_rollout_ref.rollout.top_p,
            "top_k": self.config.actor_rollout_ref.rollout.top_k,
        }
        if meta_info_overrides:
            meta_info.update(meta_info_overrides)
        return DataProto.from_dict(
            tensors={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            non_tensors={
                "raw_prompt_ids": self._object_array(raw_prompt_ids),
                "raw_prompt": self._object_array(messages_list),
            },
            meta_info=meta_info,
        )

    def _score_baseline_generate(
        self,
        messages_list: list[list[dict]],
        actor_rollout_wg,
        max_prompt_length: int,
        meta_info_overrides: dict | None = None,
    ) -> DataProto:
        prompt_batch = self._score_baseline_build_prompt_batch(
            messages_list,
            max_prompt_length,
            meta_info_overrides=meta_info_overrides,
        )
        prompt_padded, pad_size = pad_dataproto_to_divisor(prompt_batch, actor_rollout_wg.world_size)
        output_padded = actor_rollout_wg.generate_sequences(prompt_padded)
        return unpad_dataproto(output_padded, pad_size=pad_size)

    def _score_baseline_attach_metadata(
        self,
        generated: DataProto,
        source_non_tensor: dict[str, np.ndarray],
    ) -> DataProto:
        batch_size = len(generated)
        non_tensors = {}
        for key, value in source_non_tensor.items():
            if key in {"raw_prompt_ids", "raw_prompt"}:
                continue
            if isinstance(value, np.ndarray) and value.shape[0] == batch_size:
                non_tensors[key] = value.copy()
        non_tensors["uid"] = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        non_tensors["traj_uid"] = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        non_tensors["tool_callings"] = np.zeros(batch_size, dtype=np.float32)
        return DataProto(batch=generated.batch, non_tensor_batch=non_tensors, meta_info=generated.meta_info)

    @staticmethod
    def _score_baseline_scalar_rewards(reward_result: dict) -> tuple[torch.Tensor, torch.Tensor, dict]:
        reward_tensor = reward_result["reward_tensor"]
        composite = reward_tensor.sum(dim=-1).detach().cpu().float()
        extra = reward_result.get("reward_extra_info", {}) or {}
        task_values = extra.get("task_reward")
        if task_values is not None and len(task_values) == composite.shape[0]:
            task = torch.tensor(np.asarray(task_values, dtype=np.float32), dtype=torch.float32)
        else:
            task = composite
        return task, composite, extra

    def _score_baseline_build_attempt2_messages(
        self,
        raw_messages: np.ndarray,
        attempt1_texts: list[str],
    ) -> list[list[dict]]:
        score_cfg = self.config.algorithm.get("score_baseline", {})
        default_correction_prompt = os.environ.get(
            "SCORE_SELF_CORRECTION_PROMPT",
            "There might be an error in the solution above. Review the prior reasoning,\n"
            "identify any mistakes, and rewrite the solution.\n\n"
            "Use the same format: <think>...</think> for your corrected reasoning, then\n"
            "<answer>...</answer> for your final answer.",
        )
        correction_prompt = score_cfg.get(
            "self_correction_prompt",
            default_correction_prompt,
        )
        messages_list = []
        for messages, answer_text in zip(raw_messages, attempt1_texts):
            if isinstance(messages, np.ndarray):
                messages = messages.tolist()
            messages_copy = deepcopy(list(messages))
            messages_copy.append({"role": "assistant", "content": self._score_baseline_thinking_to_history(answer_text)})
            messages_copy.append({"role": "user", "content": correction_prompt})
            messages_list.append(messages_copy)
        return messages_list

    def _score_baseline_make_update_batch(
        self,
        gen_batch: DataProto,
        metrics: dict,
        timing_raw: dict,
    ) -> DataProto:
        if not self.use_reference_policy:
            raise RuntimeError(
                "SCoRe baseline requires a reference policy. Set "
                "actor_rollout_ref.actor.use_kl_loss=True with kl_loss_coef=0.0 "
                "so verl instantiates Role.RefPolicy."
            )
        if "raw_prompt" not in gen_batch.non_tensor_batch:
            raise RuntimeError("SCoRe baseline requires data.return_raw_chat=True.")

        max_prompt_length = self._score_baseline_max_prompt_length()
        raw_messages = gen_batch.non_tensor_batch["raw_prompt"]
        messages_list = [
            messages.tolist() if isinstance(messages, np.ndarray) else deepcopy(list(messages))
            for messages in raw_messages
        ]

        with _timer("score_gen_attempt1", timing_raw):
            attempt1_raw = self._score_baseline_generate(messages_list, self.actor_rollout_wg, max_prompt_length)
        attempt1 = self._score_baseline_attach_metadata(attempt1_raw, gen_batch.non_tensor_batch)

        with _timer("score_reward_attempt1", timing_raw):
            attempt1_texts = self.tokenizer.batch_decode(attempt1.batch["responses"], skip_special_tokens=True)
            reward1, info1 = self._score_baseline_format_and_match(attempt1_texts, attempt1)
        score1 = reward1

        attempt2_messages = self._score_baseline_build_attempt2_messages(raw_messages, attempt1_texts)

        with _timer("score_gen_attempt2", timing_raw):
            attempt2_raw = self._score_baseline_generate(attempt2_messages, self.actor_rollout_wg, max_prompt_length)
        attempt2 = self._score_baseline_attach_metadata(attempt2_raw, gen_batch.non_tensor_batch)

        with _timer("score_reward_attempt2", timing_raw):
            attempt2_texts = self.tokenizer.batch_decode(attempt2.batch["responses"], skip_special_tokens=True)
            reward2, info2 = self._score_baseline_format_and_match(attempt2_texts, attempt2)
        score2 = reward2

        score_cfg = self.config.algorithm.get("score_baseline", {})
        stage1_steps = int(score_cfg.get("stage1_steps", max(1, int(self.total_training_steps) // 2)))
        stage = 1 if self.global_steps <= stage1_steps else 2
        beta1 = float(score_cfg.get("beta1", 0.01))
        beta2 = float(score_cfg.get("beta2", 0.1))
        alpha = float(score_cfg.get("alpha", 10.0))

        zeros = torch.zeros_like(reward1)
        if stage == 1:
            pg_weight = torch.cat([zeros, reward2], dim=0)
            kl_coef = torch.cat([torch.full_like(reward1, beta2), zeros], dim=0)
        else:
            shaped_reward2 = reward2 + alpha * (reward2 - reward1)
            pg_weight = torch.cat([reward1, shaped_reward2], dim=0)
            kl_coef = torch.cat([torch.full_like(reward1, beta1), torch.full_like(reward2, beta1)], dim=0)

        response_len = int(self.config.data.max_response_length)
        a1_len = attempt1.batch["attention_mask"][:, -response_len:].sum(-1).float()
        a2_len = attempt2.batch["attention_mask"][:, -response_len:].sum(-1).float()

        update_batch = DataProto.concat([attempt1, attempt2])
        update_batch.batch["score_pg_weight"] = pg_weight
        update_batch.batch["score_kl_coef"] = kl_coef
        update_batch.batch["score_task_reward"] = torch.cat([reward1, reward2], dim=0)
        update_batch.non_tensor_batch["score_attempt"] = np.array(
            ["attempt1"] * len(attempt1) + ["attempt2"] * len(attempt2),
            dtype=object,
        )
        update_batch.meta_info.update({
            "score_baseline": True,
            "score_stage": stage,
            "score_length_norm": str(score_cfg.get("length_norm", "sequence")),
            "score_max_response_length": int(self.config.data.max_response_length),
            "temperature": self.config.actor_rollout_ref.rollout.temperature,
        })

        n_prompts = len(attempt1)
        interleave_indices = torch.stack(
            (torch.arange(n_prompts), torch.arange(n_prompts, 2 * n_prompts)),
            dim=1,
        ).reshape(-1)
        update_batch.reorder(interleave_indices)

        with _timer("score_ref", timing_raw):
            if not self.ref_in_actor:
                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(update_batch)
            else:
                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(update_batch)
            update_batch = update_batch.union(ref_log_prob)

        metrics.update({
            "score_baseline/stage": float(stage),
            "score_baseline/reward_attempt1": float(reward1.mean().item()),
            "score_baseline/reward_attempt2": float(reward2.mean().item()),
            "score_baseline/composite_attempt1": float(score1.mean().item()),
            "score_baseline/composite_attempt2": float(score2.mean().item()),
            "score_baseline/delta_attempt2_minus_attempt1": float((reward2 - reward1).mean().item()),
            "score_baseline/format_think_attempt1": float(np.mean(info1["format_think"])),
            "score_baseline/format_think_attempt2": float(np.mean(info2["format_think"])),
            "score_baseline/format_answer_attempt1": float(np.mean(info1["format_answer"])),
            "score_baseline/format_answer_attempt2": float(np.mean(info2["format_answer"])),
            "score_baseline/match_attempt1": float(np.mean(info1["answer_match"])),
            "score_baseline/match_attempt2": float(np.mean(info2["answer_match"])),
            "score_baseline/pg_weight_mean": float(pg_weight.mean().item()),
            "score_baseline/kl_coef_mean": float(kl_coef.mean().item()),
            "score_baseline/attempt1_len_mean": float(a1_len.mean().item()),
            "score_baseline/attempt2_len_mean": float(a2_len.mean().item()),
            "score_baseline/attempt1_truncated_frac": float((a1_len >= response_len).float().mean().item()),
            "score_baseline/attempt2_truncated_frac": float((a2_len >= response_len).float().mean().item()),
        })
        return update_batch

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        if "reflection_segment_attention_mask" in batch.batch:
            global_seqlen_lst = batch.batch["reflection_segment_attention_mask"].sum(dim=(1, 2)).tolist()
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "env_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("env_kwargs")
                # For math-only training (no env), reward-relevant fields must flow through gen_batch
                for _reward_key in ["reward_model", "answer", "solution", "extra_info"]:
                    if _reward_key in batch.non_tensor_batch:
                        non_tensor_batch_keys_to_pop.append(_reward_key)
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps
                if self._score_baseline_enabled():
                    with _timer("step", timing_raw):
                        with _timer("score_prepare", timing_raw):
                            batch = self._score_baseline_make_update_batch(
                                gen_batch=gen_batch,
                                metrics=metrics,
                                timing_raw=timing_raw,
                            )

                        if self.config.trainer.balance_batch:
                            self._balance_batch(batch, metrics=metrics)

                        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                        with _timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                        if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                            with _timer("testing", timing_raw):
                                val_metrics: dict = self._validate()
                                if is_last_step:
                                    last_val_metrics = val_metrics
                            metrics.update(val_metrics)

                        if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                            with _timer("save_checkpoint", timing_raw):
                                self._save_checkpoint()

                    metrics.update({
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    })
                    metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                    n_gpus = self.resource_pool_manager.get_n_gpus()
                    metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                    logger.log(data=metrics, step=self.global_steps)

                    progress_bar.update(1)
                    self.global_steps += 1
                    if is_last_step:
                        pprint(f"Final validation metrics: {last_val_metrics}")
                        progress_bar.close()
                        return
                    continue

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):

                        ################ agent-environment loop ###############
                        gen_batch_output, rewardflow_time = self.traj_collector.multi_turn_loop(
                                                                gen_batch=gen_batch,
                                                                actor_rollout_wg=self.actor_rollout_wg,
                                                                envs=self.envs,
                                                                is_train=True,
                                                                )
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    # batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    # # repeat to align with repeated responses in rollout
                    # batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    # batch = batch.union(gen_batch_output)
                    del batch
                    batch = gen_batch_output

                    # ── R-GRPO: extract per-rollout training metrics ───────────
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.R_GRPO:
                        _nt = batch.non_tensor_batch
                        _has_refl = _nt.get('has_reflection', np.zeros(len(batch.batch), dtype=bool))
                        metrics['rgrpo/trigger_rate']     = float(np.mean(_has_refl))
                        metrics['rgrpo/n_refl_samples']   = int(np.sum(_has_refl))
                        metrics['rgrpo/n_direct_samples'] = int(np.sum(~_has_refl.astype(bool)))

                        def _nanstat(nt_key, wandb_prefix):
                            arr = _nt.get(nt_key)
                            if arr is None:
                                return
                            arr = arr.astype(np.float32)
                            valid = arr[~np.isnan(arr)]
                            if len(valid) == 0:
                                return
                            metrics[f'{wandb_prefix}/mean'] = float(np.mean(valid))
                            metrics[f'{wandb_prefix}/max']  = float(np.max(valid))
                            metrics[f'{wandb_prefix}/min']  = float(np.min(valid))

                        _nanstat('rgrpo_initial_response_acc', 'rgrpo/accuracy/initial_response')
                        _nanstat('rgrpo_non_reflect_direct_acc', 'rgrpo/accuracy/non_reflect_direct')
                        _nanstat('rgrpo_online_final_acc', 'rgrpo/accuracy/online_final')
                        _nanstat('rgrpo_direct_reward',   'rgrpo/accuracy/direct')
                        _nanstat('rgrpo_refl_acc',        'rgrpo/accuracy/reflection')
                        _nanstat('rgrpo_reflection_future_mean', 'rgrpo/value/reflection_future_mean')
                        _nanstat('rgrpo_raw_pre_reward',  'rgrpo/reward/pre_refl_raw')
                        _nanstat('rgrpo_raw_refl_reward', 'rgrpo/reward/refl_raw')
                        _nanstat('rgrpo_raw_cont_reward', 'rgrpo/reward/cont_raw')
                        _nanstat('rgrpo_adv_pre',         'rgrpo/advantage/pre_refl')
                        _nanstat('rgrpo_adv_refl',        'rgrpo/advantage/refl')
                        _nanstat('rgrpo_adv_cont',        'rgrpo/advantage/cont')
                        _nanstat('rgrpo_n_pre_tokens',    'rgrpo/tokens/pre_refl_len')
                        _nanstat('rgrpo_n_refl_tokens',   'rgrpo/tokens/refl_len')
                        _nanstat('rgrpo_n_cont_tokens',   'rgrpo/tokens/cont_len')
                        _nanstat('rgrpo_reflection_depth', 'rgrpo/reflection/depth')
                        # ── Diagnostic metrics ─────────────────────────────────
                        _nanstat('rgrpo_delta_acc',        'rgrpo/signal/delta_acc')
                        _nanstat('rgrpo_cont_var',         'rgrpo/signal/cont_var')
                        _nanstat('rgrpo_group_reward_var', 'rgrpo/signal/group_reward_var')
                        _depth_arr = _nt.get('rgrpo_reflection_depth')
                        if _depth_arr is not None:
                            _depth_arr = _depth_arr.astype(np.float32)
                            metrics['rgrpo/reflection/depth_1_count'] = int(np.sum(_depth_arr == 1))
                            metrics['rgrpo/reflection/depth_2_count'] = int(np.sum(_depth_arr == 2))
                            metrics['rgrpo/reflection/depth_3_count'] = int(np.sum(_depth_arr >= 3))
                        # Scalar rates (broadcast arrays — just take mean == value)
                        for _scalar_key, _scalar_wkey in [
                            ('rgrpo_recovery_rate',    'rgrpo/signal/recovery_rate'),
                            ('rgrpo_false_trig_rate',  'rgrpo/signal/false_trigger_rate'),
                            ('rgrpo_all_correct_rate', 'rgrpo/signal/all_correct_rate'),
                            ('rgrpo_all_wrong_rate',   'rgrpo/signal/all_wrong_rate'),
                            ('rgrpo_zero_signal_rate', 'rgrpo/signal/zero_signal_rate'),
                        ]:
                            _arr = _nt.get(_scalar_key)
                            if _arr is not None:
                                _v = float(_arr[~np.isnan(_arr.astype(np.float32))][0]) if len(_arr) > 0 else float('nan')
                                if not np.isnan(_v):
                                    metrics[_scalar_wkey] = _v

                    # ── Hybrid-GRPO: extract per-rollout training metrics ──────
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.HYBRID_GRPO:
                        _nt   = batch.non_tensor_batch
                        _tlr  = batch.batch.get("token_level_rewards", None)
                        _rmsk = batch.batch.get("response_mask", None)

                        _roles   = _nt.get("hybrid_sample_role",   None)
                        _n_refl  = _nt.get("hybrid_n_refl",        None)
                        _rg_ids  = _nt.get("hybrid_refl_group_id", None)
                        _uids    = _nt.get("uid",                   None)

                        if _roles is not None and _tlr is not None:
                            _scores    = _tlr.sum(dim=-1).float().cpu().numpy()
                            _roles_arr = np.array([str(r) for r in _roles])
                            _d_mask    = (_roles_arr == "direct")
                            _c_mask    = (_roles_arr == "continuation")
                            n_d, n_c   = int(_d_mask.sum()), int(_c_mask.sum())
                            total      = n_d + n_c

                            metrics["hybrid/n_direct_samples"] = n_d
                            metrics["hybrid/n_cont_samples"]   = n_c
                            metrics["hybrid/trigger_rate"]     = n_c / total if total > 0 else 0.0

                            if n_d > 0:
                                _d_correct = (_scores[_d_mask] > 0.5).astype(np.float32)
                                metrics["hybrid/direct_acc"] = float(np.mean(_d_correct))
                                _d_rw = np.where(_d_correct > 0.5, 2.0, -((3/5)**2))
                                metrics["hybrid/reward/direct_mean"] = float(np.mean(_d_rw))

                            if n_c > 0 and _n_refl is not None and _rg_ids is not None and _uids is not None:
                                _n_refl_arr = np.asarray(_n_refl, dtype=np.float32)
                                _rg_arr     = np.asarray(_rg_ids)
                                _uid_arr    = np.asarray(_uids)

                                # per-event any_correct flag
                                _group_any: dict = {}
                                for _i in np.where(_c_mask)[0]:
                                    _key = (str(_uid_arr[_i]), int(_rg_arr[_i]))
                                    _group_any[_key] = _group_any.get(_key, False) or bool(_scores[_i] > 0.5)

                                n_events = len(_group_any)
                                metrics["hybrid/n_refl_events"]         = n_events
                                metrics["hybrid/refl_any_correct_rate"] = (
                                    sum(_group_any.values()) / n_events if n_events > 0 else float("nan")
                                )

                                # per-continuation reward components
                                _c_idx = np.where(_c_mask)[0]
                                _rr_list, _cr_list, _pen_list = [], [], []
                                for _i in _c_idx:
                                    _key = (str(_uid_arr[_i]), int(_rg_arr[_i]))
                                    _n_r = int(_n_refl_arr[_i])
                                    _rr  = 1.0 if _group_any.get(_key, False) else 0.0
                                    _cr  = 1.0 if _scores[_i] > 0.5 else 0.0
                                    _pen = (min(_n_r, 3) / 5) ** 2
                                    _rr_list.append(_rr); _cr_list.append(_cr); _pen_list.append(_pen)

                                metrics["hybrid/cont_acc"]               = float(np.mean(_scores[_c_mask] > 0.5))
                                metrics["hybrid/reward/refl_component"]  = float(np.mean(_rr_list))
                                metrics["hybrid/reward/cont_component"]  = float(np.mean(_cr_list))
                                metrics["hybrid/reward/penalty_mean"]    = float(np.mean(_pen_list))
                                metrics["hybrid/reward/cont_mean"]       = float(np.mean(
                                    [rr + cr - pen for rr, cr, pen in zip(_rr_list, _cr_list, _pen_list)]
                                ))

                                _c_depths = _n_refl_arr[_c_mask]
                                metrics["hybrid/n_refl_depth/mean"]          = float(np.mean(_c_depths))
                                metrics["hybrid/n_refl_depth/depth_1_count"] = int(np.sum(_c_depths == 1))
                                metrics["hybrid/n_refl_depth/depth_2_count"] = int(np.sum(_c_depths == 2))
                                metrics["hybrid/n_refl_depth/depth_3_count"] = int(np.sum(_c_depths >= 3))

                        # response token lengths
                        if _rmsk is not None and _roles is not None:
                            _resp_lens = _rmsk.sum(dim=-1).float().cpu().numpy()
                            _roles_arr2 = np.array([str(r) for r in _roles])
                            _dl = _resp_lens[_roles_arr2 == "direct"]
                            _cl = _resp_lens[_roles_arr2 == "continuation"]
                            if len(_dl):
                                metrics["hybrid/tokens/direct_resp_len/mean"] = float(np.mean(_dl))
                            if len(_cl):
                                metrics["hybrid/tokens/cont_resp_len/mean"] = float(np.mean(_cl))

                        # initial / future accuracy from rgrpo_reflection_loop fields
                        for _nk, _mk in [
                            ("rgrpo_initial_response_acc", "hybrid/initial_response_acc"),
                            ("rgrpo_refl_acc",             "hybrid/refl_future_acc"),
                        ]:
                            _arr = _nt.get(_nk)
                            if _arr is not None:
                                _v = np.asarray(_arr, dtype=np.float32)
                                _v = _v[~np.isnan(_v)]
                                if len(_v):
                                    metrics[_mk] = float(np.mean(_v))

                        for _nk, _mk in [
                            ("rgrpo_rollout_total_events", "hybrid/rollout/total_events"),
                            ("rgrpo_rollout_total_branches", "hybrid/rollout/total_branches"),
                            ("rgrpo_rollout_truncated_events", "hybrid/rollout/truncated_events"),
                            ("rgrpo_rollout_truncated_branches", "hybrid/rollout/truncated_branches"),
                            ("rgrpo_rollout_truncated", "hybrid/rollout/truncated"),
                            ("rgrpo_rollout_rounds", "hybrid/rollout/rounds"),
                        ]:
                            _arr = _nt.get(_nk)
                            if _arr is not None and len(_arr) > 0:
                                metrics[_mk] = float(np.asarray(_arr, dtype=np.float32)[0])

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.GiGPO:
                        if core_gigpo is None:
                            raise ImportError("gigpo module not found.")
                        step_rewards_tensor = core_gigpo.compute_step_discounted_returns(
                            batch=batch,
                            gamma=self.config.algorithm.gamma
                        )
                        batch.batch['step_rewards'] = step_rewards_tensor


                    if self.config.algorithm.adv_estimator == AdvantageEstimator.RewardFlow:
                        if core_rewardflow is None:
                            raise ImportError("rewardflow module not found.")
                        step_rewards_tensor = core_rewardflow.compute_step_returns(
                            batch=batch
                        )
                        batch.batch['step_rewards'] = step_rewards_tensor
                    
                    batch = adjust_batch(self.config, batch)

                    batch.batch["response_mask"] = compute_response_mask(batch)
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    if "reflection_segment_attention_mask" in batch.batch:
                        batch.meta_info["global_token_num"] = batch.batch["reflection_segment_attention_mask"].sum(dim=(1, 2)).tolist()

                    with _timer("reward", timing_raw):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]
                            same_shape = (
                                rollout_old_log_probs.shape == actor_old_log_probs.shape
                                and actor_old_log_probs.shape == response_mask.shape
                            )
                            if same_shape:
                                rollout_probs = torch.exp(rollout_old_log_probs)
                                actor_probs = torch.exp(actor_old_log_probs)
                                rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                                rollout_probs_diff = torch.masked_select(
                                    rollout_probs_diff, response_mask.bool()
                                )
                                rollout_probs_diff_max = torch.max(rollout_probs_diff)
                                rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                                rollout_probs_diff_std = torch.std(rollout_probs_diff)
                                metrics.update(
                                    {
                                        "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                        "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                        "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                        "training/rollout_probs_diff_skipped": 0.0,
                                    }
                                )
                            else:
                                metrics.update(
                                    {
                                        "training/rollout_probs_diff_skipped": 1.0,
                                        "training/rollout_log_probs_len": float(
                                            rollout_old_log_probs.size(1)
                                        ),
                                        "training/actor_old_log_probs_len": float(
                                            actor_old_log_probs.size(1)
                                        ),
                                        "training/response_mask_len": float(
                                            response_mask.size(1)
                                        ),
                                    }
                                )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                        if (
                            self.config.algorithm.get("reflection_grpo", {}).get("enable", False)
                            and "loss_mask" in batch.batch
                            and os.environ.get(
                                "TRAJ_GRPO_WRONG_REFLECTION_FULL_CHAIN_MASK", "1"
                            ).strip().lower()
                            not in {"0", "false", "no", "n"}
                        ):
                            _full_chain_flags = batch.non_tensor_batch.get(
                                "wrong_reflection_full_chain_mask", None
                            )
                            if _full_chain_flags is not None:
                                try:
                                    _full_chain_flags = (
                                        np.asarray(_full_chain_flags, dtype=np.float32) > 0.5
                                    )
                                except (TypeError, ValueError):
                                    _full_chain_flags = None
                            if (
                                _full_chain_flags is not None
                                and _full_chain_flags.shape[0] == batch.batch["responses"].size(0)
                                and np.any(_full_chain_flags)
                            ):
                                _resp_len = batch.batch["responses"].size(1)
                                _loss_mask = batch.batch["loss_mask"].clone()
                                _response_mask = batch.batch["attention_mask"][:, -_resp_len:].to(
                                    dtype=_loss_mask.dtype,
                                    device=_loss_mask.device,
                                )
                                _flag_tensor = torch.as_tensor(
                                    _full_chain_flags,
                                    dtype=torch.bool,
                                    device=_loss_mask.device,
                                )
                                _loss_resp = _loss_mask[:, -_resp_len:]
                                _loss_resp[_flag_tensor] = _response_mask[_flag_tensor]
                                _loss_mask[:, -_resp_len:] = _loss_resp
                                batch.batch["loss_mask"] = _loss_mask
                                metrics["reflection_grpo/mask/wrong_reflection_full_chain_rate"] = float(
                                    np.mean(_full_chain_flags.astype(np.float32))
                                )
                                metrics["reflection_grpo/mask/wrong_reflection_full_chain_count"] = int(
                                    np.sum(_full_chain_flags)
                                )
                        if self.config.algorithm.get("reflection_grpo", {}).get("enable", False):
                            _nt = batch.non_tensor_batch

                            def _mean_nt(nt_key, metric_key):
                                arr = _nt.get(nt_key)
                                if arr is None:
                                    return
                                arr = np.asarray(arr)
                                if arr.size == 0:
                                    return
                                if arr.dtype == np.bool_:
                                    metrics[metric_key] = float(arr.astype(np.float32).mean())
                                    return
                                try:
                                    arr = arr.astype(np.float32)
                                except (TypeError, ValueError):
                                    return
                                valid = arr[~np.isnan(arr)]
                                if valid.size > 0:
                                    metrics[metric_key] = float(valid.mean())

                            def _float_nt(nt_key):
                                arr = _nt.get(nt_key)
                                if arr is None:
                                    return None
                                arr = np.asarray(arr)
                                if arr.size == 0:
                                    return None
                                if arr.dtype == np.bool_:
                                    return arr.astype(np.float32)
                                try:
                                    return arr.astype(np.float32)
                                except (TypeError, ValueError):
                                    return None

                            def _mean_by_mask(values, mask):
                                if values is None or values.shape[0] != mask.shape[0] or not mask.any():
                                    return 0.0
                                selected = values[mask]
                                selected = selected[~np.isnan(selected)]
                                return float(selected.mean()) if selected.size > 0 else 0.0

                            _mean_nt("score", "reflection_grpo/reward/composite")
                            _mean_nt("score_before_freq_penalty", "reflection_grpo/reward/before_freq_penalty")
                            _mean_nt("task_reward", "reflection_grpo/reward/task")
                            _mean_nt("eff_reward", "reflection_grpo/reward/eff")
                            _mean_nt("reflection_freq_penalty", "reflection_grpo/reward/reflection_freq_penalty")
                            _mean_nt(
                                "wrong_reflection_freq_penalty",
                                "reflection_grpo/reward/wrong_reflection_freq_penalty",
                            )
                            _mean_nt("num_reflections", "reflection_grpo/reflection_count")
                            _mean_nt("n_iters", "reflection_grpo/n_iters")
                            _mean_nt("is_correct", "reflection_grpo/correct_rate")
                            _mean_nt("reflection_format_valid", "reflection_grpo/format/valid_rate")
                            _mean_nt("format_gate_zero", "reflection_grpo/format/gate_zero_rate")
                            _mean_nt("has_reflection_markup", "reflection_grpo/format/has_markup_rate")
                            _mean_nt("truncated", "reflection_grpo/length/truncated_rate")
                            _mean_nt(
                                "terminated_by_cap",
                                "reflection_grpo/terminated_by_cap_rate",
                            )

                            _num_refl = _float_nt("num_reflections")
                            _task_reward = _float_nt("task_reward")
                            _score = _float_nt("score")
                            _terminated = _float_nt("terminated_by_cap")
                            _format_valid = _float_nt("reflection_format_valid")
                            _has_markup = _float_nt("has_reflection_markup")
                            if _num_refl is not None:
                                refl_mask = _num_refl > 0
                                direct_mask = _num_refl == 0
                                metrics["reflection_grpo/trigger_rate"] = float(refl_mask.astype(np.float32).mean())
                                metrics["reflection_grpo/direct_solve_rate"] = float(direct_mask.astype(np.float32).mean())
                                metrics["reflection_grpo/n_reflect_samples"] = int(refl_mask.sum())
                                metrics["reflection_grpo/n_direct_samples"] = int(direct_mask.sum())
                                metrics["reflection_grpo/reflection/depth_1_count"] = int(np.sum(_num_refl == 1))
                                metrics["reflection_grpo/reflection/depth_2_count"] = int(np.sum(_num_refl == 2))
                                metrics["reflection_grpo/reflection/depth_3plus_count"] = int(np.sum(_num_refl >= 3))
                                metrics["reflection_grpo/reflection/depth_1_rate"] = float(
                                    (_num_refl == 1).astype(np.float32).mean()
                                )
                                metrics["reflection_grpo/reflection/depth_2_rate"] = float(
                                    (_num_refl == 2).astype(np.float32).mean()
                                )
                                metrics["reflection_grpo/reflection/depth_3plus_rate"] = float(
                                    (_num_refl >= 3).astype(np.float32).mean()
                                )
                                if refl_mask.any():
                                    metrics["reflection_grpo/reflection/mean_depth_given_reflection"] = float(
                                        _num_refl[refl_mask].mean()
                                    )
                                if _task_reward is not None and _task_reward.shape[0] == _num_refl.shape[0]:
                                    metrics["reflection_grpo/accuracy/direct"] = float(
                                        _task_reward[direct_mask].mean()
                                    ) if direct_mask.any() else 0.0
                                    metrics["reflection_grpo/accuracy/reflection"] = float(
                                        _task_reward[refl_mask].mean()
                                    ) if refl_mask.any() else 0.0
                                    metrics["reflection_grpo/n_correct_direct"] = int(
                                        np.rint(_task_reward[direct_mask].sum())
                                    ) if direct_mask.any() else 0
                                    metrics["reflection_grpo/n_correct_reflection"] = int(
                                        np.rint(_task_reward[refl_mask].sum())
                                    ) if refl_mask.any() else 0
                                    for _depth_name, _depth_mask in [
                                        ("depth_0", _num_refl == 0),
                                        ("depth_1", _num_refl == 1),
                                        ("depth_2", _num_refl == 2),
                                        ("depth_3plus", _num_refl >= 3),
                                    ]:
                                        metrics[f"reflection_grpo/accuracy/{_depth_name}"] = _mean_by_mask(
                                            _task_reward, _depth_mask
                                        )
                                if _score is not None and _score.shape[0] == _num_refl.shape[0]:
                                    for _depth_name, _depth_mask in [
                                        ("depth_0", _num_refl == 0),
                                        ("depth_1", _num_refl == 1),
                                        ("depth_2", _num_refl == 2),
                                        ("depth_3plus", _num_refl >= 3),
                                    ]:
                                        metrics[f"reflection_grpo/reward/{_depth_name}"] = _mean_by_mask(
                                            _score, _depth_mask
                                        )
                                if _format_valid is not None and _format_valid.shape[0] == _num_refl.shape[0]:
                                    metrics["reflection_grpo/format/valid_rate_given_reflection"] = _mean_by_mask(
                                        _format_valid, refl_mask
                                    )
                                    if _has_markup is not None and _has_markup.shape[0] == _num_refl.shape[0]:
                                        markup_mask = _has_markup > 0.5
                                        if markup_mask.any():
                                            metrics["reflection_grpo/format/valid_rate_given_markup"] = _mean_by_mask(
                                                _format_valid, markup_mask
                                            )
                                if _terminated is not None and _terminated.shape[0] == _num_refl.shape[0]:
                                    term_mask = _terminated > 0.5
                                    metrics["reflection_grpo/n_terminated_by_cap"] = int(term_mask.sum())
                                    if _task_reward is not None and term_mask.any():
                                        metrics["reflection_grpo/accuracy/terminated_by_cap"] = float(
                                            _task_reward[term_mask].mean()
                                        )

                                _uids = _nt.get("uid")
                                _has_group_scores = _score is not None and _score.shape[0] == _num_refl.shape[0]
                                _has_group_correct = _task_reward is not None and _task_reward.shape[0] == _num_refl.shape[0]
                                if _uids is not None and (_has_group_scores or _has_group_correct):
                                    _uid_arr = np.asarray(_uids)
                                    if _uid_arr.shape[0] == _num_refl.shape[0]:
                                        _group_indices = defaultdict(list)
                                        for _idx, _uid in enumerate(_uid_arr):
                                            _group_indices[str(_uid)].append(_idx)

                                        _mixed_reward = []
                                        _all_correct = []
                                        _all_wrong = []
                                        _refl_minus_direct = []
                                        _refl_better = []
                                        _direct_all_wrong_refl_any_correct = []
                                        _direct_any_correct_refl_all_wrong = []
                                        _pair_total = 0
                                        _pair_refl_correct_direct_wrong = 0
                                        _pair_refl_wrong_direct_correct = 0
                                        _correct = (_task_reward > 0.5) if _has_group_correct else None

                                        for _indices in _group_indices.values():
                                            _idx_arr = np.asarray(_indices, dtype=np.int64)
                                            if _has_group_scores:
                                                _reward_vals = _score[_idx_arr]
                                                _reward_vals = _reward_vals[~np.isnan(_reward_vals)]
                                                if _reward_vals.size > 0:
                                                    _mixed_reward.append(
                                                        float(np.max(_reward_vals) - np.min(_reward_vals) > 1e-6)
                                                    )

                                            if _correct is None:
                                                continue

                                            _corr_vals = _correct[_idx_arr]
                                            _all_correct.append(float(np.all(_corr_vals)))
                                            _all_wrong.append(float(not np.any(_corr_vals)))

                                            _direct_idx = _idx_arr[direct_mask[_idx_arr]]
                                            _refl_idx = _idx_arr[refl_mask[_idx_arr]]
                                            if _direct_idx.size == 0 or _refl_idx.size == 0:
                                                continue

                                            _direct_correct = _correct[_direct_idx]
                                            _refl_correct = _correct[_refl_idx]
                                            _direct_acc = float(np.mean(_direct_correct))
                                            _refl_acc = float(np.mean(_refl_correct))
                                            _refl_minus_direct.append(_refl_acc - _direct_acc)
                                            _refl_better.append(float(_refl_acc > _direct_acc))
                                            _direct_all_wrong_refl_any_correct.append(
                                                float((not np.any(_direct_correct)) and np.any(_refl_correct))
                                            )
                                            _direct_any_correct_refl_all_wrong.append(
                                                float(np.any(_direct_correct) and (not np.any(_refl_correct)))
                                            )

                                            _pair_total += int(_direct_correct.size * _refl_correct.size)
                                            _pair_refl_correct_direct_wrong += int(
                                                np.sum(_refl_correct) * np.sum(~_direct_correct)
                                            )
                                            _pair_refl_wrong_direct_correct += int(
                                                np.sum(~_refl_correct) * np.sum(_direct_correct)
                                            )

                                        if _mixed_reward:
                                            metrics["reflection_grpo/group/mixed_reward_rate"] = float(np.mean(_mixed_reward))
                                        if _all_correct:
                                            metrics["reflection_grpo/group/all_correct_rate"] = float(np.mean(_all_correct))
                                            metrics["reflection_grpo/group/all_wrong_rate"] = float(np.mean(_all_wrong))
                                        if _refl_minus_direct:
                                            metrics["reflection_grpo/group/reflection_acc_minus_direct_acc"] = float(
                                                np.mean(_refl_minus_direct)
                                            )
                                            metrics["reflection_grpo/group/reflection_better_than_direct_rate"] = float(
                                                np.mean(_refl_better)
                                            )
                                            metrics["reflection_grpo/group/direct_all_wrong_reflection_any_correct_rate"] = float(
                                                np.mean(_direct_all_wrong_refl_any_correct)
                                            )
                                            metrics["reflection_grpo/group/direct_any_correct_reflection_all_wrong_rate"] = float(
                                                np.mean(_direct_any_correct_refl_all_wrong)
                                            )
                                        if _pair_total > 0:
                                            metrics["reflection_grpo/pairwise/reflection_correct_direct_wrong_rate"] = (
                                                _pair_refl_correct_direct_wrong / _pair_total
                                            )
                                            metrics["reflection_grpo/pairwise/reflection_wrong_direct_correct_rate"] = (
                                                _pair_refl_wrong_direct_correct / _pair_total
                                            )

                        # compute rewards. apply_invalid_action_penalty if available
                        if self.config.actor_rollout_ref.actor.get('use_invalid_action_penalty', True):
                            batch, invalid_metrics = apply_invalid_action_penalty(batch,
                                                                                  invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef,
                                                                                  )
                            metrics.update(invalid_metrics)

                        # compute rewards. apply_kl_penalty if available
                        use_loss_mask = (
                            self.config.actor_rollout_ref.rollout.multi_turn.enable
                            or "loss_mask" in batch.batch
                        )
                        if "loss_mask" in batch.batch:
                            _resp_len = batch.batch["responses"].size(1)
                            _loss_mask = batch.batch["loss_mask"][:, -_resp_len:].float()
                            metrics["reflection_grpo/loss_mask_active_rate"] = (
                                _loss_mask.mean().detach().item()
                            )
                            metrics["reflection_grpo/loss_mask_masked_rate"] = (
                                1.0 - metrics["reflection_grpo/loss_mask_active_rate"]
                            )
                            _loss_tokens = _loss_mask.sum(dim=-1).detach().cpu().numpy()
                            metrics["reflection_grpo/loss_tokens/zero_loss_seq_rate"] = float(
                                np.mean(_loss_tokens <= 0)
                            )
                            _resp_attention = batch.batch["attention_mask"][:, -_resp_len:].float()
                            _response_tokens = _resp_attention.sum(dim=-1).detach().cpu().numpy()
                            _masked_prefix_tokens = (
                                (_resp_attention - _loss_mask).clamp(min=0).sum(dim=-1).detach().cpu().numpy()
                            )
                            _num_refl_loss = None
                            _num_refl_arr = batch.non_tensor_batch.get("num_reflections", None)
                            if _num_refl_arr is not None:
                                try:
                                    _num_refl_loss = np.asarray(_num_refl_arr, dtype=np.float32)
                                except (TypeError, ValueError):
                                    _num_refl_loss = None
                            if _num_refl_loss is not None and _num_refl_loss.shape[0] == _loss_tokens.shape[0]:
                                _direct_loss_mask = _num_refl_loss == 0
                                _refl_loss_mask = _num_refl_loss > 0
                                _total_loss_tokens = float(np.sum(_loss_tokens))
                                if _direct_loss_mask.any():
                                    metrics["reflection_grpo/loss_tokens/direct_mean"] = float(
                                        np.mean(_loss_tokens[_direct_loss_mask])
                                    )
                                    metrics["reflection_grpo/length/direct_response_mean"] = float(
                                        np.mean(_response_tokens[_direct_loss_mask])
                                    )
                                else:
                                    metrics["reflection_grpo/loss_tokens/direct_mean"] = 0.0
                                    metrics["reflection_grpo/length/direct_response_mean"] = 0.0
                                if _refl_loss_mask.any():
                                    metrics["reflection_grpo/loss_tokens/reflection_mean"] = float(
                                        np.mean(_loss_tokens[_refl_loss_mask])
                                    )
                                    metrics["reflection_grpo/length/reflection_response_mean"] = float(
                                        np.mean(_response_tokens[_refl_loss_mask])
                                    )
                                    metrics["reflection_grpo/length/masked_prefix_tokens_mean"] = float(
                                        np.mean(_masked_prefix_tokens[_refl_loss_mask])
                                    )
                                else:
                                    metrics["reflection_grpo/loss_tokens/reflection_mean"] = 0.0
                                    metrics["reflection_grpo/length/reflection_response_mean"] = 0.0
                                    metrics["reflection_grpo/length/masked_prefix_tokens_mean"] = 0.0
                                if _total_loss_tokens > 0:
                                    metrics["reflection_grpo/loss_tokens/direct_total_share"] = float(
                                        np.sum(_loss_tokens[_direct_loss_mask]) / _total_loss_tokens
                                    )
                                    metrics["reflection_grpo/loss_tokens/reflection_total_share"] = float(
                                        np.sum(_loss_tokens[_refl_loss_mask]) / _total_loss_tokens
                                    )
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch,
                                kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                                multi_turn=use_loss_mask,
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor
                        grpo_skip_std_for_uniform_outcome = self.config.algorithm.get(
                            "grpo_skip_std_for_uniform_outcome", False
                        )

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            grpo_skip_std_for_uniform_outcome=grpo_skip_std_for_uniform_outcome,
                            multi_turn=use_loss_mask,
                            use_pf_ppo=self.config.algorithm.use_pf_ppo,
                            pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                            pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                            step_advantage_w=self.config.algorithm.gigpo.step_advantage_w,
                            gigpo_mode=self.config.algorithm.gigpo.mode,
                            gigpo_enable_similarity= self.config.algorithm.gigpo.enable_similarity,
                            gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,
                        )

                        if (
                            self.config.algorithm.get("reflection_grpo", {}).get("enable", False)
                            and "advantages" in batch.batch
                        ):
                            _adv = batch.batch["advantages"].float()
                            _resp_len = batch.batch["responses"].size(1)
                            if "loss_mask" in batch.batch:
                                _adv_mask = batch.batch["loss_mask"][:, -_resp_len:].float()
                            elif "response_mask" in batch.batch:
                                _adv_mask = batch.batch["response_mask"].float()
                            else:
                                _adv_mask = None

                            _num_refl_adv = batch.non_tensor_batch.get("num_reflections", None)
                            if _adv_mask is not None and _num_refl_adv is not None:
                                try:
                                    _num_refl_adv = np.asarray(_num_refl_adv, dtype=np.float32)
                                except (TypeError, ValueError):
                                    _num_refl_adv = None
                            if (
                                _adv_mask is not None
                                and _num_refl_adv is not None
                                and _num_refl_adv.shape[0] == _adv.shape[0]
                            ):
                                _adv_token_counts = _adv_mask.sum(dim=-1)
                                _valid_seq = (_adv_token_counts > 0).detach().cpu().numpy()
                                _seq_adv = (
                                    (_adv * _adv_mask).sum(dim=-1)
                                    / _adv_token_counts.clamp(min=1.0)
                                ).detach().cpu().numpy()
                                _direct_adv_mask = (_num_refl_adv == 0) & _valid_seq
                                _refl_adv_mask = (_num_refl_adv > 0) & _valid_seq
                                if _direct_adv_mask.any():
                                    _direct_adv = _seq_adv[_direct_adv_mask]
                                    metrics["reflection_grpo/advantage/direct_mean"] = float(np.mean(_direct_adv))
                                    metrics["reflection_grpo/advantage/direct_positive_rate"] = float(
                                        np.mean(_direct_adv > 0)
                                    )
                                else:
                                    metrics["reflection_grpo/advantage/direct_mean"] = 0.0
                                    metrics["reflection_grpo/advantage/direct_positive_rate"] = 0.0
                                if _refl_adv_mask.any():
                                    _refl_adv = _seq_adv[_refl_adv_mask]
                                    metrics["reflection_grpo/advantage/reflection_mean"] = float(np.mean(_refl_adv))
                                    metrics["reflection_grpo/advantage/reflection_positive_rate"] = float(
                                        np.mean(_refl_adv > 0)
                                    )
                                else:
                                    metrics["reflection_grpo/advantage/reflection_mean"] = 0.0
                                    metrics["reflection_grpo/advantage/reflection_positive_rate"] = 0.0

                        # ── zero_adv_ratio: fraction of response tokens with no signal ──
                        if "advantages" in batch.batch and "response_mask" in batch.batch:
                            _adv   = batch.batch["advantages"]
                            if "loss_mask" in batch.batch:
                                _resp_len = batch.batch["responses"].size(1)
                                _rmask = batch.batch["loss_mask"][:, -_resp_len:].bool()
                            else:
                                _rmask = batch.batch["response_mask"].bool()
                            _valid_adv = _adv[_rmask]
                            if _valid_adv.numel() > 0:
                                _zero  = (_valid_adv.abs() < 1e-6).float().mean().item()
                                metrics["rgrpo/signal/zero_adv_ratio"] = _zero

                        # ── hybrid-grpo per-step metrics (trigger_rate / direct_acc / cont_acc / refl_any_correct_rate / ...) ──
                        if "hybrid_metrics" in batch.meta_info:
                            metrics.update(batch.meta_info.pop("hybrid_metrics"))

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = use_loss_mask
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
