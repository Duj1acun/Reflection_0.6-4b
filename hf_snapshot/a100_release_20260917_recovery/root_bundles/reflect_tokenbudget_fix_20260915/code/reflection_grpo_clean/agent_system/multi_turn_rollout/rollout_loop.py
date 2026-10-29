# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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

import torch
import numpy as np
import os
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import AutoConfig, PreTrainedTokenizer
import uuid
from verl.models.transformers.qwen2_vl import get_rope_index
from agent_system.multi_turn_rollout.utils import process_image, to_list_of_dict, torch_to_numpy, filter_group_data
from agent_system.environments import EnvironmentManagerBase
from typing import List, Dict, Tuple
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
try:
    from rewardflow.core_rewardflow import apply_rewardflow_propagation
except ImportError:
    apply_rewardflow_propagation = None
import time

class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        """
        Initialize the TrajectoryProcessor class.
        
        Parameters:
            config: Configuration object containing data processing settings
            tokenizer (PreTrainedTokenizer): Tokenizer for text encoding and decoding
            processor: Image processor for multimodal inputs
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.model_ctx_limit = self._infer_model_ctx_limit()

    def _infer_model_ctx_limit(self) -> int:
        """Infer a conservative context limit for prompt + response tokens."""
        candidates = []

        def _maybe_add(value):
            try:
                value = int(value)
            except (TypeError, ValueError):
                return
            if 0 < value < 1_000_000:
                candidates.append(value)

        try:
            _maybe_add(self.config.actor_rollout_ref.rollout.get("max_model_len", None))
        except Exception:
            pass

        _maybe_add(getattr(self.tokenizer, "model_max_length", None))

        try:
            model_path = self.config.actor_rollout_ref.model.path
        except Exception:
            model_path = None

        if model_path:
            try:
                model_config = AutoConfig.from_pretrained(
                    model_path,
                    trust_remote_code=False,
                    local_files_only=True,
                )
            except Exception:
                model_config = None

            if model_config is not None:
                for attr in ("max_position_embeddings", "model_max_length", "n_positions", "seq_length", "sliding_window"):
                    _maybe_add(getattr(model_config, attr, None))

        if candidates:
            return min(candidates)
        return int(self.config.data.max_prompt_length + self.config.data.max_response_length)

    def _truncate_hybrid_joint_response(
        self,
        prompt_ids: List[int],
        prefix_ids: List[int],
        cont_ids: List[int],
    ) -> Tuple[List[int], int]:
        """
        Hard-cap the merged hybrid response to fit both the rollout response budget
        and the model context budget. We left-truncate so the newest reflection/continuation
        tokens survive.
        """
        merged_response_ids = list(prefix_ids) + list(cont_ids)
        if not merged_response_ids:
            return [], 0

        response_budget = int(self.config.data.max_response_length)
        response_budget = min(response_budget, max(0, self.model_ctx_limit - len(prompt_ids)))
        if response_budget <= 0:
            return [], 0
        if len(merged_response_ids) <= response_budget:
            return merged_response_ids, len(prefix_ids)

        trim = len(merged_response_ids) - response_budget
        truncated_response_ids = merged_response_ids[-response_budget:]
        truncated_prefix_len = max(0, len(prefix_ids) - trim)
        truncated_prefix_len = min(truncated_prefix_len, len(truncated_response_ids))
        return truncated_response_ids, truncated_prefix_len

    def preprocess_single_sample(
        self,
        item: int,
        gen_batch: DataProto,
        obs: Dict,
    ):
        """
        Process a single observation sample, organizing environment observations (text and/or images) 
        into a format processable by the model.
        
        Parameters:
            item (int): Sample index in the batch
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation, may contain 'text', 'image', 'anchor' keys
        
        Returns:
            dict: Contains processed input data such as input_ids, attention_mask, etc.
        """

        raw_prompt = gen_batch.non_tensor_batch['raw_prompt'][item]
        data_source = gen_batch.non_tensor_batch['data_source'][item]
        
        # Get observation components
        obs_texts = obs.get('text', None)
        obs_images = obs.get('image', None)
        obs_anchors = obs.get('anchor', None)
        obs_text = obs_texts[item] if obs_texts is not None else None
        obs_image = obs_images[item] if obs_images is not None else None
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        is_multi_modal = obs_image is not None

        _obs_anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor

        # Build chat structure
        # obs_content = raw_prompt[0]['content']
        # if '<image>' in obs_content: 
        #     obs_content = obs_content.replace('<image>', '')

        # Build chat structure
        obs_content = ''
        if obs_text is not None:
            obs_content += obs_text
        else:
            print(f"Warning: No text observation found!")

        
        chat = np.array([{
            "content": obs_content,
            "role": "user",
        }])
        
        # Apply chat template
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False
        )
        
        # Initialize return dict
        row_dict = {}
        
        # Process multimodal data
        if is_multi_modal:
            # Replace image placeholder with vision tokens
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(obs_image)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                                self.processor.image_token)

        else:
            raw_prompt = prompt_with_chat_template
        
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                            tokenizer=self.tokenizer,
                                                                            max_length=self.config.data.max_prompt_length,
                                                                            pad_token_id=self.tokenizer.pad_token_id,
                                                                            left_pad=True,
                                                                            truncation=self.config.data.truncation,)
        
        

        if is_multi_modal:

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=image_grid_thw,
                    attention_mask=attention_mask[0],
                )
              ]  # (1, 3, seq_len)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.config.data.max_prompt_length:
            if self.config.data.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.config.data.max_prompt_length :]
            elif self.config.data.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.config.data.max_prompt_length]
            elif self.config.data.truncation == "middle":
                left_half = self.config.data.max_prompt_length // 2
                right_half = self.config.data.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.config.data.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.config.data.max_prompt_length}.")

        # Build final output dict
        row_dict.update({
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt_ids': raw_prompt_ids,
            'anchor_obs': _obs_anchor,
            'index': item,
            'data_source': data_source
        })

        if self.config.data.get('return_raw_chat', False):
            row_dict['raw_prompt'] = chat.tolist()
        
        return row_dict

    def preprocess_batch(
        self,
        gen_batch: DataProto, 
        obs: Dict, 
    ) -> DataProto:
        """
        Process a batch of observation samples, converting environment observations into model-processable format.
        
        Parameters:
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation dictionary
                - 'text' (None or List[str]): Text observation data
                - 'image' (np.ndarray or torch.Tensor): Image observation data
                - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
        
        Returns:
            DataProto: Contains processed batch data with preserved metadata
        """
        batch_size = len(gen_batch.batch['input_ids'])
        processed_samples = []
        
        # Process each sample in parallel
        for item in range(batch_size):
            # Extract per-sample observations
            processed = self.preprocess_single_sample(
                item=item,
                gen_batch=gen_batch,
                obs=obs,
            )
            processed_samples.append(processed)
        
        # Aggregate batch data
        batch = collate_fn(processed_samples)
        
        # Create DataProto with preserved metadata
        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        return new_batch


    def gather_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            tool_callings: np.ndarray,
            ) -> DataProto:
        """
        Collect and organize trajectory data, handling batch size adjustments to meet parallel training requirements.
        
        Parameters:
            total_batch_list (List[List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
            tool_callings (np.ndarray): Number of tool callings for each environment
        Returns:
            DataProto: Collected and organized trajectory data
        """
        batch_size = len(total_batch_list)

        success_rate = {}
        for key, value in success.items():
            success_rate[key] = np.mean(value)
        
        effective_batch = []
        for bs in range(batch_size):
            # sum the rewards for each data in total_batch_list[bs]
            for data in total_batch_list[bs]:
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                if data['active_masks']:
                    # episode_rewards
                    data['episode_rewards'] = episode_rewards[bs]
                    # episode_lengths
                    data['episode_lengths'] = episode_lengths[bs]
                    # tool_callings
                    data['tool_callings'] = tool_callings[bs]
                    # success_rate
                    for key, value in success_rate.items():
                        data[key] = value

                    effective_batch.append(data)
            
        # Convert trajectory data to DataProto format
        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )
        return gen_batch_output

    def vanilla_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Collects trajectories through parallel agent-environment agent_loop.
        Parameters:
            gen_batch (DataProto): Initial batch with prompts to start the agent_loop
            actor_rollout_wg (WorkerGroup): Worker group containing the actor model for policy decisions
            envs (EnvironmentManagerBase): Environment manager containing parallel environment instances
        
        Returns:
            total_batch_list (List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
        """

        batch_size = len(gen_batch.batch)

        # Initial observations from the environment
        obs, infos = envs.reset(kwargs=gen_batch.non_tensor_batch.pop('env_kwargs', None))

        lenght_obs = len(obs['text']) if obs['text'] is not None else len(obs['image'])
        assert len(gen_batch.batch) == lenght_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {lenght_obs}"
        
        if self.config.env.rollout.n > 0: # env grouping
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else: # no env grouping, set all to the same uid
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)
        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)
        # Trajectory collection loop
        for _step in range(self.config.env.max_steps):
            active_masks = np.logical_not(is_done)

            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            batch_input.meta_info = gen_batch.meta_info

            # pad to be divisible by dp_size
            batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
            batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
            # # unpad
            batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)

            batch.non_tensor_batch['uid'] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid

            batch = batch.union(batch_output)
            
            text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)
            
            next_obs, rewards, dones, infos = envs.step(text_actions)

            # Get inventory state for alfworld
            if self.config.env.env_name == "alfworld/AlfredTWEnv":
                inventory_obs, _, _, _ = envs.step(["inventory" for _ in range(batch_size)])
                batch.non_tensor_batch["inventory_state"] = inventory_obs['anchor']

            
            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                # dones is numpy, delete a dimension
                dones = dones.squeeze(1)

            if 'is_action_valid' in infos[0]:
                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)
            else:
                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)

            if 'tool_calling' in infos[0]:
                tool_callings[active_masks] += np.array([info['tool_calling'] for info in infos], dtype=np.float32)[active_masks]
            # Create reward tensor, only assign rewards for active environments
            # episode_rewards += torch_to_numpy(rewards) * torch_to_numpy(active_masks)
            episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
            episode_lengths[active_masks] += 1

            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)
            
            # Update episode lengths for active environments
            batch_list: list[dict] = to_list_of_dict(batch)

            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])

            # Update done states
            is_done = np.logical_or(is_done, dones)
                
            # Update observations for next step
            obs = next_obs

            # Break if all environments are done
            if is_done.all():
                break
        
        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards, 
                    episode_lengths=episode_lengths,
                    )
        
        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings
    
    def dynamic_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Conduct dynamic rollouts until a target batch size is met. 
        Keeps sampling until the desired number of effective trajectories is collected.
        Adopted from DAPO (https://arxiv.org/abs/2503.14476)

        Args:
            gen_batch (DataProto): Initial batch for rollout.
            actor_rollout_wg: Actor model workers for generating responses.
            envs (EnvironmentManagerBase): Environment manager instance.

        Returns:
            total_batch_list (List[Dict]): Complete set of rollout steps.
            total_episode_rewards (np.ndarray): Accumulated rewards.
            total_episode_lengths (np.ndarray): Lengths per episode.
            total_success (Dict[str, np.ndarray]): Success metrics.
            total_traj_uid (np.ndarray): Trajectory IDs.
        """
        total_batch_list = []
        total_episode_rewards = []
        total_episode_lengths = []
        total_success = []
        total_traj_uid = []
        total_tool_callings = []
        try_count: int = 0
        max_try_count = self.config.algorithm.filter_groups.max_num_gen_batches

        while len(total_batch_list) < self.config.data.train_batch_size * self.config.env.rollout.n and try_count < max_try_count:

            if len(total_batch_list) > 0:
                print(f"valid num={len(total_batch_list)} < target num={self.config.data.train_batch_size * self.config.env.rollout.n}. Keep generating... ({try_count}/{max_try_count})")
            try_count += 1

            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = filter_group_data(batch_list=batch_list, 
                                                                                                episode_rewards=episode_rewards, 
                                                                                                episode_lengths=episode_lengths, 
                                                                                                success=success, 
                                                                                                traj_uid=traj_uid, 
                                                                                                tool_callings=tool_callings, 
                                                                                                config=self.config,
                                                                                                last_try=(try_count == max_try_count),
                                                                                                )
            
            total_batch_list += batch_list
            total_episode_rewards.append(episode_rewards)
            total_episode_lengths.append(episode_lengths)
            total_success.append(success)
            total_traj_uid.append(traj_uid)
            total_tool_callings.append(tool_callings)

        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        total_success = {key: np.concatenate([success[key] for success in total_success], axis=0) for key in total_success[0].keys()}
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)
        total_tool_callings = np.concatenate(total_tool_callings, axis=0)

        return total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_tool_callings

    def no_env_loop(
            self,
            gen_batch: DataProto,
            actor_rollout_wg,
            is_train: bool = True,
            ) -> DataProto:
        """
        Single-turn rollout loop for pure math training without an environment.
        Uses actor_rollout_ref.rollout.n for GRPO grouping (vllm handles repetition internally).
        """
        n = self.config.actor_rollout_ref.rollout.n if is_train else 1

        # Only pass fields that generate_sequences actually needs in non_tensor_batch;
        # everything else stays in gen_batch and is merged in afterwards.
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        if "multi_modal_data" in gen_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("multi_modal_data")
        if "tools_kwargs" in gen_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")

        batch_input = gen_batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )
        batch_input.meta_info = gen_batch.meta_info

        # Pad to be divisible by dp_size
        batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
        batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
        batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size * n)

        # vllm repeats inputs by n internally; gen_batch (non-tensor part) needs matching repetition
        if n > 1 and is_train:
            gen_batch = gen_batch.repeat(repeat_times=n, interleave=True)

        batch_size = len(batch_output.batch)

        # Assign group UIDs: same uid for each consecutive n responses (one prompt group)
        uid_batch = []
        for i in range(batch_size):
            if i % n == 0:
                uid = str(uuid.uuid4())
            uid_batch.append(uid)
        uid_batch = np.array(uid_batch, dtype=object)

        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)

        gen_batch.non_tensor_batch['uid'] = uid_batch
        gen_batch.non_tensor_batch['traj_uid'] = traj_uid
        # tool_callings expected by ray_trainer validation; zero for no-env math training
        gen_batch.non_tensor_batch['tool_callings'] = np.zeros(batch_size, dtype=np.float32)

        result = gen_batch.union(batch_output)
        return result, 0.0  # 0.0 for rewardflow_time

    def reflection_grpo_loop(
            self,
            gen_batch: DataProto,
            actor_rollout_wg,
            is_train: bool = True,
            ) -> DataProto:
        """
        Batched recursive reflection rollout for scalar-reward GRPO training.

        Each sampled rollout produces exactly one final trajectory. If a complete
        <reflection>...</reflection> tag is emitted, the trajectory is continued
        from a clean-slate prompt of:

            header + reflection_tag + "\\n"

        Continuations are generated in parallel for every active sample in a
        given round. Direct and reflective trajectories remain in the same GRPO
        group and receive one terminal scalar reward downstream.
        """
        from rgrpo.core_rgrpo import REFL_PATTERN

        from rgrpo.trajectory_context import pack_segment_contexts

        def _get_base_stop_token_ids():
            eos_token_id = self.tokenizer.eos_token_id
            if isinstance(eos_token_id, int):
                return [eos_token_id]
            if isinstance(eos_token_id, (list, tuple)):
                return [tid for tid in eos_token_id if isinstance(tid, int)]
            return []

        def _truncate_cont_prompt(prompt_ids):
            if os.environ.get("FOURWAY_CONFIG"):
                if len(prompt_ids) + self.config.data.max_response_length > self.config.actor_rollout_ref.rollout.max_model_len:
                    raise ValueError("REFLECT+ continuation exceeds context; refusing to truncate problem")
                return list(prompt_ids)
            max_cont_prompt = (
                self.config.data.max_prompt_length
                + self.config.data.max_response_length // 2
            )
            if len(prompt_ids) > max_cont_prompt:
                return prompt_ids[-max_cont_prompt:]
            return list(prompt_ids)

        def _strip_response_ids(resp_row):
            return resp_row[resp_row != self.tokenizer.pad_token_id].tolist()

        def _stage_truncated(stage_ids):
            if not stage_ids:
                return False
            eos_ids = set(_get_base_stop_token_ids())
            if eos_ids and any(token_id in eos_ids for token_id in stage_ids):
                return False
            return len(stage_ids) >= int(self.config.data.max_response_length)

        def _build_prompt_batch(prompt_id_list, meta_info, n):
            if not prompt_id_list:
                return None

            max_len = max(len(ids) for ids in prompt_id_list)
            samples = []
            for ids in prompt_id_list:
                pad_len = max_len - len(ids)
                padded_ids = [self.tokenizer.pad_token_id] * pad_len + list(ids)
                attn = [0] * pad_len + [1] * len(ids)

                ids_t = torch.tensor(padded_ids, dtype=torch.long)
                attn_t = torch.tensor(attn, dtype=torch.long)
                pos_t = compute_position_id_with_mask(attn_t.unsqueeze(0))[0]

                samples.append({
                    "input_ids": ids_t,
                    "attention_mask": attn_t,
                    "position_ids": pos_t,
                    "raw_prompt_ids": np.array(ids, dtype=np.int64),
                })

            prompt_dict = collate_fn(samples)
            batch_meta = dict(meta_info)
            batch_meta["n"] = n

            base_stop_token_ids = _get_base_stop_token_ids()
            if base_stop_token_ids:
                batch_meta["stop_token_ids"] = base_stop_token_ids
            else:
                batch_meta.pop("stop_token_ids", None)

            return DataProto.from_single_dict(prompt_dict, meta_info=batch_meta)

        def _generate_from_prompt_ids(prompt_id_list, n, warn_prefix):
            if not prompt_id_list:
                return []

            prompt_batch = _build_prompt_batch(prompt_id_list, result.meta_info, n)
            prompt_padded, prompt_pad_size = pad_dataproto_to_divisor(
                prompt_batch, actor_rollout_wg.world_size
            )
            generated_padded = actor_rollout_wg.generate_sequences(prompt_padded)
            generated = unpad_dataproto(
                generated_padded, pad_size=prompt_pad_size * max(1, n)
            )

            expected_rows = len(prompt_id_list) * max(1, n)
            total_rows = len(generated)
            usable_rows = min(total_rows, expected_rows)
            usable_rows -= usable_rows % max(1, n)
            if total_rows != expected_rows and os.environ.get("FOURWAY_CONFIG"):
                raise RuntimeError("REFLECT+ rollout lost trajectory rows")
            if total_rows != expected_rows:
                print(
                    f"{warn_prefix} row mismatch: got={total_rows}, "
                    f"expected={expected_rows}, prompts={len(prompt_id_list)}, n={n}"
                )
            if usable_rows <= 0:
                return []

            if usable_rows != expected_rows:
                print(
                    f"{warn_prefix} truncating rows to {usable_rows} "
                    f"(multiple of n={n})"
                )

            resp_ids = generated.batch["responses"][:usable_rows]
            resp_texts = self.tokenizer.batch_decode(resp_ids, skip_special_tokens=True)
            grouped_rows = []
            usable_prompt_count = usable_rows // max(1, n)
            for prompt_idx in range(usable_prompt_count):
                prompt_rows = []
                for sample_idx in range(max(1, n)):
                    row_idx = prompt_idx * max(1, n) + sample_idx
                    prompt_rows.append({
                        "response_ids": _strip_response_ids(resp_ids[row_idx]),
                        "response_text": resp_texts[row_idx],
                    })
                grouped_rows.append(prompt_rows)
            if n == 1:
                return [rows[0] for rows in grouped_rows]
            return grouped_rows

        def _parse_reflection_stage(stage_text, stage_ids):
            match = REFL_PATTERN.search(stage_text)
            if match is None:
                return None

            pre_text = stage_text[:match.start()]
            full_refl_text = stage_text[:match.end()]
            reflection_tag = match.group(0).rstrip() + "\n"
            reflection_ids = self.tokenizer.encode(
                reflection_tag, add_special_tokens=False
            )

            stage_len = len(stage_ids)
            pre_end_tok = len(self.tokenizer.encode(pre_text, add_special_tokens=False))
            refl_end_tok = len(self.tokenizer.encode(full_refl_text, add_special_tokens=False))
            pre_end_tok = min(pre_end_tok, stage_len)
            refl_end_tok = min(max(pre_end_tok, refl_end_tok), stage_len)
            reflection_start_tok = pre_end_tok

            # BPE can merge the tag opener with the preceding text, e.g. " <"
            # or "。<". Locate the token containing the actual opener so the
            # policy can learn the reflection trigger.
            try:
                opener_end_tok = len(
                    self.tokenizer.encode(
                        stage_text[: match.start() + 1], add_special_tokens=False
                    )
                )
                reflection_start_tok = min(
                    pre_end_tok,
                    max(0, min(opener_end_tok - 1, stage_len)),
                )
            except Exception:
                reflection_start_tok = pre_end_tok

            return {
                "reflection_ids": reflection_ids,
                "pre_end_tok": pre_end_tok,
                "reflection_start_tok": reflection_start_tok,
                "refl_end_tok": refl_end_tok,
            }

        def _apply_final_responses(response_id_lists, loss_mask_lists):
            prompt_ids = result.batch["prompts"]
            prompt_len = prompt_ids.shape[1]
            prompt_attention_mask = result.batch["attention_mask"][:, :prompt_len]
            prompt_position_ids = result.batch["position_ids"][:, :prompt_len]
            batch_size_local = len(response_id_lists)
            max_resp_len = max(1, max(len(ids) for ids in response_id_lists))

            final_response_ids = torch.full(
                (batch_size_local, max_resp_len),
                fill_value=self.tokenizer.pad_token_id,
                dtype=prompt_ids.dtype,
                device=prompt_ids.device,
            )
            final_loss_mask = torch.zeros(
                (batch_size_local, max_resp_len),
                dtype=prompt_attention_mask.dtype,
                device=prompt_attention_mask.device,
            )
            for idx, ids in enumerate(response_id_lists):
                if not ids:
                    continue
                final_response_ids[idx, : len(ids)] = torch.tensor(
                    ids, dtype=prompt_ids.dtype, device=prompt_ids.device
                )
                mask = loss_mask_lists[idx] if idx < len(loss_mask_lists) else []
                if len(mask) != len(ids):
                    mask = [1] * len(ids)
                final_loss_mask[idx, : len(ids)] = torch.tensor(
                    mask, dtype=final_loss_mask.dtype, device=final_loss_mask.device
                )

            eos_token_id = result.meta_info.get("eos_token_id", self.tokenizer.eos_token_id)
            response_attention_mask = verl_F.get_response_mask(
                response_id=final_response_ids,
                eos_token=eos_token_id,
                dtype=prompt_attention_mask.dtype,
            )
            delta_position_id = torch.arange(
                1, final_response_ids.size(1) + 1, device=prompt_position_ids.device
            ).unsqueeze(0).repeat(batch_size_local, 1)
            response_position_ids = prompt_position_ids[:, -1:] + delta_position_id

            result.batch["responses"] = final_response_ids
            result.batch["input_ids"] = torch.cat((prompt_ids, final_response_ids), dim=-1)
            result.batch["attention_mask"] = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)
            result.batch["position_ids"] = torch.cat((prompt_position_ids, response_position_ids), dim=-1)
            result.batch["loss_mask"] = final_loss_mask * response_attention_mask

        result, _ = self.no_env_loop(gen_batch, actor_rollout_wg, is_train)
        if "rollout_log_probs" in result.batch.keys():
            result.batch.pop("rollout_log_probs")

        max_iters = int(self.config.algorithm.get("reflection_grpo", {}).get("max_iters", 4))
        max_iters = max(1, max_iters)
        remaining_reflections = max(0, max_iters - 1)
        mask_last_reflection_only = (
            os.environ.get("TRAJ_GRPO_MASK_LAST_REFLECTION_ONLY", "1")
            .strip()
            .lower()
            not in {"0", "false", "no", "n"}
        )
        # When set, the WHOLE generated chain is trained, including the
        # pre-<reflection> reasoning (for iter-1 that is the error chain that
        # triggered the reflection; for iter-2+ it is the previous
        # continuation). Default 0 keeps the paper-original behaviour
        # (pre-reflection segment masked). The user wants the model to learn
        # *when* to pivot, which requires gradient on the tokens that lead up
        # to the reflection, so the run script exports this =1.
        train_full_chain = (
            os.environ.get("TRAJ_GRPO_TRAIN_FULL_CHAIN", "0")
            .strip()
            .lower()
            not in {"0", "false", "no", "n"}
        )
        pre_reflection_mask_val = 1 if train_full_chain else 0

        response_ids = result.batch["responses"]
        batch_size = response_ids.shape[0]
        if batch_size == 0:
            return result, 0.0
        initial_response_texts = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        initial_response_id_lists = [_strip_response_ids(response_ids[i]) for i in range(batch_size)]

        full_input_ids = result.batch["input_ids"]
        full_attn_mask = result.batch["attention_mask"]
        prompt_len_padded = full_input_ids.shape[1] - response_ids.shape[1]
        header_prompt_ids = {}
        for idx in range(batch_size):
            prompt_ids_padded = full_input_ids[idx, :prompt_len_padded]
            prompt_attn = full_attn_mask[idx, :prompt_len_padded]
            header_prompt_ids[idx] = prompt_ids_padded[prompt_attn.bool()].tolist()

        states = {
            idx: {
                "ids": [],
                "loss_mask": [],
                "num_reflections": 0,
                "terminated_by_cap": False,
                "truncated": False,
                "last_reflection_start": -1,
                "segments": [],
                "raw_segments": [],
                "generated_tokens": 0,
            }
            for idx in range(batch_size)
        }
        active = [
            {
                "sample_idx": idx,
                "stage_text": initial_response_texts[idx],
                "stage_ids": initial_response_id_lists[idx],
                "header_ids": header_prompt_ids[idx],
                "sampling_prompt_ids": header_prompt_ids[idx],
                "remaining_reflections": remaining_reflections,
            }
            for idx in range(batch_size)
        ]

        while active:
            prompt_specs = []
            prompt_ids_batch = []

            for node in active:
                sample_idx = node["sample_idx"]
                states[sample_idx]["generated_tokens"] += len(node["stage_ids"])
                states[sample_idx]["raw_segments"].append({"role": "answer", "prompt_token_ids": list(node["sampling_prompt_ids"]), "output_token_ids": list(node["stage_ids"]), "text": node["stage_text"]})
                states[sample_idx]["truncated"] = (
                    states[sample_idx]["truncated"]
                    or _stage_truncated(node["stage_ids"])
                )
                parsed = _parse_reflection_stage(node["stage_text"], node["stage_ids"])

                # Preserve the exact sampling state before constructing the
                # concatenated reward/loss view. Only retained actions count.
                retained_ids = node["stage_ids"]
                if parsed is not None and node["remaining_reflections"] > 0:
                    retained_ids = retained_ids[:parsed["refl_end_tok"]]
                if retained_ids:
                    states[sample_idx]["segments"].append(
                        (list(node["sampling_prompt_ids"]), list(retained_ids))
                    )

                if parsed is None:
                    states[sample_idx]["ids"].extend(node["stage_ids"])
                    states[sample_idx]["loss_mask"].extend([1] * len(node["stage_ids"]))
                    continue

                if node["remaining_reflections"] <= 0:
                    states[sample_idx]["last_reflection_start"] = (
                        len(states[sample_idx]["ids"]) + parsed["reflection_start_tok"]
                    )
                    states[sample_idx]["ids"].extend(node["stage_ids"])
                    states[sample_idx]["loss_mask"].extend(
                        [pre_reflection_mask_val] * parsed["reflection_start_tok"]
                        + [1] * (len(node["stage_ids"]) - parsed["reflection_start_tok"])
                    )
                    states[sample_idx]["terminated_by_cap"] = True
                    continue

                states[sample_idx]["last_reflection_start"] = (
                    len(states[sample_idx]["ids"]) + parsed["reflection_start_tok"]
                )
                states[sample_idx]["ids"].extend(
                    node["stage_ids"][: parsed["refl_end_tok"]]
                )
                states[sample_idx]["loss_mask"].extend(
                    [pre_reflection_mask_val] * parsed["reflection_start_tok"]
                    + [1] * (parsed["refl_end_tok"] - parsed["reflection_start_tok"])
                )
                states[sample_idx]["num_reflections"] += 1

                prompt_specs.append(
                    {
                        "sample_idx": sample_idx,
                        "header_ids": node["header_ids"],
                        "remaining_reflections": node["remaining_reflections"] - 1,
                    }
                )
                prompt_ids_batch.append(
                    _truncate_cont_prompt(node["header_ids"] + parsed["reflection_ids"])
                )

            if not prompt_specs:
                break

            next_rows = _generate_from_prompt_ids(
                prompt_ids_batch, n=1, warn_prefix="[REFL-GRPO]"
            )
            next_active = []
            for prompt_idx, spec in enumerate(prompt_specs):
                if prompt_idx < len(next_rows):
                    row = next_rows[prompt_idx]
                    next_active.append(
                        {
                            "sample_idx": spec["sample_idx"],
                            "stage_text": row["response_text"],
                            "stage_ids": row["response_ids"],
                            "header_ids": spec["header_ids"],
                            "sampling_prompt_ids": prompt_ids_batch[prompt_idx],
                            "remaining_reflections": spec["remaining_reflections"],
                        }
                    )
            active = next_active

        final_response_id_lists = [states[idx]["ids"] for idx in range(batch_size)]
        final_loss_mask_lists = []
        last_segment_start_arr = np.full(batch_size, -1, dtype=np.int32)
        for idx in range(batch_size):
            ids = final_response_id_lists[idx]
            mask = states[idx]["loss_mask"]
            last_start = int(states[idx].get("last_reflection_start", -1))
            last_segment_start_arr[idx] = last_start
            if mask_last_reflection_only and not train_full_chain and last_start >= 0:
                last_start = min(max(0, last_start), len(ids))
                mask = [0] * last_start + [1] * (len(ids) - last_start)
            final_loss_mask_lists.append(mask)
        _apply_final_responses(final_response_id_lists, final_loss_mask_lists)

        segment_tensors = pack_segment_contexts(
            [states[idx]["segments"] for idx in range(batch_size)],
            result.batch["responses"], self.tokenizer.pad_token_id,
        )
        for key, value in segment_tensors.items():
            result.batch[key] = value

        existing_extra_info = result.non_tensor_batch.get("extra_info", None)
        new_extra_info = []
        num_reflections_arr = np.zeros(batch_size, dtype=np.float32)
        n_iters_arr = np.ones(batch_size, dtype=np.float32)
        terminated_arr = np.zeros(batch_size, dtype=bool)

        for idx in range(batch_size):
            base_info = {}
            if existing_extra_info is not None and idx < len(existing_extra_info):
                item = existing_extra_info[idx]
                if isinstance(item, dict):
                    base_info.update(item)

            num_reflections = int(states[idx]["num_reflections"])
            n_iters = 1 + num_reflections
            terminated_by_cap = bool(states[idx]["terminated_by_cap"])
            truncated = bool(states[idx]["truncated"])

            base_info.update({
                "num_reflections": num_reflections,
                "n_iters": int(n_iters),
                "phi": int(max_iters),
                "terminated_by_cap": terminated_by_cap,
                "truncated": truncated,
            })
            new_extra_info.append(base_info)

            num_reflections_arr[idx] = float(num_reflections)
            n_iters_arr[idx] = float(n_iters)
            terminated_arr[idx] = terminated_by_cap

        result.non_tensor_batch["extra_info"] = np.array(new_extra_info, dtype=object)
        result.non_tensor_batch["reflection_grpo_num_reflections"] = num_reflections_arr
        result.non_tensor_batch["reflection_grpo_n_iters"] = n_iters_arr
        result.non_tensor_batch["reflection_grpo_terminated_by_cap"] = terminated_arr
        result.non_tensor_batch["reflection_grpo_last_segment_start"] = last_segment_start_arr

        if os.environ.get("FOURWAY_CONFIG"):
            result.non_tensor_batch["fourway_segments"] = np.array([states[i]["raw_segments"] for i in range(batch_size)], dtype=object)
            result.non_tensor_batch["fourway_generated_tokens"] = np.array([states[i]["generated_tokens"] for i in range(batch_size)])
        return result, 0.0

    def rgrpo_reflection_loop(
            self,
            gen_batch: DataProto,
            actor_rollout_wg,
            is_train: bool = True,
            ) -> DataProto:
        """
        R-GRPO rollout with recursive reflection.

        Training uses:
          - one base sample per reflection event (up to </reflection>)
          - K continuation branches per reflection event
          - within-event GRPO normalization over those K continuation branches

        Validation uses:
          - one deterministic recursive path per sampled response

        The crucial credit-assignment rule is:
          - pre-reflection / reflection tokens are supervised by the mean future
            value induced by the K continuation branches
          - continuation tokens are supervised only by the relative outcome of
            sibling continuation branches generated from the same reflection
        """
        from collections import defaultdict as _dd
        from rgrpo.core_rgrpo import (
            REFL_PATTERN,
            _grpo_advantage,
            _refl_reward,
            _pre_refl_reward,
            _cont_reward,
            compute_math_score,
        )
        from verl.utils.dataset.rl_dataset import collate_fn

        def _get_base_stop_token_ids():
            eos_token_id = self.tokenizer.eos_token_id
            if isinstance(eos_token_id, int):
                return [eos_token_id]
            if isinstance(eos_token_id, (list, tuple)):
                return [tid for tid in eos_token_id if isinstance(tid, int)]
            return []

        def _truncate_cont_prompt(prompt_ids):
            max_cont_prompt = (
                self.config.data.max_prompt_length
                + self.config.data.max_response_length // 2
            )
            if len(prompt_ids) > max_cont_prompt:
                return prompt_ids[-max_cont_prompt:]
            return list(prompt_ids)

        def _strip_response_ids(resp_row):
            return resp_row[resp_row != self.tokenizer.pad_token_id].tolist()

        def _build_prompt_batch(prompt_id_list, meta_info, n):
            if not prompt_id_list:
                return None

            max_len = max(len(ids) for ids in prompt_id_list)
            samples = []
            for ids in prompt_id_list:
                pad_len = max_len - len(ids)
                padded_ids = [self.tokenizer.pad_token_id] * pad_len + list(ids)
                attn = [0] * pad_len + [1] * len(ids)

                ids_t = torch.tensor(padded_ids, dtype=torch.long)
                attn_t = torch.tensor(attn, dtype=torch.long)
                pos_t = compute_position_id_with_mask(attn_t.unsqueeze(0))[0]

                samples.append({
                    "input_ids": ids_t,
                    "attention_mask": attn_t,
                    "position_ids": pos_t,
                    "raw_prompt_ids": np.array(ids, dtype=np.int64),
                })

            prompt_dict = collate_fn(samples)
            batch_meta = dict(meta_info)
            batch_meta["n"] = n

            base_stop_token_ids = _get_base_stop_token_ids()
            if base_stop_token_ids:
                batch_meta["stop_token_ids"] = base_stop_token_ids
            else:
                batch_meta.pop("stop_token_ids", None)

            return DataProto.from_single_dict(prompt_dict, meta_info=batch_meta)

        def _generate_from_prompt_ids(prompt_id_list, n, warn_prefix):
            if not prompt_id_list:
                return []

            prompt_batch = _build_prompt_batch(prompt_id_list, result.meta_info, n)
            prompt_padded, prompt_pad_size = pad_dataproto_to_divisor(
                prompt_batch, actor_rollout_wg.world_size
            )
            generated_padded = actor_rollout_wg.generate_sequences(prompt_padded)
            generated = unpad_dataproto(
                generated_padded, pad_size=prompt_pad_size * max(1, n)
            )

            expected_rows = len(prompt_id_list) * max(1, n)
            total_rows = len(generated)
            usable_rows = min(total_rows, expected_rows)
            usable_rows -= usable_rows % max(1, n)
            if total_rows != expected_rows:
                print(
                    f"{warn_prefix} row mismatch: got={total_rows}, "
                    f"expected={expected_rows}, prompts={len(prompt_id_list)}, n={n}"
                )
            if usable_rows <= 0:
                return []

            if usable_rows != expected_rows:
                print(
                    f"{warn_prefix} truncating rows to {usable_rows} "
                    f"(multiple of n={n})"
                )

            resp_ids = generated.batch["responses"][:usable_rows]
            resp_texts = self.tokenizer.batch_decode(resp_ids, skip_special_tokens=True)
            grouped_rows = []
            usable_prompt_count = usable_rows // max(1, n)
            for prompt_idx in range(usable_prompt_count):
                prompt_rows = []
                for sample_idx in range(max(1, n)):
                    row_idx = prompt_idx * max(1, n) + sample_idx
                    prompt_rows.append({
                        "response_ids": _strip_response_ids(resp_ids[row_idx]),
                        "response_text": resp_texts[row_idx],
                    })
                grouped_rows.append(prompt_rows)
            if n == 1:
                return [rows[0] for rows in grouped_rows]
            return grouped_rows

        def _parse_reflection_stage(stage_text, stage_ids):
            match = REFL_PATTERN.search(stage_text)
            if match is None:
                return None

            pre_text = stage_text[:match.start()]
            full_refl_text = stage_text[:match.end()]
            reflection_tag = match.group(0).rstrip() + "\n"
            reflection_ids = self.tokenizer.encode(
                reflection_tag, add_special_tokens=False
            )

            stage_len = len(stage_ids)
            pre_end_tok = len(self.tokenizer.encode(pre_text, add_special_tokens=False))
            refl_end_tok = len(self.tokenizer.encode(full_refl_text, add_special_tokens=False))
            pre_end_tok = min(pre_end_tok, stage_len)
            refl_end_tok = min(max(pre_end_tok, refl_end_tok), stage_len)

            return {
                "match_start_char": match.start(),
                "match_end_char": match.end(),
                "pre_text": pre_text,
                "full_refl_text": full_refl_text,
                "reflection_tag": reflection_tag,
                "reflection_ids": reflection_ids,
                "pre_end_tok": pre_end_tok,
                "refl_end_tok": refl_end_tok,
            }

        def _generate_from_prompt_ids_chunked(prompt_id_list, n, warn_prefix):
            if not prompt_id_list:
                return []

            chunk_size = max(1, int(max_branch_prompts_per_call))
            if len(prompt_id_list) <= chunk_size:
                return _generate_from_prompt_ids(prompt_id_list, n=n, warn_prefix=warn_prefix)

            grouped = []
            for start in range(0, len(prompt_id_list), chunk_size):
                end = min(start + chunk_size, len(prompt_id_list))
                chunk_groups = _generate_from_prompt_ids(
                    prompt_id_list[start:end],
                    n=n,
                    warn_prefix=f"{warn_prefix} chunk={start // chunk_size + 1}",
                )
                grouped.extend(chunk_groups)
            return grouped

        def _terminal_path(stage_text, stage_ids, offset):
            return {
                "text": stage_text,
                "ids": list(stage_ids),
                "events": [],
                "terminal_segment": (offset, offset + len(stage_ids)),
                "depth": 0,
            }

        def _simulate_future_accuracy_batch(nodes):
            if not nodes:
                return []

            scores = [0.0] * len(nodes)
            active = [
                {
                    "node_idx": node_idx,
                    "stage_text": node["stage_text"],
                    "stage_ids": node["stage_ids"],
                    "header_ids": node["header_ids"],
                    "ground_truth": node["ground_truth"],
                    "remaining_reflections": node["remaining_reflections"],
                }
                for node_idx, node in enumerate(nodes)
            ]

            while active:
                prompt_specs = []
                prompt_ids_batch = []

                for node in active:
                    parsed = _parse_reflection_stage(node["stage_text"], node["stage_ids"])
                    if parsed is None or node["remaining_reflections"] <= 0:
                        scores[node["node_idx"]] = float(
                            compute_math_score(
                                solution_str=node["stage_text"],
                                ground_truth=node["ground_truth"],
                            )
                        )
                        continue

                    prompt_specs.append(
                        {
                            "node_idx": node["node_idx"],
                            "header_ids": node["header_ids"],
                            "ground_truth": node["ground_truth"],
                            "remaining_reflections": node["remaining_reflections"] - 1,
                            "fallback_text": node["stage_text"],
                            "fallback_ids": node["stage_ids"],
                        }
                    )
                    prompt_ids_batch.append(
                        _truncate_cont_prompt(node["header_ids"] + parsed["reflection_ids"])
                    )

                if not prompt_specs:
                    break

                next_rows = _generate_from_prompt_ids_chunked(
                    prompt_ids_batch, n=1, warn_prefix="[RGRPO AUX]"
                )
                next_active = []
                for prompt_idx, spec in enumerate(prompt_specs):
                    if prompt_idx < len(next_rows):
                        row = next_rows[prompt_idx]
                        next_active.append(
                            {
                                "node_idx": spec["node_idx"],
                                "stage_text": row["response_text"],
                                "stage_ids": row["response_ids"],
                                "header_ids": spec["header_ids"],
                                "ground_truth": spec["ground_truth"],
                                "remaining_reflections": spec["remaining_reflections"],
                            }
                        )
                    else:
                        scores[spec["node_idx"]] = float(
                            compute_math_score(
                                solution_str=spec["fallback_text"],
                                ground_truth=spec["ground_truth"],
                            )
                        )
                active = next_active

            return scores

        def _estimate_reflection_future_value_batch(cases):
            if not cases:
                return []

            prompt_ids_batch = [
                _truncate_cont_prompt(case["header_ids"] + case["reflection_ids"])
                for case in cases
            ]
            aux_groups = _generate_from_prompt_ids_chunked(
                prompt_ids_batch, n=K, warn_prefix="[RGRPO AUX]"
            )

            future_values = []
            branch_acc_groups = []
            aux_nodes = []
            aux_owner = []

            for case_idx, case in enumerate(cases):
                if case_idx >= len(aux_groups):
                    branch_acc_groups.append([])
                    future_values.append(0.0)
                    continue

                group = aux_groups[case_idx]
                branch_acc_groups.append([0.0] * len(group))
                for branch_idx, aux_row in enumerate(group):
                    aux_owner.append((case_idx, branch_idx))
                    aux_nodes.append(
                        {
                            "stage_text": aux_row["response_text"],
                            "stage_ids": aux_row["response_ids"],
                            "header_ids": case["header_ids"],
                            "ground_truth": case["ground_truth"],
                            "remaining_reflections": case["remaining_reflections"],
                        }
                    )
                future_values.append(0.0)

            aux_scores = _simulate_future_accuracy_batch(aux_nodes)
            for owner_idx, score in enumerate(aux_scores):
                case_idx, branch_idx = aux_owner[owner_idx]
                branch_acc_groups[case_idx][branch_idx] = score

            for case_idx, branch_scores in enumerate(branch_acc_groups):
                future_values[case_idx] = (
                    float(np.mean(branch_scores)) if branch_scores else 0.0
                )

            return [
                {
                    "future_value": future_values[case_idx],
                    "branch_final_accs": branch_acc_groups[case_idx],
                }
                for case_idx in range(len(cases))
            ]

        def _build_eval_recursive_paths_batch(root_nodes):
            if not root_nodes:
                return {}

            states = {}
            active = []
            for node in root_nodes:
                states[node["sample_idx"]] = {"ids": [], "depth": 0}
                active.append(dict(node))

            while active:
                prompt_specs = []
                prompt_ids_batch = []

                for node in active:
                    sample_idx = node["sample_idx"]
                    parsed = _parse_reflection_stage(node["stage_text"], node["stage_ids"])
                    if parsed is None or node["remaining_reflections"] <= 0:
                        states[sample_idx]["ids"].extend(node["stage_ids"])
                        continue

                    states[sample_idx]["ids"].extend(node["stage_ids"][: parsed["refl_end_tok"]])
                    states[sample_idx]["depth"] = max(
                        states[sample_idx]["depth"], node["depth"]
                    )
                    prompt_specs.append(
                        {
                            "sample_idx": sample_idx,
                            "header_ids": node["header_ids"],
                            "remaining_reflections": node["remaining_reflections"] - 1,
                            "depth": node["depth"] + 1,
                            "fallback_ids": node["stage_ids"][parsed["refl_end_tok"] :],
                        }
                    )
                    prompt_ids_batch.append(
                        _truncate_cont_prompt(node["header_ids"] + parsed["reflection_ids"])
                    )

                if not prompt_specs:
                    break

                next_rows = _generate_from_prompt_ids_chunked(
                    prompt_ids_batch, n=1, warn_prefix="[RGRPO VAL]"
                )
                next_active = []
                for prompt_idx, spec in enumerate(prompt_specs):
                    sample_idx = spec["sample_idx"]
                    if prompt_idx < len(next_rows):
                        row = next_rows[prompt_idx]
                        next_active.append(
                            {
                                "sample_idx": sample_idx,
                                "stage_text": row["response_text"],
                                "stage_ids": row["response_ids"],
                                "header_ids": spec["header_ids"],
                                "remaining_reflections": spec["remaining_reflections"],
                                "depth": spec["depth"],
                            }
                        )
                    else:
                        states[sample_idx]["ids"].extend(spec["fallback_ids"])
                active = next_active

            for sample_idx, state in states.items():
                state["text"] = self.tokenizer.decode(
                    state["ids"], skip_special_tokens=True
                )
            return states

        def _build_online_recursive_paths_batch(root_nodes):
            if not root_nodes:
                return {}

            states = {}
            active = []
            for node in root_nodes:
                states[node["sample_idx"]] = {
                    "ids": [],
                    "events": [],
                    "terminal_segment": (0, len(node["stage_ids"])),
                    "depth": 0,
                }
                active.append(dict(node))

            while active:
                parsed_specs = []
                future_cases = []
                online_prompt_ids = []

                for node in active:
                    sample_idx = node["sample_idx"]
                    state = states[sample_idx]
                    offset = len(state["ids"])
                    parsed = _parse_reflection_stage(node["stage_text"], node["stage_ids"])
                    if parsed is None or node["remaining_reflections"] <= 0:
                        state["ids"].extend(node["stage_ids"])
                        state["terminal_segment"] = (offset, offset + len(node["stage_ids"]))
                        continue

                    state["ids"].extend(node["stage_ids"][: parsed["refl_end_tok"]])
                    n_refl_tokens = max(0, parsed["refl_end_tok"] - parsed["pre_end_tok"])
                    parsed_specs.append(
                        {
                            "sample_idx": sample_idx,
                            "header_ids": node["header_ids"],
                            "ground_truth": node["ground_truth"],
                            "dir_acc": node["dir_acc"],
                            "remaining_reflections": node["remaining_reflections"] - 1,
                            "depth": node["depth"],
                            "offset": offset,
                            "parsed": parsed,
                        }
                    )
                    future_cases.append(
                        {
                            "header_ids": node["header_ids"],
                            "reflection_ids": parsed["reflection_ids"],
                            "ground_truth": node["ground_truth"],
                            "remaining_reflections": node["remaining_reflections"] - 1,
                        }
                    )
                    online_prompt_ids.append(
                        _truncate_cont_prompt(node["header_ids"] + parsed["reflection_ids"])
                    )

                if not parsed_specs:
                    break

                future_infos = _estimate_reflection_future_value_batch(future_cases)
                next_rows = _generate_from_prompt_ids_chunked(
                    online_prompt_ids, n=1, warn_prefix="[RGRPO ONLINE]"
                )

                next_active = []
                for prompt_idx, spec in enumerate(parsed_specs):
                    state = states[spec["sample_idx"]]
                    parsed = spec["parsed"]
                    future_info = (
                        future_infos[prompt_idx]
                        if prompt_idx < len(future_infos)
                        else {"future_value": 0.0, "branch_final_accs": []}
                    )
                    future_value = float(future_info["future_value"])
                    branch_final_accs = future_info["branch_final_accs"]
                    event = {
                        "depth": spec["depth"],
                        "pre_start": spec["offset"],
                        "pre_end": spec["offset"] + parsed["pre_end_tok"],
                        "refl_start": spec["offset"] + parsed["pre_end_tok"],
                        "refl_end": spec["offset"] + parsed["refl_end_tok"],
                        "future_value": future_value,
                        "raw_pre_reward": _pre_refl_reward(future_value, spec["dir_acc"]),
                        "raw_refl_reward": _refl_reward(
                            future_value,
                            spec["dir_acc"],
                            max(0, parsed["refl_end_tok"] - parsed["pre_end_tok"]),
                        ),
                        "delta_acc": future_value - spec["dir_acc"],
                        "branch_var": float(np.var(branch_final_accs))
                        if len(branch_final_accs) > 1
                        else 0.0,
                        "n_pre_tokens": float(parsed["pre_end_tok"]),
                        "n_refl_tokens": float(
                            max(0, parsed["refl_end_tok"] - parsed["pre_end_tok"])
                        ),
                        "dir_acc": spec["dir_acc"],
                    }
                    state["events"].append(event)
                    state["depth"] = max(state["depth"], spec["depth"])

                    if prompt_idx < len(next_rows):
                        row = next_rows[prompt_idx]
                        next_active.append(
                            {
                                "sample_idx": spec["sample_idx"],
                                "stage_text": row["response_text"],
                                "stage_ids": row["response_ids"],
                                "header_ids": spec["header_ids"],
                                "ground_truth": spec["ground_truth"],
                                "dir_acc": spec["dir_acc"],
                                "remaining_reflections": spec["remaining_reflections"],
                                "depth": spec["depth"] + 1,
                            }
                        )
                    else:
                        tail_ids = []
                        tail_start = len(state["ids"])
                        state["terminal_segment"] = (tail_start, tail_start + len(tail_ids))

                active = next_active

            for sample_idx, state in states.items():
                state["text"] = self.tokenizer.decode(
                    state["ids"], skip_special_tokens=True
                )
            return states

        def _rebuild_training_batch(sample_specs):
            if not sample_specs:
                return

            prompt_batch = _build_prompt_batch(
                [spec["prompt_ids"] for spec in sample_specs],
                result.meta_info,
                n=1,
            )
            prompt_ids = prompt_batch.batch["input_ids"].to(result.batch["input_ids"].device)
            prompt_attention_mask = prompt_batch.batch["attention_mask"].to(result.batch["attention_mask"].device)
            prompt_position_ids = prompt_batch.batch["position_ids"].to(result.batch["position_ids"].device)

            batch_size_local = len(sample_specs)
            max_resp_len = max(1, max(len(spec["response_ids"]) for spec in sample_specs))
            final_response_ids = torch.full(
                (batch_size_local, max_resp_len),
                fill_value=self.tokenizer.pad_token_id,
                dtype=prompt_ids.dtype,
                device=prompt_ids.device,
            )
            for idx, spec in enumerate(sample_specs):
                ids = spec["response_ids"]
                if not ids:
                    continue
                final_response_ids[idx, : len(ids)] = torch.tensor(
                    ids, dtype=prompt_ids.dtype, device=prompt_ids.device
                )

            eos_token_id = result.meta_info.get("eos_token_id", self.tokenizer.eos_token_id)
            response_attention_mask = verl_F.get_response_mask(
                response_id=final_response_ids,
                eos_token=eos_token_id,
                dtype=prompt_attention_mask.dtype,
            )
            delta_position_id = torch.arange(
                1, final_response_ids.size(1) + 1, device=prompt_position_ids.device
            ).unsqueeze(0).repeat(batch_size_local, 1)
            response_position_ids = prompt_position_ids[:, -1:] + delta_position_id

            input_ids = torch.cat((prompt_ids, final_response_ids), dim=-1)
            attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)
            position_ids = torch.cat((prompt_position_ids, response_position_ids), dim=-1)

            assert prompt_ids.shape[0] == batch_size_local
            assert final_response_ids.shape[0] == batch_size_local
            assert input_ids.shape[0] == batch_size_local

            rebuilt = DataProto.from_dict(
                tensors={
                    "prompts": prompt_ids,
                    "responses": final_response_ids,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                },
                meta_info=result.meta_info,
            )
            result.batch = rebuilt.batch
            assert result.batch.batch_size[0] == batch_size_local

        def _apply_final_responses(response_id_lists):
            prompt_ids = result.batch["prompts"]
            prompt_len = prompt_ids.shape[1]
            prompt_attention_mask = result.batch["attention_mask"][:, :prompt_len]
            prompt_position_ids = result.batch["position_ids"][:, :prompt_len]
            batch_size_local = len(response_id_lists)
            max_resp_len = max(1, max(len(ids) for ids in response_id_lists))

            final_response_ids = torch.full(
                (batch_size_local, max_resp_len),
                fill_value=self.tokenizer.pad_token_id,
                dtype=prompt_ids.dtype,
                device=prompt_ids.device,
            )
            for idx, ids in enumerate(response_id_lists):
                if not ids:
                    continue
                final_response_ids[idx, : len(ids)] = torch.tensor(
                    ids, dtype=prompt_ids.dtype, device=prompt_ids.device
                )

            eos_token_id = result.meta_info.get("eos_token_id", self.tokenizer.eos_token_id)
            response_attention_mask = verl_F.get_response_mask(
                response_id=final_response_ids,
                eos_token=eos_token_id,
                dtype=prompt_attention_mask.dtype,
            )
            delta_position_id = torch.arange(
                1, final_response_ids.size(1) + 1, device=prompt_position_ids.device
            ).unsqueeze(0).repeat(batch_size_local, 1)
            response_position_ids = prompt_position_ids[:, -1:] + delta_position_id

            result.batch["responses"] = final_response_ids
            result.batch["input_ids"] = torch.cat((prompt_ids, final_response_ids), dim=-1)
            result.batch["attention_mask"] = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)
            result.batch["position_ids"] = torch.cat((prompt_position_ids, response_position_ids), dim=-1)

        # ── 1. Standard initial rollout (n responses per prompt) ───────────────
        result, _ = self.no_env_loop(gen_batch, actor_rollout_wg, is_train)
        max_reflections = int(self.config.algorithm.get("rgrpo", {}).get("max_reflections", 3))
        n = self.config.actor_rollout_ref.rollout.n if is_train else 1
        # 验证时固定 K=1：greedy 生成下多条续推完全相同，避免反思题被重复计数
        K = int(self.config.actor_rollout_ref.rollout.get("k_continuations", n)) if is_train else 1
        # Hybrid-GRPO: reflection tokens and continuation tokens are trained as
        # separate samples so continuation is conditioned only on the reflection.
        use_hybrid_trajectory = (
            str(getattr(self.config.algorithm, "adv_estimator", "")).lower() == "hybrid_grpo"
        )
        hybrid_cfg = self.config.algorithm.get("hybrid_grpo", {})
        use_prefix_contrast = bool(
            hybrid_cfg.get(
                "prefix_contrast",
                os.environ.get("HYBRID_V2_PREFIX_CONTRAST", "0").lower()
                in {"1", "true", "yes", "y"},
            )
        )
        prefix_contrast_k = int(
            hybrid_cfg.get(
                "prefix_contrast_k",
                os.environ.get("HYBRID_V2_PREFIX_CONTRAST_K", K),
            )
        )
        prefix_contrast_k = max(1, prefix_contrast_k)

        response_ids = result.batch["responses"]
        batch_size = response_ids.shape[0]
        if batch_size == 0:
            return result, 0.0
        # Code-level safety rails for recursive reflection rollout. These do not
        # change n/K/max_reflections; they only prevent pathological fan-out from
        # creating one huge CPU/Ray object before generation starts.
        max_branch_prompts_per_call = 64
        max_active_events_per_round = 256
        max_total_reflection_events = 512
        max_total_reflection_branches = 2048

        initial_response_texts = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        initial_response_id_lists = [_strip_response_ids(response_ids[i]) for i in range(batch_size)]

        full_input_ids = result.batch["input_ids"]
        full_attn_mask = result.batch["attention_mask"]
        prompt_len_padded = full_input_ids.shape[1] - response_ids.shape[1]
        header_prompt_ids = {}
        for idx in range(batch_size):
            prompt_ids_padded = full_input_ids[idx, :prompt_len_padded]
            prompt_attn = full_attn_mask[idx, :prompt_len_padded]
            header_prompt_ids[idx] = prompt_ids_padded[prompt_attn.bool()].tolist()

        if not is_train:
            if max_reflections <= 0:
                return result, 0.0

            final_response_id_lists = list(initial_response_id_lists)
            eval_roots = [
                {
                    "sample_idx": idx,
                    "stage_text": initial_response_texts[idx],
                    "stage_ids": initial_response_id_lists[idx],
                    "header_ids": header_prompt_ids[idx],
                    "remaining_reflections": max_reflections,
                    "depth": 1,
                }
                for idx in range(batch_size)
            ]
            eval_paths = _build_eval_recursive_paths_batch(eval_roots)
            for idx, eval_path in eval_paths.items():
                final_response_id_lists[idx] = eval_path["ids"]

            _apply_final_responses(final_response_id_lists)
            return result, 0.0

        # ── 2. Decode responses, detect reflections ────────────────────────────
        initial_has_reflection = np.array(
            [bool(REFL_PATTERN.search(t)) for t in initial_response_texts], dtype=bool
        )

        # ── 3. Collect ground truths ───────────────────────────────────────────
        reward_model_info = result.non_tensor_batch.get("reward_model", None)
        ground_truths = []
        for i in range(batch_size):
            if reward_model_info is not None:
                rm_i = reward_model_info[i]
                gt = rm_i.get("ground_truth", "") if isinstance(rm_i, dict) else str(rm_i)
            else:
                gt = ""
            ground_truths.append(gt)

        # ── 4. Compute direct rewards for all initial responses ────────────────
        direct_rewards = np.array(
            [compute_math_score(solution_str=initial_response_texts[i], ground_truth=ground_truths[i])
             for i in range(batch_size)],
            dtype=np.float32,
        )

        # ── 5. Compute dir_acc per prompt group (uid) ──────────────────────────
        uids = result.non_tensor_batch["uid"]
        uid_to_indices = _dd(list)
        for i, uid in enumerate(uids):
            uid_to_indices[uid].append(i)

        dir_acc_per_uid = {}
        for uid, indices in uid_to_indices.items():
            direct_idxs = [i for i in indices if not initial_has_reflection[i]]
            if direct_idxs:
                dir_acc_per_uid[uid] = float(np.mean(direct_rewards[direct_idxs]))
            else:
                dir_acc_per_uid[uid] = float(np.mean(direct_rewards[indices]))

        # ── 6. Expand reflected samples into event bases + K continuation branches ──
        orig_non_tensor_batch = dict(result.non_tensor_batch)
        direct_reward_by_idx = {idx: float(direct_rewards[idx]) for idx in range(batch_size)}
        event_samples = []
        expanded_samples = []

        def _make_base_sample(prompt_ids, response_ids, source_idx, role, has_refl=True):
            return {
                "prompt_ids": list(prompt_ids),
                "response_ids": list(response_ids),
                "source_idx": int(source_idx),
                "role": role,
                "has_reflection": bool(has_refl),
                "token_adv": [0.0] * len(response_ids) if has_refl else None,
                "value": np.nan,
                "raw_pre_reward": np.nan,
                "raw_refl_reward": np.nan,
                "raw_cont_reward": np.nan,
                "adv_pre": np.nan,
                "adv_refl": np.nan,
                "adv_cont": np.nan,
                "future_value": np.nan,
                "cf_continue_value": np.nan,
                "prefix_contrast_delta": np.nan,
                "delta_acc": np.nan,
                "branch_var": np.nan,
                "cf_branch_var": np.nan,
                "n_pre_tokens": 0.0,
                "n_refl_tokens": 0.0,
                "n_cont_tokens": 0.0,
                "reflection_depth": 0.0,
            }

        def _build_reflection_samples_batched(root_events, use_hybrid=False):
            if not root_events:
                return {"samples": [], "event_samples": [], "stats": {}}

            event_records = {}
            branch_records = {}
            event_order = []
            active_events = []
            next_event_id = 0
            next_branch_id = 0
            truncated_events = 0
            truncated_branches = 0
            round_idx = 0

            for event in root_events:
                if len(active_events) >= max_active_events_per_round:
                    truncated_events += 1
                    continue
                event_copy = dict(event)
                event_copy["event_id"] = next_event_id
                next_event_id += 1
                active_events.append(event_copy)

            while active_events:
                round_idx += 1
                prompt_specs = []
                prompt_ids_batch = []
                cf_prompt_ids_batch = []

                for event in active_events:
                    parsed = _parse_reflection_stage(event["stage_text"], event["stage_ids"])
                    if parsed is None or event["remaining_reflections"] <= 0:
                        continue

                    cont_prompt_ids = _truncate_cont_prompt(
                        event["header_ids"] + parsed["reflection_ids"]
                    )
                    cf_continue_prompt_ids = _truncate_cont_prompt(
                        list(event["prompt_ids"])
                        + list(event["stage_ids"][: parsed["pre_end_tok"]])
                    )
                    _base_resp_ids = list(event["stage_ids"][: parsed["refl_end_tok"]])
                    # Hybrid-GRPO: 该事件的完整前缀 = 祖先前缀 + 本事件的 base_response_ids
                    _hybrid_prefix = list(event.get("response_prefix_ids", [])) + _base_resp_ids
                    event_records[event["event_id"]] = {
                        "event_id": event["event_id"],
                        "prompt_ids": list(event["prompt_ids"]),
                        "header_ids": list(event["header_ids"]),
                        "ground_truth": event["ground_truth"],
                        "dir_acc": float(event["dir_acc"]),
                        "source_idx": int(event["source_idx"]),
                        "depth": int(event["depth"]),
                        "remaining_reflections": int(event["remaining_reflections"]),
                        "pre_end_tok": int(parsed["pre_end_tok"]),
                        "refl_end_tok": int(parsed["refl_end_tok"]),
                        "base_response_ids": _base_resp_ids,
                        "cont_prompt_ids": list(cont_prompt_ids),
                        "cf_continue_prompt_ids": list(cf_continue_prompt_ids),
                        "cf_continue_value": np.nan,
                        "cf_continue_values": [],
                        "child_branch_ids": [],
                        # Hybrid-GRPO: 从根 header 到本事件 refl_tag 末尾的累积响应前缀
                        "hybrid_prefix_ids": _hybrid_prefix,
                    }
                    event_order.append(event["event_id"])
                    prompt_specs.append(
                        {
                            "event_id": event["event_id"],
                            "header_ids": list(event["header_ids"]),
                            "ground_truth": event["ground_truth"],
                            "dir_acc": float(event["dir_acc"]),
                            "source_idx": int(event["source_idx"]),
                            "depth": int(event["depth"]),
                            "remaining_reflections": int(event["remaining_reflections"]),
                            "cont_prompt_ids": list(cont_prompt_ids),
                            # Hybrid-GRPO: 透传给子事件使用
                            "hybrid_prefix_ids": _hybrid_prefix,
                        }
                    )
                    prompt_ids_batch.append(cont_prompt_ids)
                    if use_prefix_contrast:
                        cf_prompt_ids_batch.append(cf_continue_prompt_ids)

                if not prompt_specs:
                    break

                print(
                    "[RGRPO BRANCH] "
                    f"round={round_idx} active={len(active_events)} "
                    f"prompts={len(prompt_specs)} K={K} "
                    f"events={next_event_id} branches={next_branch_id}",
                    flush=True,
                )
                aux_groups = _generate_from_prompt_ids_chunked(
                    prompt_ids_batch, n=K, warn_prefix="[RGRPO BRANCH]"
                )
                cf_groups = []
                if use_prefix_contrast:
                    cf_groups = _generate_from_prompt_ids_chunked(
                        cf_prompt_ids_batch,
                        n=prefix_contrast_k,
                        warn_prefix="[RGRPO PREFIX-CF]",
                    )
                    for prompt_idx, spec in enumerate(prompt_specs):
                        group = cf_groups[prompt_idx] if prompt_idx < len(cf_groups) else []
                        cf_values = [
                            float(
                                compute_math_score(
                                    solution_str=row["response_text"],
                                    ground_truth=spec["ground_truth"],
                                )
                            )
                            for row in group
                        ]
                        event_records[spec["event_id"]]["cf_continue_values"] = cf_values
                        event_records[spec["event_id"]]["cf_continue_value"] = (
                            float(np.mean(cf_values)) if cf_values else 0.0
                        )
                next_active = []

                for prompt_idx, spec in enumerate(prompt_specs):
                    group = aux_groups[prompt_idx] if prompt_idx < len(aux_groups) else []
                    next_remaining = spec["remaining_reflections"] - 1

                    for row in group:
                        if next_branch_id >= max_total_reflection_branches:
                            truncated_branches += 1
                            continue

                        branch_id = next_branch_id
                        next_branch_id += 1

                        branch_stage_text = row["response_text"]
                        branch_stage_ids = row["response_ids"]
                        parsed_branch = _parse_reflection_stage(branch_stage_text, branch_stage_ids)
                        next_event_id_ref = None
                        terminal_value = np.nan

                        can_recurse = parsed_branch is not None and next_remaining > 0
                        if not can_recurse:
                            cont_response_ids = list(branch_stage_ids)
                            terminal_value = float(
                                compute_math_score(
                                    solution_str=branch_stage_text,
                                    ground_truth=spec["ground_truth"],
                                )
                            )
                        else:
                            can_add_event = (
                                next_event_id < max_total_reflection_events
                                and len(next_active) < max_active_events_per_round
                            )
                            if can_add_event:
                                cont_response_ids = list(
                                    branch_stage_ids[: parsed_branch["pre_end_tok"]]
                                )
                                next_event = {
                                    "event_id": next_event_id,
                                    "prompt_ids": list(spec["cont_prompt_ids"]),
                                    "stage_text": branch_stage_text,
                                    "stage_ids": list(branch_stage_ids),
                                    "header_ids": list(spec["header_ids"]),
                                    "ground_truth": spec["ground_truth"],
                                    "dir_acc": float(spec["dir_acc"]),
                                    "remaining_reflections": int(next_remaining),
                                    "depth": int(spec["depth"] + 1),
                                    "source_idx": int(spec["source_idx"]),
                                    # Hybrid-GRPO: 子事件的前缀 = 父事件的 hybrid_prefix_ids
                                    # 注意：spec["hybrid_prefix_ids"] 已含父事件 base_response_ids；
                                    # 此处 cont_response_ids（非叶分支的 pre_refl 部分）在子事件
                                    # 处理时会通过 base_response_ids 计入 hybrid_prefix_ids
                                    "response_prefix_ids": list(spec.get("hybrid_prefix_ids", [])),
                                }
                                next_event_id_ref = next_event_id
                                next_event_id += 1
                                next_active.append(next_event)
                            else:
                                truncated_events += 1
                                cont_response_ids = list(branch_stage_ids)
                                terminal_value = float(
                                    compute_math_score(
                                        solution_str=branch_stage_text,
                                        ground_truth=spec["ground_truth"],
                                    )
                                )

                        branch_records[branch_id] = {
                            "branch_id": branch_id,
                            "parent_event_id": spec["event_id"],
                            "source_idx": int(spec["source_idx"]),
                            "prompt_ids": list(spec["cont_prompt_ids"]),
                            "cont_response_ids": cont_response_ids,
                            "reflection_depth": float(spec["depth"]),
                            "next_event_id": next_event_id_ref,
                            "terminal_value": terminal_value,
                            # Hybrid-GRPO 额外字段
                            "hybrid_n_refl": int(spec["depth"]),      # depth == 反思轮数
                            "hybrid_refl_group_id": int(spec["event_id"]),
                            "hybrid_prefix_ids": list(spec.get("hybrid_prefix_ids", [])),
                            "hybrid_header_ids": list(spec["header_ids"]),
                        }
                        event_records[spec["event_id"]]["child_branch_ids"].append(branch_id)

                print(
                    "[RGRPO BRANCH] "
                    f"round={round_idx} next_active={len(next_active)} "
                    f"total_events={next_event_id} total_branches={next_branch_id} "
                    f"truncated_events={truncated_events} "
                    f"truncated_branches={truncated_branches}",
                    flush=True,
                )
                active_events = next_active

            for event_id in reversed(event_order):
                event = event_records[event_id]
                child_values = []
                cont_rewards = []

                for branch_id in event["child_branch_ids"]:
                    branch = branch_records[branch_id]
                    if branch["next_event_id"] is not None:
                        branch_value = float(
                            event_records[branch["next_event_id"]]["future_value"]
                        )
                    else:
                        branch_value = float(branch["terminal_value"])

                    branch["value"] = branch_value
                    branch["raw_cont_reward"] = float(
                        _cont_reward(branch_value, len(branch["cont_response_ids"]))
                    )
                    child_values.append(branch_value)
                    cont_rewards.append(branch["raw_cont_reward"])

                event["future_value"] = (
                    float(np.mean(child_values)) if child_values else 0.0
                )
                event["branch_var"] = (
                    float(np.var(child_values)) if len(child_values) > 1 else 0.0
                )
                cf_values = event.get("cf_continue_values", [])
                event["cf_branch_var"] = (
                    float(np.var(cf_values)) if len(cf_values) > 1 else 0.0
                )
                if not use_prefix_contrast:
                    event["cf_continue_value"] = np.nan
                    event["prefix_contrast_delta"] = np.nan
                else:
                    event["cf_continue_value"] = float(event.get("cf_continue_value", 0.0))
                    event["prefix_contrast_delta"] = (
                        float(event["future_value"]) - float(event["cf_continue_value"])
                    )

                if len(cont_rewards) > 1:
                    cont_advs = _grpo_advantage(cont_rewards)
                else:
                    cont_advs = cont_rewards

                for branch_id, cont_adv in zip(event["child_branch_ids"], cont_advs):
                    branch_records[branch_id]["adv_cont"] = float(cont_adv)

            event_samples_local = []
            expanded_samples_local = []
            for event_id in event_order:
                event = event_records[event_id]
                base_sample = _make_base_sample(
                    prompt_ids=event["prompt_ids"],
                    response_ids=event["base_response_ids"],
                    source_idx=event["source_idx"],
                    role="event_base",
                    has_refl=True,
                )
                base_sample["future_value"] = float(event["future_value"])
                base_sample["cf_continue_value"] = float(event["cf_continue_value"])
                base_sample["prefix_contrast_delta"] = float(event["prefix_contrast_delta"])
                base_sample["value"] = float(event["future_value"])
                base_sample["raw_pre_reward"] = _pre_refl_reward(
                    event["future_value"], event["dir_acc"]
                )
                base_sample["raw_refl_reward"] = _refl_reward(
                    event["future_value"],
                    event["dir_acc"],
                    max(0, event["refl_end_tok"] - event["pre_end_tok"]),
                )
                base_sample["delta_acc"] = float(event["future_value"] - event["dir_acc"])
                base_sample["branch_var"] = float(event["branch_var"])
                base_sample["cf_branch_var"] = float(event["cf_branch_var"])
                base_sample["n_pre_tokens"] = float(event["pre_end_tok"])
                base_sample["n_refl_tokens"] = float(
                    max(0, event["refl_end_tok"] - event["pre_end_tok"])
                )
                base_sample["reflection_depth"] = float(event["depth"])
                base_sample["pre_end_tok"] = int(event["pre_end_tok"])
                base_sample["refl_end_tok"] = int(event["refl_end_tok"])
                event_samples_local.append(base_sample)
                if use_hybrid:
                    refl_response_ids = event["base_response_ids"][
                        event["pre_end_tok"] : event["refl_end_tok"]
                    ]
                    if refl_response_ids:
                        refl_prompt_ids = (
                            list(event["prompt_ids"])
                            + event["base_response_ids"][: event["pre_end_tok"]]
                        )
                        refl_sample = _make_base_sample(
                            prompt_ids=refl_prompt_ids,
                            response_ids=refl_response_ids,
                            source_idx=event["source_idx"],
                            role="event_reflection",
                            has_refl=True,
                        )
                        refl_sample["future_value"] = float(event["future_value"])
                        refl_sample["cf_continue_value"] = float(event["cf_continue_value"])
                        refl_sample["prefix_contrast_delta"] = float(event["prefix_contrast_delta"])
                        refl_sample["value"] = float(event["future_value"])
                        refl_sample["raw_refl_reward"] = float(base_sample["raw_refl_reward"])
                        refl_sample["delta_acc"] = float(base_sample["delta_acc"])
                        refl_sample["branch_var"] = float(base_sample["branch_var"])
                        refl_sample["cf_branch_var"] = float(base_sample["cf_branch_var"])
                        refl_sample["n_refl_tokens"] = float(len(refl_response_ids))
                        refl_sample["reflection_depth"] = float(event["depth"])
                        refl_sample["hybrid_refl_group_id"] = int(event["event_id"])
                        refl_sample["hybrid_n_refl"] = int(event["depth"])
                        refl_sample["hybrid_sample_role"] = "event_reflection"
                        refl_sample["hybrid_prefix_len"] = 0
                        expanded_samples_local.append(refl_sample)
                else:
                    expanded_samples_local.append(base_sample)

                for branch_id in event["child_branch_ids"]:
                    branch = branch_records[branch_id]
                    if not branch["cont_response_ids"]:
                        continue

                    if use_hybrid:
                        # ── Hybrid-GRPO：续推样本只训练 continuation 本身 ──
                        # 只对叶节点（terminal branch）创建训练样本
                        if branch["next_event_id"] is not None:
                            # 非叶节点，其续推由子事件的 branch 负责训练
                            continue
                        cont_sample = _make_base_sample(
                            prompt_ids=branch["prompt_ids"],
                            response_ids=branch["cont_response_ids"],
                            source_idx=branch["source_idx"],
                            role="continuation",
                            has_refl=True,
                        )
                        cont_sample["token_adv"] = [0.0] * len(branch["cont_response_ids"])
                        cont_sample["hybrid_refl_group_id"] = int(branch["hybrid_refl_group_id"])
                        cont_sample["hybrid_n_refl"]        = int(branch["hybrid_n_refl"])
                        cont_sample["hybrid_sample_role"]   = "continuation"
                        cont_sample["hybrid_prefix_len"]    = 0
                        cont_sample["raw_cont_reward"]      = float(branch["raw_cont_reward"])
                        cont_sample["value"]                = float(branch["value"])
                        cont_sample["future_value"]         = float(event["future_value"])
                        cont_sample["cf_continue_value"]    = float(event["cf_continue_value"])
                        cont_sample["prefix_contrast_delta"]= float(event["prefix_contrast_delta"])
                        cont_sample["cf_branch_var"]        = float(event["cf_branch_var"])
                        cont_sample["reflection_depth"]     = float(branch["reflection_depth"])
                        cont_sample["n_cont_tokens"]        = float(len(branch["cont_response_ids"]))
                    else:
                        # ── 原始 R-GRPO 模式 ─────────────────────────────────────
                        cont_sample = _make_base_sample(
                            prompt_ids=branch["prompt_ids"],
                            response_ids=branch["cont_response_ids"],
                            source_idx=branch["source_idx"],
                            role="continuation",
                            has_refl=True,
                        )
                        cont_sample["token_adv"] = [float(branch["adv_cont"])] * len(
                            cont_sample["response_ids"]
                        )
                        cont_sample["raw_cont_reward"] = float(branch["raw_cont_reward"])
                        cont_sample["adv_cont"]        = float(branch["adv_cont"])
                        cont_sample["value"]           = float(branch["value"])
                        cont_sample["n_cont_tokens"]   = float(len(branch["cont_response_ids"]))
                        cont_sample["reflection_depth"]= float(branch["reflection_depth"])
                        cont_sample["hybrid_refl_group_id"] = -1
                        cont_sample["hybrid_n_refl"]        = 0
                        cont_sample["hybrid_sample_role"]   = "continuation"

                    expanded_samples_local.append(cont_sample)

            return {
                "samples": expanded_samples_local,
                "event_samples": event_samples_local,
                "stats": {
                    "total_events": int(next_event_id),
                    "total_branches": int(next_branch_id),
                    "truncated_events": int(truncated_events),
                    "truncated_branches": int(truncated_branches),
                    "rounds": int(round_idx),
                },
            }

        root_reflected_events = []
        for idx in range(batch_size):
            if not initial_has_reflection[idx]:
                direct_sample = _make_base_sample(
                    prompt_ids=header_prompt_ids[idx],
                    response_ids=initial_response_id_lists[idx],
                    source_idx=idx,
                    role="direct",
                    has_refl=False,
                )
                direct_sample["value"]                 = direct_reward_by_idx[idx]
                direct_sample["hybrid_refl_group_id"] = -1
                direct_sample["hybrid_n_refl"]        = 0
                direct_sample["hybrid_sample_role"]   = "direct"
                direct_sample["hybrid_prefix_len"]    = 0
                expanded_samples.append(direct_sample)
                continue

            root_reflected_events.append(
                {
                    "prompt_ids": header_prompt_ids[idx],
                    "stage_text": initial_response_texts[idx],
                    "stage_ids": initial_response_id_lists[idx],
                    "header_ids": header_prompt_ids[idx],
                    "ground_truth": ground_truths[idx],
                    "dir_acc": dir_acc_per_uid.get(uids[idx], 0.0),
                    "remaining_reflections": max_reflections,
                    "depth": 1,
                    "source_idx": idx,
                    # Hybrid-GRPO: 该事件之前的累积响应前缀（根事件为空）
                    "response_prefix_ids": [],
                }
            )

        batched_reflection_samples = _build_reflection_samples_batched(
            root_reflected_events, use_hybrid=use_hybrid_trajectory
        )
        rollout_stats = batched_reflection_samples.get("stats", {})
        expanded_samples.extend(batched_reflection_samples["samples"])
        event_samples.extend(batched_reflection_samples["event_samples"])
        rollout_total_events = int(rollout_stats.get("total_events", 0))
        rollout_total_branches = int(rollout_stats.get("total_branches", 0))
        rollout_truncated_events = int(rollout_stats.get("truncated_events", 0))
        rollout_truncated_branches = int(rollout_stats.get("truncated_branches", 0))
        rollout_rounds = int(rollout_stats.get("rounds", 0))
        rollout_truncated = float(
            (rollout_truncated_events + rollout_truncated_branches) > 0
        )
        if root_reflected_events or rollout_truncated:
            print(
                "[RGRPO] "
                f"root_reflections={len(root_reflected_events)} "
                f"expanded_samples={len(expanded_samples)} "
                f"event_samples={len(event_samples)} "
                f"events={rollout_total_events} branches={rollout_total_branches} "
                f"rounds={rollout_rounds} truncated_events={rollout_truncated_events} "
                f"truncated_branches={rollout_truncated_branches}",
                flush=True,
            )

        for sample in event_samples:
            adv_pre = float(sample["raw_pre_reward"])
            adv_refl = float(sample["raw_refl_reward"])
            pre_end = min(len(sample["token_adv"]), sample["pre_end_tok"])
            refl_end = min(len(sample["token_adv"]), sample["refl_end_tok"])
            if pre_end > 0:
                sample["token_adv"][:pre_end] = [float(adv_pre)] * pre_end
            if refl_end > pre_end:
                sample["token_adv"][pre_end:refl_end] = [float(adv_refl)] * (refl_end - pre_end)
            sample["adv_pre"] = float(adv_pre)
            sample["adv_refl"] = float(adv_refl)

        for sample in expanded_samples:
            sample.pop("pre_end_tok", None)
            sample.pop("refl_end_tok", None)

        if not expanded_samples:
            return result, 0.0

        source_indices = np.array([sample["source_idx"] for sample in expanded_samples], dtype=np.int64)
        expanded_batch_size = len(expanded_samples)
        has_reflection = np.array([sample["has_reflection"] for sample in expanded_samples], dtype=bool)
        token_adv_obj = np.empty(expanded_batch_size, dtype=object)
        token_adv_obj[:] = [sample["token_adv"] for sample in expanded_samples]

        _m_initial_response = np.array(
            [direct_reward_by_idx[sample["source_idx"]] for sample in expanded_samples],
            dtype=np.float32,
        )
        _m_non_reflect_direct = np.array(
            [
                direct_reward_by_idx[sample["source_idx"]] if sample["role"] == "direct" else np.nan
                for sample in expanded_samples
            ],
            dtype=np.float32,
        )
        _m_online_final = np.array(
            [float(sample["value"]) if not np.isnan(sample["value"]) else np.nan for sample in expanded_samples],
            dtype=np.float32,
        )
        _m_refl_acc = np.array(
            [float(sample["future_value"]) if not np.isnan(sample["future_value"]) else np.nan for sample in expanded_samples],
            dtype=np.float32,
        )
        _m_cf_continue_acc = np.array(
            [
                float(sample["cf_continue_value"])
                if not np.isnan(sample["cf_continue_value"])
                else np.nan
                for sample in expanded_samples
            ],
            dtype=np.float32,
        )
        _m_prefix_contrast_delta = np.array(
            [
                float(sample["prefix_contrast_delta"])
                if not np.isnan(sample["prefix_contrast_delta"])
                else np.nan
                for sample in expanded_samples
            ],
            dtype=np.float32,
        )
        _m_raw_pre_reward = np.array([sample["raw_pre_reward"] for sample in expanded_samples], dtype=np.float32)
        _m_raw_refl_reward = np.array([sample["raw_refl_reward"] for sample in expanded_samples], dtype=np.float32)
        _m_raw_cont_reward = np.array([sample["raw_cont_reward"] for sample in expanded_samples], dtype=np.float32)
        _m_adv_pre = np.array([sample["adv_pre"] for sample in expanded_samples], dtype=np.float32)
        _m_adv_refl = np.array([sample["adv_refl"] for sample in expanded_samples], dtype=np.float32)
        _m_adv_cont = np.array([sample["adv_cont"] for sample in expanded_samples], dtype=np.float32)
        _m_n_pre_tokens = np.array([sample["n_pre_tokens"] for sample in expanded_samples], dtype=np.float32)
        _m_n_refl_tokens = np.array([sample["n_refl_tokens"] for sample in expanded_samples], dtype=np.float32)
        _m_n_cont_tokens = np.array([sample["n_cont_tokens"] for sample in expanded_samples], dtype=np.float32)
        _m_delta_acc = np.array([sample["delta_acc"] for sample in expanded_samples], dtype=np.float32)
        _m_cont_var = np.array([sample["branch_var"] for sample in expanded_samples], dtype=np.float32)
        _m_cf_cont_var = np.array([sample["cf_branch_var"] for sample in expanded_samples], dtype=np.float32)
        _m_reflection_depth = np.array([sample["reflection_depth"] for sample in expanded_samples], dtype=np.float32)

        _m_group_reward_var = np.full(expanded_batch_size, np.nan, dtype=np.float32)
        _scalar_recovery = 0
        _scalar_false_trig = 0
        _scalar_all_correct = 0
        _scalar_all_wrong = 0
        _scalar_zero_signal = 0
        for uid, indices in uid_to_indices.items():
            group_r = direct_rewards[indices]
            gvar = float(np.var(group_r))
            expanded_mask = np.array([uids[sidx] == uid for sidx in source_indices], dtype=bool)
            _m_group_reward_var[expanded_mask] = gvar
            if gvar == 0.0:
                _scalar_zero_signal += 1
            if float(np.mean(group_r)) == 1.0:
                _scalar_all_correct += 1
            if float(np.mean(group_r)) == 0.0:
                _scalar_all_wrong += 1
        for sample in event_samples:
            if sample["future_value"] > dir_acc_per_uid.get(uids[sample["source_idx"]], 0.0):
                _scalar_recovery += 1
            if dir_acc_per_uid.get(uids[sample["source_idx"]], 0.0) > 0.5:
                _scalar_false_trig += 1

        n_groups = len(uid_to_indices)
        n_refl_events = len(event_samples)
        _scalar_all_correct_rate = _scalar_all_correct / n_groups if n_groups > 0 else 0.0
        _scalar_all_wrong_rate = _scalar_all_wrong / n_groups if n_groups > 0 else 0.0
        _scalar_zero_signal_rate = _scalar_zero_signal / n_groups if n_groups > 0 else 0.0
        _scalar_recovery_rate = _scalar_recovery / n_refl_events if n_refl_events > 0 else float("nan")
        _scalar_false_trig_rate = _scalar_false_trig / n_refl_events if n_refl_events > 0 else float("nan")

        _rebuild_training_batch(expanded_samples)

        new_non_tensor_batch = {}
        for key, value in orig_non_tensor_batch.items():
            if key in {
                "uid",
                "traj_uid",
                "tool_callings",
                "has_reflection",
                "rgrpo_token_adv",
                "rgrpo_initial_response_acc",
                "rgrpo_non_reflect_direct_acc",
                "rgrpo_online_final_acc",
                "rgrpo_direct_reward",
                "rgrpo_refl_acc",
                "rgrpo_cf_continue_acc",
                "rgrpo_prefix_contrast_delta",
                "rgrpo_reflection_future_mean",
                "rgrpo_raw_pre_reward",
                "rgrpo_raw_refl_reward",
                "rgrpo_raw_cont_reward",
                "rgrpo_adv_pre",
                "rgrpo_adv_refl",
                "rgrpo_adv_cont",
                "rgrpo_n_pre_tokens",
                "rgrpo_n_refl_tokens",
                "rgrpo_n_cont_tokens",
                "rgrpo_reflection_depth",
                "rgrpo_delta_acc",
                "rgrpo_cont_var",
                "rgrpo_cf_cont_var",
                "rgrpo_group_reward_var",
                "rgrpo_recovery_rate",
                "rgrpo_false_trig_rate",
                "rgrpo_all_correct_rate",
                "rgrpo_all_wrong_rate",
                "rgrpo_zero_signal_rate",
                "rgrpo_rollout_total_events",
                "rgrpo_rollout_total_branches",
                "rgrpo_rollout_truncated_events",
                "rgrpo_rollout_truncated_branches",
                "rgrpo_rollout_truncated",
                "rgrpo_rollout_rounds",
                # Hybrid-GRPO 专用字段
                "hybrid_refl_group_id",
                "hybrid_n_refl",
                "hybrid_sample_role",
                "hybrid_prefix_len",
                "hybrid_cf_continue_acc",
                "hybrid_prefix_contrast_delta",
            }:
                continue
            if key == "raw_prompt_ids":
                new_non_tensor_batch[key] = np.array(
                    [np.array(sample["prompt_ids"], dtype=np.int64) for sample in expanded_samples],
                    dtype=object,
                )
                continue
            if key == "raw_prompt":
                new_non_tensor_batch[key] = np.array(
                    [self.tokenizer.decode(sample["prompt_ids"], skip_special_tokens=True) for sample in expanded_samples],
                    dtype=object,
                )
                continue
            if isinstance(value, np.ndarray) and len(value) == batch_size:
                new_non_tensor_batch[key] = value[source_indices]
                continue
            if isinstance(value, list) and len(value) == batch_size:
                new_non_tensor_batch[key] = np.array([value[i] for i in source_indices], dtype=object)
                continue
            new_non_tensor_batch[key] = value

        new_non_tensor_batch["uid"] = np.array([uids[i] for i in source_indices], dtype=object)
        new_non_tensor_batch["traj_uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(expanded_batch_size)], dtype=object
        )
        new_non_tensor_batch["tool_callings"] = np.zeros(expanded_batch_size, dtype=np.float32)
        new_non_tensor_batch["has_reflection"] = has_reflection
        new_non_tensor_batch["rgrpo_token_adv"] = token_adv_obj

        # ── Hybrid-GRPO 专用字段 ──────────────────────────────────────────────
        _hybrid_rg_id = np.empty(expanded_batch_size, dtype=object)
        _hybrid_rg_id[:] = [sample.get("hybrid_refl_group_id", -1) for sample in expanded_samples]
        _hybrid_n_refl = np.array(
            [sample.get("hybrid_n_refl", 0) for sample in expanded_samples], dtype=np.float32
        )
        _hybrid_roles = np.empty(expanded_batch_size, dtype=object)
        _hybrid_roles[:] = [sample.get("hybrid_sample_role", "direct") for sample in expanded_samples]
        _hybrid_prefix_len = np.array(
            [sample.get("hybrid_prefix_len", 0) for sample in expanded_samples], dtype=np.int64
        )
        new_non_tensor_batch["hybrid_refl_group_id"] = _hybrid_rg_id
        new_non_tensor_batch["hybrid_n_refl"]        = _hybrid_n_refl
        new_non_tensor_batch["hybrid_sample_role"]   = _hybrid_roles
        new_non_tensor_batch["hybrid_prefix_len"]    = _hybrid_prefix_len
        new_non_tensor_batch["hybrid_cf_continue_acc"] = _m_cf_continue_acc
        new_non_tensor_batch["hybrid_prefix_contrast_delta"] = _m_prefix_contrast_delta
        new_non_tensor_batch.update({
            "rgrpo_initial_response_acc": _m_initial_response,
            "rgrpo_non_reflect_direct_acc": _m_non_reflect_direct,
            "rgrpo_online_final_acc": _m_online_final,
            "rgrpo_direct_reward": _m_initial_response,
            "rgrpo_refl_acc": _m_refl_acc,
            "rgrpo_cf_continue_acc": _m_cf_continue_acc,
            "rgrpo_prefix_contrast_delta": _m_prefix_contrast_delta,
            "rgrpo_reflection_future_mean": _m_refl_acc,
            "rgrpo_raw_pre_reward": _m_raw_pre_reward,
            "rgrpo_raw_refl_reward": _m_raw_refl_reward,
            "rgrpo_raw_cont_reward": _m_raw_cont_reward,
            "rgrpo_adv_pre": _m_adv_pre,
            "rgrpo_adv_refl": _m_adv_refl,
            "rgrpo_adv_cont": _m_adv_cont,
            "rgrpo_n_pre_tokens": _m_n_pre_tokens,
            "rgrpo_n_refl_tokens": _m_n_refl_tokens,
            "rgrpo_n_cont_tokens": _m_n_cont_tokens,
            "rgrpo_reflection_depth": _m_reflection_depth,
            "rgrpo_delta_acc": _m_delta_acc,
            "rgrpo_cont_var": _m_cont_var,
            "rgrpo_cf_cont_var": _m_cf_cont_var,
            "rgrpo_group_reward_var": _m_group_reward_var,
            "rgrpo_recovery_rate": np.full(expanded_batch_size, _scalar_recovery_rate, dtype=np.float32),
            "rgrpo_false_trig_rate": np.full(expanded_batch_size, _scalar_false_trig_rate, dtype=np.float32),
            "rgrpo_all_correct_rate": np.full(expanded_batch_size, _scalar_all_correct_rate, dtype=np.float32),
            "rgrpo_all_wrong_rate": np.full(expanded_batch_size, _scalar_all_wrong_rate, dtype=np.float32),
            "rgrpo_zero_signal_rate": np.full(expanded_batch_size, _scalar_zero_signal_rate, dtype=np.float32),
            "rgrpo_rollout_total_events": np.full(expanded_batch_size, rollout_total_events, dtype=np.float32),
            "rgrpo_rollout_total_branches": np.full(expanded_batch_size, rollout_total_branches, dtype=np.float32),
            "rgrpo_rollout_truncated_events": np.full(expanded_batch_size, rollout_truncated_events, dtype=np.float32),
            "rgrpo_rollout_truncated_branches": np.full(expanded_batch_size, rollout_truncated_branches, dtype=np.float32),
            "rgrpo_rollout_truncated": np.full(expanded_batch_size, rollout_truncated, dtype=np.float32),
            "rgrpo_rollout_rounds": np.full(expanded_batch_size, rollout_rounds, dtype=np.float32),
        })
        result.non_tensor_batch = new_non_tensor_batch

        return result, 0.0

    def multi_turn_loop(
            self,
            gen_batch: DataProto,
            actor_rollout_wg,
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            ) -> DataProto:
        """
        Select and run the appropriate rollout loop (dynamic or vanilla).

        Args:
            gen_batch (DataProto): Initial prompt batch.
            actor_rollout_wg: Actor model workers.
            envs (EnvironmentManagerBase): Environment manager for interaction.
            is_train (bool): Whether in training mode (affects dynamic sampling).

        Returns:
            DataProto: Final collected trajectory data with metadata.
        """
        # No-environment path: pure math training (GRPO / R-GRPO without env)
        if envs is None:
            if self.config.algorithm.get("reflection_grpo", {}).get("enable", False):
                if is_train and self.config.algorithm.reflection_grpo.get("branching", {}).get("enable", False):
                    from rgrpo.branch_rollout import branch_reflection_rollout
                    return branch_reflection_rollout(self, gen_batch, actor_rollout_wg)
                return self.reflection_grpo_loop(
                    gen_batch=gen_batch, actor_rollout_wg=actor_rollout_wg, is_train=is_train
                )
            if self.config.algorithm.adv_estimator in ('r_grpo', 'hybrid_grpo'):
                return self.rgrpo_reflection_loop(
                    gen_batch=gen_batch, actor_rollout_wg=actor_rollout_wg, is_train=is_train
                )
            return self.no_env_loop(gen_batch=gen_batch, actor_rollout_wg=actor_rollout_wg, is_train=is_train)

        if is_train:
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)

        # Initial observations from the environment
        if self.config.algorithm.filter_groups.enable and is_train:
            # Dynamic Sampling (for DAPO and Dynamic GiGPO)
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                self.dynamic_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        else:
            # Vanilla Sampling   
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
            
        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)
        assert len(total_batch_list) == len(totoal_tool_callings)


        rewardflow_start_time = time.time()
        if self.config.algorithm.adv_estimator == "rewardflow" and is_train:
            if apply_rewardflow_propagation is None:
                raise ImportError("rewardflow module not found but rewardflow advantage estimator is selected")
            total_batch_list = apply_rewardflow_propagation(total_batch_list, self.config, self.tokenizer, envs)
        rewardflow_end_time = time.time()
        print(f"rewardflow time: {rewardflow_end_time - rewardflow_start_time} seconds")
        rewardflow_time = rewardflow_end_time - rewardflow_start_time

        # Create trajectory data
        gen_batch_output: DataProto = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
            tool_callings=totoal_tool_callings,
        )
        
        return gen_batch_output, rewardflow_time
