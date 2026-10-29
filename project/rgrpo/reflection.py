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
Reflection Rollout Manager for R-GRPO.

This module handles the generation of rollouts with reflection capabilities,
including multi-round reflection loops and continuation sampling.
"""

import re
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

import numpy as np
import torch

from .core_rgrpo import (
    RoundData,
    DirectRollout,
    ReflRollout,
    TrainingExample,
    find_first_reflection,
    split_response_at_first_reflection,
    build_training_examples,
    MAX_REFLECTIONS,
    K_CONTINUATIONS,
)

logger = logging.getLogger(__name__)


# ── System Prompt ──────────────────────────────────────────────────────────────

REFLECTION_SYSTEM_PROMPT = (
    "You are a math reasoning assistant. "
    "When you make an error during reasoning, "
    "use <reflection>...</reflection> to reflect on the mistake, "
    "then continue reasoning based on the reflection."
)


def build_prompt(tokenizer, problem: str, system_prompt: str = None) -> str:
    """Build the formatted chat prompt for a math problem."""
    if system_prompt is None:
        system_prompt = REFLECTION_SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


@dataclass
class ReflectionRolloutConfig:
    """Configuration for reflection rollout generation."""
    max_reflections: int = MAX_REFLECTIONS
    k_continuations: int = K_CONTINUATIONS
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.95
    system_prompt: str = REFLECTION_SYSTEM_PROMPT


class ReflectionRolloutManager:
    """
    Manager for generating rollouts with reflection capabilities.
    
    This class handles:
    1. Multi-round reflection generation (up to max_reflections rounds)
    2. Continuation sampling after each reflection point
    3. Parsing reflection tags from generated text
    4. Building training examples with proper advantages
    """
    
    def __init__(
        self,
        tokenizer,
        config: ReflectionRolloutConfig = None,
    ):
        """
        Initialize the reflection rollout manager.
        
        Args:
            tokenizer: HuggingFace tokenizer for the model
            config: Configuration for rollout generation
        """
        self.tokenizer = tokenizer
        self.config = config or ReflectionRolloutConfig()
    
    def generate_rollouts_for_batch(
        self,
        rollout_wg,  # Worker group for rollout generation
        problems: List[dict],  # List of {question, answer} dicts
        n_rollouts: int = 8,
    ) -> List[Tuple[str, List[DirectRollout], List[ReflRollout]]]:
        """
        Generate rollouts for a batch of problems using the reflective loop.
        
        Args:
            rollout_wg: Worker group for generation (must support generate_sequences)
            problems: List of problem dicts with 'question' and 'answer' keys
            n_rollouts: Number of rollouts per problem (G in the paper)
        
        Returns:
            List of (header, direct_rollouts, refl_rollouts) tuples per problem
        """
        B = len(problems)
        headers = [build_prompt(self.tokenizer, d["question"], self.config.system_prompt) 
                   for d in problems]
        answers = [d["answer"] for d in problems]
        
        # Per-(problem, rollout) state — indexed by flat index i = prob*G + g
        total = B * n_rollouts
        prompts = [headers[i // n_rollouts] for i in range(total)]
        hdrs = [headers[i // n_rollouts] for i in range(total)]
        gts = [answers[i // n_rollouts] for i in range(total)]
        
        rounds_data: List[List[RoundData]] = [[] for _ in range(total)]
        pre_ids_rounds: List[List[List[int]]] = [[] for _ in range(total)]
        pre_lps_rounds: List[List[List[float]]] = [[] for _ in range(total)]
        refl_ids_rounds: List[List[List[int]]] = [[] for _ in range(total)]
        refl_lps_rounds: List[List[List[float]]] = [[] for _ in range(total)]
        refl_tags_per_rollout: List[List[str]] = [[] for _ in range(total)]
        active = list(range(total))
        
        for _round in range(self.config.max_reflections + 1):
            if not active:
                break
            
            # Generate for all active rollouts
            batch_prompts = [prompts[i] for i in active]
            outputs = self._generate_batch(rollout_wg, batch_prompts)
            
            next_active = []
            for j, idx in enumerate(active):
                if j >= len(outputs):
                    continue
                    
                tids, lps, output_text = outputs[j]
                
                rounds_data[idx].append(RoundData(
                    prompt=prompts[idx],
                    output_text=output_text,
                    token_ids=tids,
                    log_probs=lps,
                ))
                
                match = find_first_reflection(output_text)
                if match and len(refl_tags_per_rollout[idx]) < self.config.max_reflections:
                    pre_ids, pre_lps, refl_ids, refl_lps, refl_tag = \
                        split_response_at_first_reflection(
                            self.tokenizer, prompts[idx], output_text, tids, lps
                        )
                    pre_ids_rounds[idx].append(pre_ids or [])
                    pre_lps_rounds[idx].append(pre_lps or [])
                    refl_ids_rounds[idx].append(refl_ids or [])
                    refl_lps_rounds[idx].append(refl_lps or [])
                    refl_tags_per_rollout[idx].append(refl_tag)
                    # Only keep the latest reflection tag for next round prompt
                    prompts[idx] = hdrs[idx] + refl_tag.rstrip() + "\n"
                    next_active.append(idx)
            
            active = next_active
        
        # Assemble per-problem results
        results = []
        for p in range(B):
            directs: List[DirectRollout] = []
            refls: List[ReflRollout] = []
            
            for g in range(n_rollouts):
                idx = p * n_rollouts + g
                if not rounds_data[idx]:
                    continue
                    
                final_output = rounds_data[idx][-1].output_text
                final_acc = self._compute_accuracy(final_output, answers[p])
                
                if refl_tags_per_rollout[idx]:
                    refls.append(ReflRollout(
                        rounds=rounds_data[idx],
                        pre_refl_token_ids=pre_ids_rounds[idx],
                        pre_refl_log_probs=pre_lps_rounds[idx],
                        refl_token_ids=refl_ids_rounds[idx],
                        refl_log_probs=refl_lps_rounds[idx],
                        refl_tags=refl_tags_per_rollout[idx],
                        final_accuracy=final_acc,
                    ))
                else:
                    rd = rounds_data[idx][0]
                    directs.append(DirectRollout(
                        prompt=rd.prompt,
                        output_text=rd.output_text,
                        token_ids=rd.token_ids,
                        log_probs=rd.log_probs,
                        accuracy=final_acc,
                    ))
            
            results.append((headers[p], directs, refls))
        
        return results
    
    def generate_continuations(
        self,
        rollout_wg,
        prob_data: List[Tuple[str, List[DirectRollout], List[ReflRollout]]],
        gt_answers: List[str],
    ):
        """
        Generate K continuation samples for each reflection point.
        
        This mutates the ReflRollout objects in-place, populating
        cont_rollouts and cont_accuracies.
        
        Args:
            rollout_wg: Worker group for generation
            prob_data: Output from generate_rollouts_for_batch
            gt_answers: Ground truth answers for each problem
        """
        # Collect all continuation prompts
        cont_prompts: List[str] = []
        cont_meta: List[tuple] = []  # (rollout_obj, round_idx, gt, cont_prompt, header)

        for p_idx, (header, _, refl_rollouts) in enumerate(prob_data):
            gt = gt_answers[p_idx]
            for rollout in refl_rollouts:
                n_rounds = len(rollout.refl_tags)
                rollout.cont_rollouts = [[] for _ in range(n_rounds)]
                rollout.cont_accuracies = [[] for _ in range(n_rounds)]

                for r, refl_tag in enumerate(rollout.refl_tags):
                    cont_prompt = header + refl_tag.rstrip() + "\n"
                    for _ in range(self.config.k_continuations):
                        cont_prompts.append(cont_prompt)
                        cont_meta.append((rollout, r, gt, cont_prompt, header))
        
        if not cont_prompts:
            return
        
        # Generate all continuations in parallel
        outputs = self._generate_batch(rollout_wg, cont_prompts)
        
        # First pass: identify which continuations have nested reflections
        # Store intermediate results and collect nested prompts
        intermediate_results: List[Tuple[List[int], List[float], str]] = []
        nested_prompts: List[str] = []
        nested_indices: List[int] = []  # Maps nested prompt index -> original index
        
        for j, (rollout, r, gt, cont_prompt, header) in enumerate(cont_meta):
            if j >= len(outputs):
                intermediate_results.append(([], [], ""))
                continue

            tids, lps, output_text = outputs[j]

            # Check for nested reflection
            match = find_first_reflection(output_text)
            if match:
                nested_tag = match.group(0)
                # Clean-slate: header + nested_tag only (matches training format).
                # Do NOT stack: cont_prompt + nested_tag would accumulate reflection tags.
                nested_prompt = header + nested_tag.rstrip() + "\n"
                nested_prompts.append(nested_prompt)
                nested_indices.append(j)
                # Store placeholder - will be replaced after nested generation
                intermediate_results.append((tids, lps, output_text))
            else:
                intermediate_results.append((tids, lps, output_text))
        
        # Second pass: batch generate all nested continuations in parallel
        if nested_prompts:
            nested_outputs = self._generate_batch(rollout_wg, nested_prompts)
            
            # Replace intermediate results with nested outputs
            for i, orig_idx in enumerate(nested_indices):
                if i < len(nested_outputs):
                    intermediate_results[orig_idx] = nested_outputs[i]
        
        # Final pass: populate rollout data
        for j, (rollout, r, gt, cont_prompt, header) in enumerate(cont_meta):
            tids, lps, output_text = intermediate_results[j]
            
            acc = self._compute_accuracy(output_text, gt)
            rollout.cont_rollouts[r].append(
                RoundData(
                    prompt=cont_prompt,
                    output_text=output_text,
                    token_ids=tids,
                    log_probs=lps,
                )
            )
            rollout.cont_accuracies[r].append(acc)
    
    def build_reflection_info(
        self,
        prob_data: List[Tuple[str, List[DirectRollout], List[ReflRollout]]],
    ) -> Dict[str, Any]:
        """
        Build reflection info dict for advantage computation.
        
        Args:
            prob_data: Output from generate_rollouts_for_batch with continuations
        
        Returns:
            Dict containing:
            - has_reflection: bool array
            - segment_rewards: dict mapping sample keys to per-token advantages
        """
        all_examples = []
        all_stats = []
        
        for p_idx, (header, directs, refls) in enumerate(prob_data):
            examples, stats = build_training_examples(
                self.tokenizer, header, directs, refls
            )
            all_examples.extend(examples)
            all_stats.append(stats)
        
        # Build reflection info structure
        has_reflection = []
        segment_rewards = {}
        
        for i, ex in enumerate(all_examples):
            is_refl = ex.example_type in ("refl_joint", "refl_orig", "continuation")
            has_reflection.append(is_refl)
            
            if is_refl:
                # Store per-token advantages
                segment_rewards[f"batch_{i}"] = ex.token_advantages
        
        return {
            "has_reflection": np.array(has_reflection, dtype=bool),
            "segment_rewards": segment_rewards,
            "examples": all_examples,
            "stats": all_stats,
        }
    
    def _generate_batch(
        self,
        rollout_wg,
        prompts: List[str],
    ) -> List[Tuple[List[int], List[float], str]]:
        """
        Generate responses for a batch of prompts.
        
        Returns list of (token_ids, log_probs, output_text) tuples.
        """
        try:
            from verl import DataProto
            
            # Tokenize prompts
            tokenized = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            
            # Create DataProto for generation
            batch_dict = {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
            }
            
            # Add position_ids if needed
            position_ids = torch.arange(
                tokenized["input_ids"].size(1)
            ).unsqueeze(0).expand(len(prompts), -1)
            batch_dict["position_ids"] = position_ids
            
            data = DataProto.from_single_dict(batch_dict)
            data.non_tensor_batch = {
                "raw_prompt_ids": [
                    self.tokenizer.encode(p, add_special_tokens=False)
                    for p in prompts
                ],
            }
            
            # Set generation parameters
            data.meta_info = {
                "do_sample": True,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "max_new_tokens": self.config.max_new_tokens,
            }
            
            # Generate
            output = rollout_wg.generate_sequences(data)
            
            # Extract results
            results = []
            responses = output.batch.get("responses", None)
            log_probs = output.batch.get("old_log_probs", None)
            
            if responses is None:
                return []
            
            for i in range(len(prompts)):
                resp_ids = responses[i].tolist() if hasattr(responses[i], 'tolist') else list(responses[i])
                lps = log_probs[i].tolist() if log_probs is not None and hasattr(log_probs[i], 'tolist') else []
                
                # Decode response
                output_text = self.tokenizer.decode(resp_ids, skip_special_tokens=True)
                
                results.append((resp_ids, lps, output_text))
            
            return results
            
        except Exception as e:
            logger.warning(f"Generation failed: {e}")
            return [([], [], "") for _ in prompts]
    
    def _compute_accuracy(self, generated: str, ground_truth: str) -> float:
        """Compute accuracy score (0 or 1) for a generated response."""
        from .core_rgrpo import extract_boxed
        
        try:
            pred = extract_boxed(generated)
            if pred is not None:
                # Normalize and compare
                pred_clean = self._normalize_answer(pred)
                gt_clean = self._normalize_answer(ground_truth)
                if pred_clean == gt_clean:
                    return 1.0
        except Exception:
            pass
        return 0.0
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize an answer for comparison."""
        # Remove whitespace and common formatting
        answer = answer.strip().lower()
        answer = re.sub(r'\s+', '', answer)
        # Handle common math formatting
        answer = answer.replace('\\frac', '/frac/')
        answer = answer.replace('\\left', '')
        answer = answer.replace('\\right', '')
        return answer


def integrate_reflection_rollout(
    rollout_wg,
    batch_data,
    tokenizer,
    config: ReflectionRolloutConfig = None,
) -> Dict[str, Any]:
    """
    High-level integration function for reflection rollout.
    
    Call this from the main training loop to generate reflection rollouts
    and build the necessary data structures.
    
    Args:
        rollout_wg: Worker group for generation
        batch_data: Batch of training data with 'question' and 'answer' keys
        tokenizer: HuggingFace tokenizer
        config: Optional configuration
    
    Returns:
        Dict with rollout data and reflection info for advantage computation
    """
    manager = ReflectionRolloutManager(tokenizer, config)
    
    # Extract problems from batch
    problems = []
    if hasattr(batch_data, 'non_tensor_batch'):
        questions = batch_data.non_tensor_batch.get('question', [])
        answers = batch_data.non_tensor_batch.get('answer', [])
        for q, a in zip(questions, answers):
            problems.append({"question": q, "answer": a})
    elif isinstance(batch_data, list):
        problems = batch_data
    else:
        # Try dict-like access
        questions = batch_data.get('question', batch_data.get('questions', []))
        answers = batch_data.get('answer', batch_data.get('answers', []))
        for q, a in zip(questions, answers):
            problems.append({"question": q, "answer": a})
    
    if not problems:
        return {"reflection_info": None, "examples": [], "stats": []}
    
    # Generate rollouts
    n_rollouts = config.k_continuations if config else K_CONTINUATIONS
    prob_data = manager.generate_rollouts_for_batch(
        rollout_wg, problems, n_rollouts=8
    )
    
    # Generate continuations
    gt_answers = [p["answer"] for p in problems]
    manager.generate_continuations(rollout_wg, prob_data, gt_answers)
    
    # Build reflection info
    reflection_info = manager.build_reflection_info(prob_data)
    
    return reflection_info
