#!/usr/bin/env python3
"""One-GPU smoke test for the repository's PAG and SCoRe rollout paths."""

import argparse
import os
import sys

import numpy as np
import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from transformers import AutoConfig, AutoTokenizer

from verl import DataProto
from verl.utils.distributed import initialize_global_process_group
from verl.utils.torch_functional import pad_sequence_to_length
from verl.workers.reward_manager.multiturn import MultiTurnRewardManager
from verl.workers.reward_manager.pag import PAGRewardManager
from verl.workers.rollout.vllm_rollout.vllm_multiturn_rollout_spmd import (
    vLLMMutliTurnRollout,
)
from verl.workers.rollout.vllm_rollout.vllm_pag_rollout_spmd import vLLMPAGRollout


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=("pag", "score"))
    parser.add_argument(
        "--model-path",
        default="/mnt/data/models/Qwen2.5-1.5B-Instruct",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    return parser.parse_args()


def make_input(tokenizer, prompt_length):
    messages = [
        {
            "role": "system",
            "content": "Please reason step by step, and put your final answer within \\boxed{}.",
        },
        {"role": "user", "content": "Calculate 1+1."},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    encoded = tokenizer([text], return_tensors="pt")
    input_ids = pad_sequence_to_length(
        encoded["input_ids"], prompt_length, tokenizer.pad_token_id, left_pad=True
    ).cuda()
    attention_mask = pad_sequence_to_length(
        encoded["attention_mask"], prompt_length, 0, left_pad=True
    ).cuda()
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    batch = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
        batch_size=1,
    )
    return DataProto(
        batch=batch,
        meta_info={
            "eos_token_id": [tokenizer.eos_token_id, 151643],
            "do_sample": False,
            "num_turns": 2,
        },
    )


def main():
    args = parse_args()
    if not os.path.isfile(os.path.join(args.model_path, "model.safetensors")):
        raise FileNotFoundError(f"Model download is incomplete: {args.model_path}")

    _, rank, _ = initialize_global_process_group()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model_config = AutoConfig.from_pretrained(args.model_path, local_files_only=True)

    prompt_length = 256
    rollout_config = DictConfig(
        {
            "tensor_model_parallel_size": 1,
            "prompt_length": prompt_length,
            "response_length": 128,
            "dtype": "bfloat16",
            "enforce_eager": True,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "load_format": "dummy_dtensor",
            "disable_log_stats": True,
            "enable_chunked_prefill": False,
            "free_cache_engine": False,
            "n": 1,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_num_batched_tokens": 2048,
            "max_model_len": 1024,
            "num_turns": 2,
            "val_kwargs": {
                "num_turns": 2,
                "n": 1,
                "do_sample": False,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
            },
        }
    )
    prompts = make_input(tokenizer, prompt_length)

    if args.method == "pag":
        rollout = vLLMPAGRollout(
            model_path=args.model_path,
            config=rollout_config,
            tokenizer=tokenizer,
            model_hf_config=model_config,
            trust_remote_code=True,
        )
        # PAG intentionally sleeps after initialization because the trainer normally
        # wakes it through FSDPVLLMShardingManager. This standalone test must do so.
        rollout.inference_engine.wake_up()
        reward_manager = PAGRewardManager(
            tokenizer,
            num_examine=0,
            config={"num_turns": 2, "policy_rs": True, "rs_coef": 1.0},
        )
    else:
        rollout = vLLMMutliTurnRollout(
            model_path=args.model_path,
            config=rollout_config,
            tokenizer=tokenizer,
            model_hf_config=model_config,
            trust_remote_code=True,
        )
        reward_manager = MultiTurnRewardManager(
            tokenizer,
            num_examine=0,
            config={"num_turns": 2, "policy_rs": True, "rs_coef": 1.0},
        )

    outputs = rollout.generate_sequences(prompts)
    outputs.non_tensor_batch.update(
        {
            "reward_model": np.array([{"ground_truth": "2"}], dtype=object),
            "data_source": np.array(["smoke"], dtype=object),
        }
    )
    reward = reward_manager(outputs, return_dict=True)

    if rank == 0:
        response_attention = outputs.batch["attention_mask"][:, prompt_length:]
        response_ids = outputs.batch["responses"][0][response_attention[0].bool()]
        print("\n=== decoded multi-turn response ===")
        print(tokenizer.decode(response_ids, skip_special_tokens=True))
        print("\n=== reward metrics ===")
        print(reward["metrics"])
        print(f"{args.method} rollout smoke test: PASS")

    # vLLM 0.8.2's external-launcher CuMem allocator can segfault during Python
    # interpreter teardown. This is a one-shot smoke-test subprocess, so bypass
    # object finalizers after flushing; the OS releases its CUDA context.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
