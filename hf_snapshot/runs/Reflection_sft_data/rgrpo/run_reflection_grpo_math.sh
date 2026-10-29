#!/usr/bin/env bash
# Reflection-GRPO: complete-trajectory GRPO with batched recursive reflection.
#
# Key differences from the old R-GRPO script:
# 1. Uses standard GRPO advantage on full trajectories only.
# 2. Uses recursive reflection continuation in batched rounds.
# 3. Uses scalar composite reward R = R_task * R_eff.

export CUDA_VISIBLE_DEVICES=0,1

set -xeo pipefail

export PYTHONPATH=/ssd/data/wangruxin/Relf_grpo:${PYTHONPATH:-}
export VLLM_USE_V1=0
export RAY_TMPDIR=/ssd/data/wangruxin/rtg
project_name='Reflection-GRPO-without-right-penalty'
exp_name="${EXP_NAME:-reflection-grpo-qwen2.5-math-1.5b-mathlighteval-flat-wrong-penalty-without-right-penalty}"
resume_mode="${RESUME_MODE:-disable}"

# ============ Algorithm ============
adv_estimator=grpo
max_iters="${MAX_ITERS:-3}"          # total iterations, including the initial attempt

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.02

clip_ratio_low=0.1
clip_ratio_high=0.1

# ============ Data ============
max_prompt_length=1024
max_response_length="${MAX_RESPONSE_LENGTH:-2048}"
loss_agg_mode="seq-mean-token-mean"

train_prompt_bsz=128
n_resp_per_prompt=8
train_prompt_mini_bsz=32
val_batch_size=128
val_before_train="${VAL_BEFORE_TRAIN:-True}"
test_freq="${TEST_FREQ:-13}"

# ============ Cluster ============
NNODES="${NNODES:-1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-2}"

# ============ Paths ============
RAY_DATA_HOME="/ssd/data/wangruxin/R_grpo_sft"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/ssd/data/wangruxin/Qwen2.5-Math-1.5B}"
SFT_ADAPTER_PATH="${SFT_ADAPTER_PATH:-/ssd/data/wangruxin/reasoning_paths_directional/sft_training/outputs/phase2/final}"
MERGED_SFT_MODEL_PATH="${MERGED_SFT_MODEL_PATH:-${RAY_DATA_HOME}/sft_training/outputs/phase2/final_merged_hf}"
MODEL_PATH="${MODEL_PATH:-${MERGED_SFT_MODEL_PATH}}"
CKPTS_DIR="${CKPTS_DIR:-${RAY_DATA_HOME}/RewardFlow/outputs/reflection_grpo/${exp_name}}"
TRAIN_FILE="${RAY_DATA_HOME}/data/math_lighteval_rgrpo.jsonl"
TEST_FILE="${RAY_DATA_HOME}/data/math500_rgrpo_fixed.jsonl"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    if [[ "${MODEL_PATH}" == "${MERGED_SFT_MODEL_PATH}" ]] && [[ -f "${SFT_ADAPTER_PATH}/adapter_config.json" ]]; then
        echo "[INFO] Merging SFT adapter into base model..."
        BASE_MODEL_PATH="${BASE_MODEL_PATH}" SFT_ADAPTER_PATH="${SFT_ADAPTER_PATH}" MERGED_SFT_MODEL_PATH="${MERGED_SFT_MODEL_PATH}" python3 - <<'PY'
import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = os.environ["BASE_MODEL_PATH"]
adapter = os.environ["SFT_ADAPTER_PATH"]
out = os.environ["MERGED_SFT_MODEL_PATH"]

if os.path.isfile(os.path.join(out, "config.json")):
    print(f"[INFO] Merged model already exists at {out}, skipping merge.")
else:
    os.makedirs(out, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter)
    model = model.merge_and_unload()
    model.save_pretrained(out, safe_serialization=True)

    tokenizer_source = adapter if os.path.isfile(os.path.join(adapter, "tokenizer.json")) else base
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    tokenizer.save_pretrained(out)
    print(f"[INFO] Saved merged model to {out}")
PY
    else
        echo "[ERROR] MODEL_PATH=${MODEL_PATH} does not contain a full HuggingFace model."
        echo "[ERROR] Please set MODEL_PATH to a directory with config.json and model weights."
        exit 1
    fi
fi

echo "[INFO] Using model path: ${MODEL_PATH}"
mkdir -p "${CKPTS_DIR}"

temperature="${TEMPERATURE:-0.7}"
top_p="${TOP_P:-0.95}"
top_k=-1

sp_size=1
use_dynamic_bsz=True
gen_tp=1
offload=False

python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=question \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.val_batch_size=${val_batch_size} \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.optim.lr=2.5e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.grad_clip=1.5 \
    actor_rollout_ref.actor.use_invalid_action_penalty=False \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${train_prompt_mini_bsz} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${train_prompt_mini_bsz} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    +algorithm.reflection_grpo.enable=True \
    +algorithm.reflection_grpo.max_iters=${max_iters} \
    custom_reward_function.name=compute_reflection_grpo_score \
    custom_reward_function.path=rgrpo.core_refl_grpo \
    reward_model.reward_manager=naive \
    env.env_name=none \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=${val_before_train} \
    trainer.test_freq=${test_freq} \
    trainer.save_freq=26 \
    trainer.total_epochs=5 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode="${resume_mode}"
