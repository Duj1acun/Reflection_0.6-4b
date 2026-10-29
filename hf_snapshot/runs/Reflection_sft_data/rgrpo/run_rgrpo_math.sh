#!/usr/bin/env bash
# R-GRPO Smoke Test: Quick verification on GPU 7
# This is a minimal smoke test to verify the R-GRPO pipeline works correctly


# GPUs for training (default to GPU 0,1; override via env if needed)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

set -xeo pipefail

# 设置PYTHONPATH以包含verl模块
export PYTHONPATH=/ssd/data/wangruxin/Relf_grpo:${PYTHONPATH:-}

# ============ vLLM 引擎配置 ============
export VLLM_USE_V1=0

# ============ Ray 配置 ============
export RAY_TMPDIR=/ssd/data/wangruxin/ray_tmp_rgrpo

# ============ 项目配置 ============
project_name='GRPO-Baseline-v4'
exp_name="${EXP_NAME:-rgrpo-qwen2.5-math-1.5b-v4}"

# ============ 算法配置 ============
adv_estimator=r_grpo          # R-GRPO advantage estimator

# KL 配置 (对齐 GRPO baseline)
use_kl_in_reward=False
kl_coef=0.014
use_kl_loss=True
kl_loss_coef=0.014

# PPO clip 配置
clip_ratio_low=0.2
clip_ratio_high=0.2

# ============ 数据配置 ============
max_prompt_length=1024
max_response_length=2048

# R-GRPO CRITICAL: seq-mean-token-mean 损失聚合
loss_agg_mode="seq-mean-token-mean"

# 批量配置 (对齐 GRPO baseline: 相同 bsz 和 G=8 rollout 组)
train_prompt_bsz=128
n_resp_per_prompt=8             # G=8, 与 GRPO 相同 (保证比较公平)
k_continuations=4               # K=4 continuation 采样 (R-GRPO 独有超参, 减半加速)
train_prompt_mini_bsz=32
val_batch_size=128
val_before_train="${VAL_BEFORE_TRAIN:-True}"
test_freq="${TEST_FREQ:-13}"

# ============ 集群配置 ============
NNODES="${NNODES:-1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-2}"

# ============ 路径配置 ============
RAY_DATA_HOME="${RAY_DATA_HOME:-/mnt/data/runs/Reflection_sft_data}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/mnt/data/models/Qwen3-4B}"
SFT_ADAPTER_PATH="${SFT_ADAPTER_PATH:-${RAY_DATA_HOME}/sft_training/outputs/qwen3_4b/phase2_replay1_lr1e5/final}"
MERGED_SFT_MODEL_PATH="${MERGED_SFT_MODEL_PATH:-${RAY_DATA_HOME}/sft_training/outputs/qwen3_4b/phase2_replay1_lr1e5/final_merged_hf}"
MODEL_PATH="${MODEL_PATH:-${MERGED_SFT_MODEL_PATH}}"
CKPTS_DIR="${CKPTS_DIR:-${RAY_DATA_HOME}/RewardFlow/outputs/rgrpo_smoke_test/${exp_name}}"
TRAIN_FILE="${TRAIN_FILE:-${RAY_DATA_HOME}/rl_data/limo_train.parquet}"
TEST_FILE="${TEST_FILE:-${RAY_DATA_HOME}/rl_data/math500_val.parquet}"

# If MODEL_PATH is the merged SFT target but not created yet, merge LoRA adapter
# (SFT output) into the base model once before R-GRPO training.
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

# Create checkpoint directory
mkdir -p "${CKPTS_DIR}"

# ============ 生成配置 ============
temperature=0.7
top_p=0.95
top_k=-1

# ============ 性能配置 ============
sp_size=1                     # 序列并行度
use_dynamic_bsz=True
gen_tp=1                      # vLLM 张量并行度
offload=False                 # 全参数训练建议关闭 offload

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
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.optim.lr=1.4e-6 \
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
    +actor_rollout_ref.rollout.k_continuations=${k_continuations} \
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
    custom_reward_function.name=compute_math_score \
    custom_reward_function.path=rgrpo.core_rgrpo \
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
    trainer.total_epochs=10 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto
