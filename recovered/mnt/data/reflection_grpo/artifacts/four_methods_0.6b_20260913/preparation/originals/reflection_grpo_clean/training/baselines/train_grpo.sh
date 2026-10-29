#!/usr/bin/env bash
# Standard GRPO baseline using the same data and math reward backend as REFLECT+.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -a
    source "${PROJECT_ROOT}/.env"
    set +a
fi

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

set -xeo pipefail

MODEL_ROOT="${MODEL_ROOT:-${PROJECT_ROOT}/models}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/artifacts}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export VLLM_USE_V1=0
export RAY_BASE_TMPDIR="${RAY_BASE_TMPDIR:-/tmp/reflect_plus_grpo}"
export RAY_TMPDIR="${RAY_TMPDIR:-${RAY_BASE_TMPDIR}_$$}"
if (( ${#RAY_TMPDIR} > 38 )); then
    echo "[WARN] RAY_TMPDIR is too long for Ray sockets; using /tmp/rpg_$$."
    export RAY_TMPDIR="/tmp/rpg_$$"
fi
export RAY_INIT_ADDRESS="${RAY_INIT_ADDRESS:-local}"
export RAY_INCLUDE_DASHBOARD="${RAY_INCLUDE_DASHBOARD:-0}"
if [[ "${RAY_INIT_ADDRESS}" == "local" ]]; then
    unset RAY_ADDRESS
fi
_ray_visible_devices="${CUDA_VISIBLE_DEVICES// /}"
if [[ -n "${_ray_visible_devices}" ]]; then
    IFS=',' read -r -a _ray_cuda_devices <<< "${_ray_visible_devices}"
    export RAY_INIT_NUM_GPUS="${RAY_INIT_NUM_GPUS:-${#_ray_cuda_devices[@]}}"
fi
mkdir -p "${RAY_TMPDIR}"

if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
    unset PYTORCH_CUDA_ALLOC_CONF
fi

export HYBRID_V2_MATH_SCORE_BACKEND="${HYBRID_V2_MATH_SCORE_BACKEND:-process}"
export HYBRID_V2_MATH_SCORE_TIMEOUT="${HYBRID_V2_MATH_SCORE_TIMEOUT:-30}"

project_name="${PROJECT_NAME:-REFLECT-Plus}"
exp_name="${EXP_NAME:-grpo-baseline-qwen2.5-math-1.5b}"
resume_mode="${RESUME_MODE:-auto}"

adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef="${KL_LOSS_COEF:-0.0}"

clip_ratio_low="${CLIP_RATIO_LOW:-0.20}"
clip_ratio_high="${CLIP_RATIO_HIGH:-0.26}"

max_prompt_length="${MAX_PROMPT_LENGTH:-1024}"
max_response_length="${MAX_RESPONSE_LENGTH:-3072}"
max_model_len="${MAX_MODEL_LEN:-4096}"

loss_agg_mode="${LOSS_AGG_MODE:-seq-mean-token-mean}"

train_prompt_bsz="${TRAIN_PROMPT_BSZ:-64}"   # aligned with decoupled single-GPU run
n_resp_per_prompt="${N_RESP_PER_PROMPT:-8}"
train_prompt_mini_bsz="${PPO_MINI_BATCH_SIZE:-64}"
ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}"
val_batch_size="${VAL_BATCH_SIZE:-512}"
val_before_train="${VAL_BEFORE_TRAIN:-True}"
test_freq="${TEST_FREQ:-10}"
save_freq="${SAVE_FREQ:-50}"
total_training_steps="${TOTAL_TRAINING_STEPS:-1000}"
total_epochs_cap="${TOTAL_EPOCHS_CAP:-100000}"

NNODES="${NNODES:-1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-1}"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-${MODEL_ROOT}/Qwen2.5-Math-1.5B}"
SFT_ADAPTER_PATH="${SFT_ADAPTER_PATH:-${OUTPUT_ROOT}/sft/phase2/final}"
MERGED_SFT_MODEL_PATH="${MERGED_SFT_MODEL_PATH:-${OUTPUT_ROOT}/sft/phase2_merged_hf}"
MODEL_PATH="${MODEL_PATH:-${BASE_MODEL_PATH}}"
CKPTS_DIR="${CKPTS_DIR:-${OUTPUT_ROOT}/checkpoints/${exp_name}}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_ROOT}/rl/limo_grpo.jsonl}"
TEST_FILE="${TEST_FILE:-${DATA_ROOT}/rl/math500_grpo.jsonl}"

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

LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${exp_name}_$(date +%Y%m%d_%H%M%S).log"
echo "[INFO] Training log -> ${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

temperature="${TEMPERATURE:-1.0}"
top_p="${TOP_P:-1.0}"
top_k="${TOP_K:--1}"

sp_size=1
use_dynamic_bsz=True
gen_tp="${GEN_TP:-1}"
offload="${OFFLOAD:-True}"   # single-GPU: offload actor params+optim to CPU (aligned with decoupled)
actor_lr="${ACTOR_LR:-1e-6}"
actor_weight_decay="${ACTOR_WEIGHT_DECAY:-0.0}"
actor_grad_clip="${ACTOR_GRAD_CLIP:-1.0}"
actor_ppo_max_token_len_per_gpu="${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-32768}"
log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-32}"
log_prob_max_token_len_per_gpu="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-65536}"
rollout_gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.45}"   # single-GPU colocated (aligned with decoupled)
rollout_max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-131072}"
rollout_max_num_seqs="${ROLLOUT_MAX_NUM_SEQS:-128}"
rollout_swap_space="${ROLLOUT_SWAP_SPACE:-4}"

python3 -m verl.trainer.main_ppo \
    ray_init.num_cpus=${RAY_NUM_CPUS:-32} \
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
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len_per_gpu} \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.optim.weight_decay=${actor_weight_decay} \
    actor_rollout_ref.actor.grad_clip=${actor_grad_clip} \
    actor_rollout_ref.actor.use_invalid_action_penalty=False \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${log_prob_max_token_len_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${log_prob_max_token_len_per_gpu} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_memory_utilization} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${rollout_max_num_batched_tokens} \
    actor_rollout_ref.rollout.max_num_seqs=${rollout_max_num_seqs} \
    +actor_rollout_ref.rollout.swap_space=${rollout_swap_space} \
    actor_rollout_ref.rollout.max_model_len=${max_model_len} \
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
    custom_reward_function.path=rgrpo.core_hybrid_grpo_v2 \
    reward_model.reward_manager=naive \
    env.env_name=none \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=${val_before_train} \
    trainer.test_freq=${test_freq} \
    trainer.save_freq=${save_freq} \
    trainer.total_training_steps=${total_training_steps} \
    trainer.total_epochs=${total_epochs_cap} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode="${resume_mode}"
