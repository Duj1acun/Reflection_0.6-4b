#!/usr/bin/env bash
set -xeo pipefail

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONPATH=/ssd/data/wangruxin/Relf_grpo:${PYTHONPATH:-}
export VLLM_USE_V1=0
export RAY_INCLUDE_DASHBOARD="${RAY_INCLUDE_DASHBOARD:-false}"
unset RAY_ADDRESS RAY_INIT_ADDRESS || true

project_name="${PROJECT_NAME:-Trajectory-GRPO-Math}"
exp_name="${EXP_NAME:-trajectory-grpo-qwen2.5-math-1.5b-dg-cont-only-kl-clip020-gpu0-5-21}"
resume_mode="${RESUME_MODE:-disable}"
resume_from_path="${RESUME_FROM_PATH:-null}"

# Keep Ray and outputs isolated from the earlier 5-20 continuation-only run.
# Keep the Ray path short: AF_UNIX socket paths must stay under 107 bytes.
export RAY_TMPDIR="${RAY_TMPDIR:-/ssd/data/wangruxin/ray_cg0_kl020}"
export CKPTS_DIR="${CKPTS_DIR:-/ssd/data/wangruxin/R_grpo_sft/RewardFlow/outputs/trajectory_grpo/${exp_name}}"
export VAL_DUMP_DIR="${VAL_DUMP_DIR:-/ssd/data/wangruxin/outputs/val_result/${exp_name}}"
export LOG_DIR="${LOG_DIR:-/ssd/data/wangruxin/outputs/train_logs}"

MIN_FREE_MIB="${MIN_FREE_MIB:-76000}"
WAIT_INTERVAL_SECONDS="${WAIT_INTERVAL_SECONDS:-60}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-0}"  # 0 means wait indefinitely.

gpu_compute_apps() {
    nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" \
        --query-compute-apps=pid,process_name,used_gpu_memory \
        --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d'
}

gpu_free_mib() {
    nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" \
        --query-gpu=memory.free \
        --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -dc '0-9'
}

mkdir -p "${LOG_DIR}" "${RAY_TMPDIR}" "${CKPTS_DIR}" "${VAL_DUMP_DIR}"

echo "[INFO] Standalone continuation-only trajectory GRPO + KL loss"
echo "[INFO] EXP_NAME=${exp_name}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] RAY_TMPDIR=${RAY_TMPDIR}"
echo "[INFO] CKPTS_DIR=${CKPTS_DIR}"
echo "[INFO] VAL_DUMP_DIR=${VAL_DUMP_DIR}"
echo "[INFO] Waiting for selected GPU to be empty and have >= ${MIN_FREE_MIB} MiB free"

start_ts="$(date +%s)"
while true; do
    apps="$(gpu_compute_apps || true)"
    free_mib="$(gpu_free_mib || true)"

    if [[ -z "${apps}" && -n "${free_mib}" ]] && (( free_mib >= MIN_FREE_MIB )); then
        echo "[INFO] GPU ready: free_mib=${free_mib}, compute_apps=0"
        break
    fi

    now_ts="$(date +%s)"
    elapsed=$(( now_ts - start_ts ))
    if (( MAX_WAIT_SECONDS > 0 && elapsed >= MAX_WAIT_SECONDS )); then
        echo "[ERROR] Timed out waiting for GPU after ${elapsed}s" >&2
        echo "[ERROR] free_mib=${free_mib:-unknown}, required=${MIN_FREE_MIB}" >&2
        if [[ -n "${apps}" ]]; then
            echo "[ERROR] Current GPU compute apps:" >&2
            echo "${apps}" | sed 's/^/[ERROR]   /' >&2
        fi
        exit 1
    fi

    echo "[WAIT] GPU not ready after ${elapsed}s: free_mib=${free_mib:-unknown}, required=${MIN_FREE_MIB}"
    if [[ -n "${apps}" ]]; then
        echo "[WAIT] GPU compute apps:"
        echo "${apps}" | sed 's/^/[WAIT]   /'
    fi
    sleep "${WAIT_INTERVAL_SECONDS}"
done

if [[ "${SKIP_GPU_PREFLIGHT:-0}" != "1" ]]; then
    _free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
        -i "${CUDA_VISIBLE_DEVICES}" 2>/dev/null | head -1 | tr -dc '0-9')"
    if [[ -n "${_free_mib}" ]] && (( _free_mib < ${MIN_FREE_MIB:-76000} )); then
        echo "[ERROR] GPU ${CUDA_VISIBLE_DEVICES} has only ${_free_mib} MiB free" >&2
        echo "[ERROR] (< ${MIN_FREE_MIB:-76000}). Refusing to launch." >&2
        exit 1
    fi
fi

if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
    unset PYTORCH_CUDA_ALLOC_CONF
fi

# Reward and trajectory masking: same continuation-only setting as the 5-20 run.
export HYBRID_V2_MATH_SCORE_BACKEND="${HYBRID_V2_MATH_SCORE_BACKEND:-process}"
export HYBRID_V2_MATH_SCORE_TIMEOUT="${HYBRID_V2_MATH_SCORE_TIMEOUT:-30}"
export TRAJ_GRPO_REWARD_PHI="${TRAJ_GRPO_REWARD_PHI:-4}"
export TRAJ_GRPO_REWARD_STYLE="${TRAJ_GRPO_REWARD_STYLE:-efficiency}"
export TRAJ_GRPO_EFF_FREE_REFLECTIONS="${TRAJ_GRPO_EFF_FREE_REFLECTIONS:-1}"
export TRAJ_GRPO_BINARY_DIRECT_WRONG_REWARD="${TRAJ_GRPO_BINARY_DIRECT_WRONG_REWARD:-0.0}"
export TRAJ_GRPO_BINARY_WRONG_REFLECTION_PENALTY="${TRAJ_GRPO_BINARY_WRONG_REFLECTION_PENALTY:-1}"
export TRAJ_GRPO_WRONG_REFL_PENALTY_A="${TRAJ_GRPO_WRONG_REFL_PENALTY_A:-0.05}"
export TRAJ_GRPO_WRONG_REFL_PENALTY_B="${TRAJ_GRPO_WRONG_REFL_PENALTY_B:-0.15}"
export TRAJ_GRPO_WRONG_REFL_PENALTY_C="${TRAJ_GRPO_WRONG_REFL_PENALTY_C:-0.25}"
export TRAJ_GRPO_REQUIRE_REFLECTION_FORMAT="${TRAJ_GRPO_REQUIRE_REFLECTION_FORMAT:-1}"
export TRAJ_GRPO_REFLECTION_FREQ_PENALTY="0.0"
export TRAJ_GRPO_WRONG_REFLECTION_FREQ_PENALTY="0.0"
export TRAJ_GRPO_WRONG_NO_REFLECT_PENALTY="${TRAJ_GRPO_WRONG_NO_REFLECT_PENALTY:-1}"
export TRAJ_GRPO_WRONG_NO_REFLECT_N="${TRAJ_GRPO_WRONG_NO_REFLECT_N:-3}"
export TRAJ_GRPO_MASK_LAST_REFLECTION_ONLY="1"
export TRAJ_GRPO_TRAIN_FULL_CHAIN="0"
export TRAJ_GRPO_WRONG_REFLECTION_FULL_CHAIN_MASK="0"
export TRAJ_GRPO_USE_EFF_REWARD="0"

# Direction-2 direct baseline remains active; reflect-advantage floor stays off.
export TRAJ_GRPO_DECOUPLED_BASELINE="${TRAJ_GRPO_DECOUPLED_BASELINE:-1}"
export TRAJ_GRPO_REFLECT_FLOOR_DIFFICULTY_GATED="0"
export TRAJ_GRPO_REFLECT_ADV_FLOOR_DECAY_STEPS="0"
export TRAJ_GRPO_REFLECT_ADV_FLOOR="-1000.0"

adv_estimator=grpo
grpo_skip_std_for_uniform_outcome="${GRPO_SKIP_STD_FOR_UNIFORM_OUTCOME:-True}"
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef="${KL_LOSS_COEF:-0.01}"

clip_ratio_low="0.20"
clip_ratio_high="0.20"

max_prompt_length="${MAX_PROMPT_LENGTH:-1024}"
max_response_length="${MAX_RESPONSE_LENGTH:-1024}"
max_model_len="${MAX_MODEL_LEN:-4096}"
max_iters="${MAX_ITERS:-3}"

trajectory_max_len=$((max_prompt_length + max_iters * max_response_length))
if (( trajectory_max_len > max_model_len )) && [[ "${ALLOW_LONG_TRAJECTORY:-0}" != "1" ]]; then
    echo "[ERROR] max_prompt_length + max_iters * max_response_length = ${trajectory_max_len}, exceeds MAX_MODEL_LEN=${max_model_len}."
    echo "[ERROR] Use e.g. MAX_RESPONSE_LENGTH=1024 MAX_ITERS=3, or set a real long-context MODEL_PATH/MAX_MODEL_LEN."
    echo "[ERROR] Set ALLOW_LONG_TRAJECTORY=1 only if the actor model truly supports the longer sequence."
    exit 1
fi

train_prompt_bsz="${TRAIN_PROMPT_BSZ:-64}"
n_resp_per_prompt="${N_RESP_PER_PROMPT:-8}"
train_prompt_mini_bsz="${PPO_MINI_BATCH_SIZE:-64}"
ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}"
val_batch_size="${VAL_BATCH_SIZE:-512}"
val_before_train="${VAL_BEFORE_TRAIN:-True}"
test_freq="${TEST_FREQ:-10}"
save_freq="${SAVE_FREQ:-50}"
total_training_steps="${TOTAL_TRAINING_STEPS:-1000}"
total_epochs_cap="${TOTAL_EPOCHS_CAP:-100000}"

loss_agg_mode="${LOSS_AGG_MODE:-seq-mean-token-mean}"

NNODES="${NNODES:-1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-1}"

RAY_DATA_HOME="${RAY_DATA_HOME:-/ssd/data/wangruxin/R_grpo_sft}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/ssd/data/wangruxin/Qwen2.5-Math-1.5B}"
SFT_ADAPTER_PATH="${SFT_ADAPTER_PATH:-/ssd/data/wangruxin/reasoning_paths_directional/sft_training/outputs/phase2/final}"
MERGED_SFT_MODEL_PATH="${MERGED_SFT_MODEL_PATH:-${RAY_DATA_HOME}/sft_training/outputs/phase2/final_merged_hf}"
MODEL_PATH="${MODEL_PATH:-${MERGED_SFT_MODEL_PATH}}"
TRAIN_FILE="${TRAIN_FILE:-${RAY_DATA_HOME}/data/math_lighteval_rgrpo.jsonl}"
TEST_FILE="${TEST_FILE:-${RAY_DATA_HOME}/data/math500_rgrpo_fixed.jsonl}"

echo "[INFO] Resolved model paths:"
echo "[INFO]   BASE_MODEL_PATH      = ${BASE_MODEL_PATH}"
echo "[INFO]   SFT_ADAPTER_PATH     = ${SFT_ADAPTER_PATH}"
echo "[INFO]   MERGED_SFT_MODEL_PATH= ${MERGED_SFT_MODEL_PATH}"
echo "[INFO]   MODEL_PATH           = ${MODEL_PATH}"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    if [[ "${MODEL_PATH}" == "${MERGED_SFT_MODEL_PATH}" ]] && [[ -f "${SFT_ADAPTER_PATH}/adapter_config.json" ]]; then
        echo "[INFO] Merging SFT adapter into base model..."
        BASE_MODEL_PATH="${BASE_MODEL_PATH}" SFT_ADAPTER_PATH="${SFT_ADAPTER_PATH}" MERGED_SFT_MODEL_PATH="${MERGED_SFT_MODEL_PATH}" python3 - <<'PY'
import json
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
    acfg_path = os.path.join(adapter, "adapter_config.json")
    if os.path.isfile(acfg_path):
        with open(acfg_path) as f:
            recorded_base = (json.load(f).get("base_model_name_or_path") or "").rstrip("/")
        if recorded_base and os.path.basename(recorded_base) != os.path.basename(base.rstrip("/")):
            msg = (
                f"[ERROR] SFT adapter was trained on base "
                f"'{recorded_base}' but BASE_MODEL_PATH='{base}'. "
                f"Set BASE_MODEL_PATH to the matching base "
                f"(or ALLOW_SFT_BASE_MISMATCH=1 if you are certain it is the "
                f"same model under a different path)."
            )
            if os.environ.get("ALLOW_SFT_BASE_MISMATCH", "0") not in {"1", "true", "yes"}:
                raise SystemExit(msg)
            print(msg.replace("[ERROR]", "[WARN] (override)"))
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

    tok_src = adapter if os.path.isfile(os.path.join(adapter, "tokenizer.json")) else base
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    tokenizer.save_pretrained(out)
    print(f"[INFO] Saved merged model to {out}")
PY
    else
        echo "[ERROR] MODEL_PATH=${MODEL_PATH} does not contain a HuggingFace model (missing config.json)." >&2
        exit 1
    fi
fi

echo "[INFO] Using model path: ${MODEL_PATH}"
LOG_FILE="${LOG_DIR}/${exp_name}_$(date +%Y%m%d_%H%M%S).log"
echo "[INFO] Training log -> ${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[INFO] Validation dumps -> ${VAL_DUMP_DIR}"
echo "[INFO] TRAIN_FULL_CHAIN=${TRAJ_GRPO_TRAIN_FULL_CHAIN} MASK_LAST_ONLY=${TRAJ_GRPO_MASK_LAST_REFLECTION_ONLY}"
echo "[INFO] USE_KL_LOSS=${use_kl_loss} KL_LOSS_COEF=${kl_loss_coef}"
echo "[INFO] CLIP_RATIO_LOW=${clip_ratio_low} CLIP_RATIO_HIGH=${clip_ratio_high}"

temperature="${TEMPERATURE:-0.7}"
top_p="${TOP_P:-1.0}"
top_k="${TOP_K:--1}"

sp_size=1
use_dynamic_bsz=True
gen_tp="${GEN_TP:-1}"
offload="${OFFLOAD:-True}"
actor_lr="${ACTOR_LR:-1e-6}"
actor_weight_decay="${ACTOR_WEIGHT_DECAY:-0.0}"
actor_grad_clip="${ACTOR_GRAD_CLIP:-1.0}"
actor_ppo_max_token_len_per_gpu="${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-32768}"
log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-32}"
log_prob_max_token_len_per_gpu="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-65536}"
rollout_gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.45}"
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
    algorithm.grpo_skip_std_for_uniform_outcome=${grpo_skip_std_for_uniform_outcome} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    +algorithm.reflection_grpo.enable=True \
    +algorithm.reflection_grpo.max_iters=${max_iters} \
    algorithm.filter_groups.enable=False \
    custom_reward_function.name=compute_trajectory_grpo_score \
    custom_reward_function.path=rgrpo.core_trajectory_grpo_decoupled_dg \
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
    trainer.resume_mode="${resume_mode}" \
    trainer.resume_from_path="${resume_from_path}"
