#!/usr/bin/env bash
# Decoupled-baseline + DIFFICULTY-GATED-FLOOR trajectory GRPO.
#
# Model:  Qwen2.5-1.5B-Instruct + phase2_sweep/checkpoint-100 SFT adapter
#         (merged on first launch to checkpoint-100_merged_hf)
# GPU:    GPU2 (CUDA_DEVICE_ORDER=PCI_BUS_ID, CUDA_VISIBLE_DEVICES=2)
#
# Training parameters identical to trajectory-grpo-qwen2.5-math-1.5b-dg-5-20:
#   TRAJ_GRPO_REFLECT_ADV_FLOOR=-1.0  (floor0=-1.0, confirmed from dg-5-20 log)
# All other TRAJ_GRPO_* / verl / trainer params match run_trajectory_grpo_decoupled_dg_math.sh.

set -xeo pipefail

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

# GPU preflight: refuse to start if GPU2 has < MIN_FREE_MIB VRAM free.
if [[ "${SKIP_GPU_PREFLIGHT:-0}" != "1" ]]; then
    _free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
        -i "${CUDA_VISIBLE_DEVICES}" 2>/dev/null | head -1 | tr -dc '0-9')"
    if [[ -n "${_free_mib}" ]] && (( _free_mib < ${MIN_FREE_MIB:-45000} )); then
        echo "[ERROR] GPU ${CUDA_VISIBLE_DEVICES} has only ${_free_mib} MiB free" >&2
        echo "[ERROR] (< ${MIN_FREE_MIB:-45000}). Refusing to launch." >&2
        exit 1
    fi
fi

export PYTHONPATH=/ssd/data/wangruxin/Relf_grpo:${PYTHONPATH:-}
export VLLM_USE_V1=0
# Isolated Ray cluster — must not share with any concurrent run.
export RAY_TMPDIR="${RAY_TMPDIR:-/ssd/data/wangruxin/ray_instruct_dg_gpu2}"
unset RAY_ADDRESS RAY_INIT_ADDRESS || true
export RAY_INCLUDE_DASHBOARD="${RAY_INCLUDE_DASHBOARD:-false}"
mkdir -p "${RAY_TMPDIR}"

if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
    unset PYTORCH_CUDA_ALLOC_CONF
fi

# Math scorer (same as dg-5-20).
export HYBRID_V2_MATH_SCORE_BACKEND="${HYBRID_V2_MATH_SCORE_BACKEND:-process}"
export HYBRID_V2_MATH_SCORE_TIMEOUT="${HYBRID_V2_MATH_SCORE_TIMEOUT:-30}"

# ── Reward env vars (identical to run_trajectory_grpo_decoupled_dg_math.sh) ──
export TRAJ_GRPO_REWARD_PHI="${TRAJ_GRPO_REWARD_PHI:-4}"
export TRAJ_GRPO_REWARD_STYLE="${TRAJ_GRPO_REWARD_STYLE:-efficiency}"
export TRAJ_GRPO_EFF_FREE_REFLECTIONS="${TRAJ_GRPO_EFF_FREE_REFLECTIONS:-1}"
export TRAJ_GRPO_BINARY_DIRECT_WRONG_REWARD="${TRAJ_GRPO_BINARY_DIRECT_WRONG_REWARD:--0.25}"
export TRAJ_GRPO_BINARY_WRONG_REFLECTION_PENALTY="${TRAJ_GRPO_BINARY_WRONG_REFLECTION_PENALTY:-1}"
export TRAJ_GRPO_WRONG_REFL_PENALTY_A="${TRAJ_GRPO_WRONG_REFL_PENALTY_A:-0.05}"
export TRAJ_GRPO_WRONG_REFL_PENALTY_B="${TRAJ_GRPO_WRONG_REFL_PENALTY_B:-0.15}"
export TRAJ_GRPO_WRONG_REFL_PENALTY_C="${TRAJ_GRPO_WRONG_REFL_PENALTY_C:-0.25}"
export TRAJ_GRPO_REQUIRE_REFLECTION_FORMAT="${TRAJ_GRPO_REQUIRE_REFLECTION_FORMAT:-1}"
export TRAJ_GRPO_REFLECTION_FREQ_PENALTY="${TRAJ_GRPO_REFLECTION_FREQ_PENALTY:-0.0}"
export TRAJ_GRPO_WRONG_REFLECTION_FREQ_PENALTY="${TRAJ_GRPO_WRONG_REFLECTION_FREQ_PENALTY:-0.0}"
export TRAJ_GRPO_WRONG_NO_REFLECT_PENALTY="${TRAJ_GRPO_WRONG_NO_REFLECT_PENALTY:-1}"
export TRAJ_GRPO_WRONG_NO_REFLECT_N="${TRAJ_GRPO_WRONG_NO_REFLECT_N:-3}"
export TRAJ_GRPO_MASK_LAST_REFLECTION_ONLY="${TRAJ_GRPO_MASK_LAST_REFLECTION_ONLY:-0}"
export TRAJ_GRPO_TRAIN_FULL_CHAIN="${TRAJ_GRPO_TRAIN_FULL_CHAIN:-1}"
export TRAJ_GRPO_WRONG_REFLECTION_FULL_CHAIN_MASK="${TRAJ_GRPO_WRONG_REFLECTION_FULL_CHAIN_MASK:-0}"

# ── Advantage env vars ────────────────────────────────────────────────────────
export TRAJ_GRPO_DECOUPLED_BASELINE="${TRAJ_GRPO_DECOUPLED_BASELINE:-1}"
# floor0=-1.0: confirmed from dg-5-20 log (identical to that run).
export TRAJ_GRPO_REFLECT_ADV_FLOOR="${TRAJ_GRPO_REFLECT_ADV_FLOOR:--1.0}"
export TRAJ_GRPO_REFLECT_ADV_FLOOR_DECAY_STEPS="${TRAJ_GRPO_REFLECT_ADV_FLOOR_DECAY_STEPS:-100}"
export TRAJ_GRPO_REFLECT_FLOOR_DIFFICULTY_GATED="${TRAJ_GRPO_REFLECT_FLOOR_DIFFICULTY_GATED:-1}"

project_name="${PROJECT_NAME:-Trajectory-GRPO-Math}"
exp_name="${EXP_NAME:-trajectory-grpo-qwen2.5-1.5b-instruct-dg-5-21}"
# Fresh start: disable auto-resume so a botched first launch cannot silently
# resume a corrupt checkpoint. Switch to 'auto' for subsequent restarts.
resume_mode="${RESUME_MODE:-disable}"
resume_from_path="${RESUME_FROM_PATH:-null}"

adv_estimator=grpo
grpo_skip_std_for_uniform_outcome="${GRPO_SKIP_STD_FOR_UNIFORM_OUTCOME:-True}"
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef="${KL_LOSS_COEF:-0.0}"

clip_ratio_low="${CLIP_RATIO_LOW:-0.20}"
clip_ratio_high="${CLIP_RATIO_HIGH:-0.26}"

max_prompt_length="${MAX_PROMPT_LENGTH:-1024}"
max_response_length="${MAX_RESPONSE_LENGTH:-1024}"
max_model_len="${MAX_MODEL_LEN:-4096}"
max_iters="${MAX_ITERS:-3}"

trajectory_max_len=$((max_prompt_length + max_iters * max_response_length))
if (( trajectory_max_len > max_model_len )) && [[ "${ALLOW_LONG_TRAJECTORY:-0}" != "1" ]]; then
    echo "[ERROR] max_prompt_length + max_iters * max_response_length = ${trajectory_max_len} > MAX_MODEL_LEN=${max_model_len}." >&2
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
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/ssd/data/wangruxin/hf_cache/Qwen2.5-1.5B-Instruct}"
SFT_ADAPTER_PATH="${SFT_ADAPTER_PATH:-${RAY_DATA_HOME}/sft_training/outputs/qwen25_1.5b_instruct/phase2_sweep/checkpoint-1100}"
MERGED_SFT_MODEL_PATH="${MERGED_SFT_MODEL_PATH:-${RAY_DATA_HOME}/sft_training/outputs/qwen25_1.5b_instruct/phase2_sweep/checkpoint-1100_merged_hf}"
MODEL_PATH="${MODEL_PATH:-${MERGED_SFT_MODEL_PATH}}"
CKPTS_DIR="${CKPTS_DIR:-${RAY_DATA_HOME}/RewardFlow/outputs/trajectory_grpo/${exp_name}}"
TRAIN_FILE="${TRAIN_FILE:-${RAY_DATA_HOME}/data/math_lighteval_rgrpo.jsonl}"
TEST_FILE="${TEST_FILE:-${RAY_DATA_HOME}/data/math500_rgrpo_fixed.jsonl}"

echo "[INFO] Resolved model paths:"
echo "[INFO]   BASE_MODEL_PATH       = ${BASE_MODEL_PATH}"
echo "[INFO]   SFT_ADAPTER_PATH      = ${SFT_ADAPTER_PATH}"
echo "[INFO]   MERGED_SFT_MODEL_PATH = ${MERGED_SFT_MODEL_PATH}"
echo "[INFO]   MODEL_PATH            = ${MODEL_PATH}"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    if [[ "${MODEL_PATH}" == "${MERGED_SFT_MODEL_PATH}" ]] && [[ -f "${SFT_ADAPTER_PATH}/adapter_config.json" ]]; then
        echo "[INFO] Merging SFT adapter into base model..."
        BASE_MODEL_PATH="${BASE_MODEL_PATH}" \
        SFT_ADAPTER_PATH="${SFT_ADAPTER_PATH}" \
        MERGED_SFT_MODEL_PATH="${MERGED_SFT_MODEL_PATH}" \
        python3 - <<'PY'
import os, json, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base    = os.environ["BASE_MODEL_PATH"]
adapter = os.environ["SFT_ADAPTER_PATH"]
out     = os.environ["MERGED_SFT_MODEL_PATH"]

if os.path.isfile(os.path.join(out, "config.json")):
    print(f"[INFO] Merged model already exists at {out}, skipping.")
else:
    acfg = os.path.join(adapter, "adapter_config.json")
    if os.path.isfile(acfg):
        recorded = (json.load(open(acfg)).get("base_model_name_or_path") or "").rstrip("/")
        if recorded and os.path.basename(recorded) != os.path.basename(base.rstrip("/")):
            if os.environ.get("ALLOW_SFT_BASE_MISMATCH", "0") not in {"1", "true", "yes"}:
                raise SystemExit(
                    f"[ERROR] Adapter base '{recorded}' != BASE_MODEL_PATH '{base}'. "
                    "Set ALLOW_SFT_BASE_MISMATCH=1 to override."
                )
    os.makedirs(out, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32,
                                                  device_map="cpu", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, adapter)
    model = model.merge_and_unload()
    model.save_pretrained(out, safe_serialization=True)
    tok_src = adapter if os.path.isfile(os.path.join(adapter, "tokenizer.json")) else base
    AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True).save_pretrained(out)
    print(f"[INFO] Saved merged model to {out}")
PY
    else
        echo "[ERROR] MODEL_PATH=${MODEL_PATH} has no config.json and no adapter to merge." >&2
        exit 1
    fi
fi

echo "[INFO] Using model path: ${MODEL_PATH}"
mkdir -p "${CKPTS_DIR}"

LOG_DIR="${LOG_DIR:-/ssd/data/wangruxin/outputs/train_logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${exp_name}_$(date +%Y%m%d_%H%M%S).log"
echo "[INFO] Training log -> ${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

export VAL_DUMP_DIR="${VAL_DUMP_DIR:-/ssd/data/wangruxin/outputs/val_result/${exp_name}}"
mkdir -p "${VAL_DUMP_DIR}"
echo "[INFO] Validation dumps -> ${VAL_DUMP_DIR}"

temperature="${TEMPERATURE:-1.0}"
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
