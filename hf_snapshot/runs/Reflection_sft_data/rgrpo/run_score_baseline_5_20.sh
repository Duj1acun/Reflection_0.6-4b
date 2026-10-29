#!/usr/bin/env bash
# SCoRe baseline on the existing verl/eval math pipeline.
#
# This keeps the model, train/eval JSONL, math checker, validation, W&B, and
# dump format aligned with run_trajectory_grpo_5_20.sh. Only the training
# rollout/objective is changed to the two-stage SCoRe objective.

set -xeo pipefail

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
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
export RAY_TMPDIR="${RAY_TMPDIR:-/ssd/data/wangruxin/ray_score_5_20}"
unset RAY_ADDRESS RAY_INIT_ADDRESS || true
export RAY_INCLUDE_DASHBOARD="${RAY_INCLUDE_DASHBOARD:-false}"
mkdir -p "${RAY_TMPDIR}"

if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
    unset PYTORCH_CUDA_ALLOC_CONF
fi

# The SCoRe trainer computes the original SCoRe format_and_match reward inside
# ray_trainer.py. The custom checker path is retained for compatibility with the
# existing verl configuration, but it is not the main SCoRe training reward.
export HYBRID_V2_MATH_SCORE_BACKEND="${HYBRID_V2_MATH_SCORE_BACKEND:-process}"
export HYBRID_V2_MATH_SCORE_TIMEOUT="${HYBRID_V2_MATH_SCORE_TIMEOUT:-30}"
export TRAJ_GRPO_REWARD_PHI="${TRAJ_GRPO_REWARD_PHI:-4}"
export TRAJ_GRPO_REWARD_STYLE="${TRAJ_GRPO_REWARD_STYLE:-efficiency}"
export TRAJ_GRPO_EFF_FREE_REFLECTIONS="${TRAJ_GRPO_EFF_FREE_REFLECTIONS:-1}"
export TRAJ_GRPO_REQUIRE_REFLECTION_FORMAT="${TRAJ_GRPO_REQUIRE_REFLECTION_FORMAT:-1}"
export TRAJ_GRPO_WRONG_NO_REFLECT_PENALTY="${TRAJ_GRPO_WRONG_NO_REFLECT_PENALTY:-1}"
export TRAJ_GRPO_WRONG_NO_REFLECT_N="${TRAJ_GRPO_WRONG_NO_REFLECT_N:-3}"

if [[ -z "${SCORE_SYSTEM_PROMPT:-}" ]]; then
    SCORE_SYSTEM_PROMPT="$(cat <<'EOF'
You are an expert math tutor. Solve each problem in two parts:

1. Wrap all your reasoning in <think>...</think> tags. Work through the
   problem step by step inside that block.
2. After </think>, wrap your final answer in <answer>...</answer> tags.
   Put only the bare answer between the tags -- no surrounding prose, no
   leading "The answer is".

Answer-format examples:
- <answer>42</answer>
- <answer>3.14</answer>
- <answer>-2/3</answer>
- <answer>y = 2x - 13</answer>
- <answer>\boxed{x^2 + 1}</answer>
EOF
)"
fi
export SCORE_SYSTEM_PROMPT

if [[ -z "${SCORE_SELF_CORRECTION_PROMPT:-}" ]]; then
    SCORE_SELF_CORRECTION_PROMPT="$(cat <<'EOF'
There might be an error in the solution above. Review the prior reasoning,
identify any mistakes, and rewrite the solution.

Use the same format: <think>...</think> for your corrected reasoning, then
<answer>...</answer> for your final answer.
EOF
)"
fi
export SCORE_SELF_CORRECTION_PROMPT

project_name="${PROJECT_NAME:-SCoRe-Baseline-Math}"
exp_name="${EXP_NAME:-score-baseline-qwen2.5-math-1.5b-5-20}"
resume_mode="${RESUME_MODE:-disable}"
resume_from_path="${RESUME_FROM_PATH:-null}"

adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0

max_prompt_length="${MAX_PROMPT_LENGTH:-1024}"
max_response_length="${MAX_RESPONSE_LENGTH:-1024}"
max_model_len="${MAX_MODEL_LEN:-4096}"
score_max_prompt_length_attempt2="${SCORE_MAX_PROMPT_LENGTH_ATTEMPT2:-2048}"

train_prompt_bsz="${TRAIN_PROMPT_BSZ:-64}"
train_prompt_mini_bsz="${PPO_MINI_BATCH_SIZE:-$((train_prompt_bsz * 2))}"
ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}"
val_batch_size="${VAL_BATCH_SIZE:-512}"
val_before_train="${VAL_BEFORE_TRAIN:-True}"
test_freq="${TEST_FREQ:-50}"
save_freq="${SAVE_FREQ:-50}"
total_training_steps="${TOTAL_TRAINING_STEPS:-500}"
score_stage1_steps="${SCORE_STAGE1_STEPS:-$((total_training_steps / 2))}"
total_epochs_cap="${TOTAL_EPOCHS_CAP:-100000}"

loss_agg_mode="${LOSS_AGG_MODE:-seq-mean-token-mean}"

NNODES="${NNODES:-1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-1}"

RAY_DATA_HOME="${RAY_DATA_HOME:-/ssd/data/wangruxin/R_grpo_sft}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/ssd/data/wangruxin/hf_cache/Qwen2.5-Math-1.5B}"
MODEL_PATH="${MODEL_PATH:-${BASE_MODEL_PATH}}"
CKPTS_DIR="${CKPTS_DIR:-${RAY_DATA_HOME}/RewardFlow/outputs/score_baseline/${exp_name}}"
TRAIN_FILE="${TRAIN_FILE:-${RAY_DATA_HOME}/data/math_lighteval_rgrpo.jsonl}"
TEST_FILE="${TEST_FILE:-${RAY_DATA_HOME}/data/math500_rgrpo_fixed.jsonl}"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "[ERROR] MODEL_PATH=${MODEL_PATH} does not contain a HuggingFace model (missing config.json)." >&2
    exit 1
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
rollout_max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-262144}"
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
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.20 \
    actor_rollout_ref.actor.clip_ratio_high=0.26 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${log_prob_max_token_len_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${log_prob_max_token_len_per_gpu} \
    actor_rollout_ref.rollout.n=1 \
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
    algorithm.filter_groups.enable=False \
    +algorithm.score_baseline.enable=True \
    +algorithm.score_baseline.stage1_steps=${score_stage1_steps} \
    +algorithm.score_baseline.beta1=${SCORE_BETA1:-0.01} \
    +algorithm.score_baseline.beta2=${SCORE_BETA2:-0.1} \
    +algorithm.score_baseline.alpha=${SCORE_ALPHA:-10.0} \
    +algorithm.score_baseline.length_norm="${SCORE_LENGTH_NORM:-sequence}" \
    +algorithm.score_baseline.max_prompt_length_attempt2=${score_max_prompt_length_attempt2} \
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
