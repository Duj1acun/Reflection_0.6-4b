#!/usr/bin/env bash
# Shared experiment settings aligned with reflection_grpo_clean's GRPO/REFLECT+.
set -euo pipefail

METHOD="${1:?Expected pag or score}"
shift
[[ "$METHOD" == pag || "$METHOD" == score ]] || { echo 'Expected pag or score' >&2; exit 2; }
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REFERENCE_ROOT="${REFERENCE_ROOT:-/root/reflection_grpo_clean}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:-${BASE_MODEL_PATH:-$REFERENCE_ROOT/models/Qwen2.5-Math-1.5B}}"
TRAIN_FILE="${TRAIN_FILE:-$REPO_DIR/datasets/limo_aligned.parquet}"
TEST_FILE="${TEST_FILE:-$REPO_DIR/datasets/math500_aligned.parquet}"
PROJECT_NAME="${PROJECT_NAME:-REFLECT-Plus}"
EXP_NAME="${EXP_NAME:-${METHOD}-qwen2.5-math-1.5b-aligned}"
CKPTS_DIR="${CKPTS_DIR:-$REPO_DIR/checkpoints/$PROJECT_NAME/$EXP_NAME}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1000}"
SCORE_STAGE1_STEPS="${SCORE_STAGE1_STEPS:-$((TOTAL_TRAINING_STEPS / 2))}"
SCORE_STAGE="${SCORE_STAGE:-both}"
RESUME_MODE="${RESUME_MODE:-disable}"
DRY_RUN="${DRY_RUN:-0}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
if [[ "$METHOD" == pag ]]; then
    MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}" # answer + verifier + revision
else
    MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1536}" # answer + revision
    [[ "$SCORE_STAGE" == both || "$SCORE_STAGE" == 1 || "$SCORE_STAGE" == 2 ]] || exit 2
    if (( SCORE_STAGE1_STEPS < 1 || SCORE_STAGE1_STEPS >= TOTAL_TRAINING_STEPS )); then
        echo 'Require 0 < SCORE_STAGE1_STEPS < TOTAL_TRAINING_STEPS (cumulative).' >&2
        exit 2
    fi
fi

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_V1=0
cd "$REPO_DIR"

COMMON_ARGS=(
    algorithm.adv_estimator=gae
    "data.train_files=['$TRAIN_FILE']"
    "data.val_files=['$TEST_FILE']"
    data.prompt_key=prompt
    data.filter_overlong_prompts=False
    data.truncation=left
    "data.train_batch_size=${TRAIN_PROMPT_BSZ:-64}"
    "data.val_batch_size=${VAL_BATCH_SIZE:-512}"
    "data.max_prompt_length=$MAX_PROMPT_LENGTH"
    "data.max_response_length=$MAX_RESPONSE_LENGTH"
    "actor_rollout_ref.model.path=$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.use_dynamic_bsz=True
    "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-64}"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}"
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-32768}"
    actor_rollout_ref.actor.ppo_epochs=1
    "actor_rollout_ref.actor.optim.lr=${ACTOR_LR:-1e-6}"
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0
    "actor_rollout_ref.actor.optim.weight_decay=${ACTOR_WEIGHT_DECAY:-0.0}"
    "actor_rollout_ref.actor.grad_clip=${ACTOR_GRAD_CLIP:-1.0}"
    "actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG_MODE:-seq-mean-token-mean}"
    "actor_rollout_ref.actor.fsdp_config.param_offload=${OFFLOAD:-True}"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OFFLOAD:-True}"
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.0
    actor_rollout_ref.actor.entropy_coeff=0
    "actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW:-0.20}"
    "actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH:-0.26}"
    "actor_rollout_ref.ref.fsdp_config.param_offload=${REF_OFFLOAD:-False}"
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-32}"
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-65536}"
    actor_rollout_ref.rollout.name=vllm
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP:-1}"
    "actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-131072}"
    "actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS:-128}"
    "actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN"
    "actor_rollout_ref.rollout.enforce_eager=${ENFORCE_EAGER:-False}"
    actor_rollout_ref.rollout.free_cache_engine=False
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.45}"
    "actor_rollout_ref.rollout.n=${N_RESP_PER_PROMPT:-8}"
    "actor_rollout_ref.rollout.temperature=${TEMPERATURE:-1.0}"
    "actor_rollout_ref.rollout.top_p=${TOP_P:-1.0}"
    "actor_rollout_ref.rollout.top_k=${TOP_K:--1}"
    actor_rollout_ref.rollout.num_turns=2
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-32}"
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-65536}"
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.val_kwargs.do_sample=False
    actor_rollout_ref.rollout.val_kwargs.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0
    actor_rollout_ref.rollout.val_kwargs.temperature=0
    actor_rollout_ref.rollout.val_kwargs.num_turns=2
    reward_model.rs_coef=1.0
    "critic.optim.lr=${CRITIC_LR:-2e-6}"
    critic.use_dynamic_bsz=True
    critic.model.use_remove_padding=True
    "critic.model.path=$MODEL_PATH"
    "critic.model.fsdp_config.param_offload=${CRITIC_OFFLOAD:-${OFFLOAD:-True}}"
    "critic.model.fsdp_config.optimizer_offload=${CRITIC_OFFLOAD:-${OFFLOAD:-True}}"
    "critic.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}"
    "critic.ppo_max_token_len_per_gpu=${CRITIC_PPO_MAX_TOKEN_LEN_PER_GPU:-32768}"
    "trainer.logger=${LOGGER:-['console','wandb']}"
    "trainer.project_name=$PROJECT_NAME"
    "trainer.experiment_name=$EXP_NAME"
    "trainer.n_gpus_per_node=${NGPUS_PER_NODE:-1}"
    "trainer.nnodes=${NNODES:-1}"
    "trainer.save_freq=${SAVE_FREQ:-50}"
    "trainer.test_freq=${TEST_FREQ:-10}"
    "trainer.total_epochs=${TOTAL_EPOCHS_CAP:-100000}"
    "trainer.default_local_dir=$CKPTS_DIR"
    trainer.log_val_generations=2
)

run_ppo() {
    if [[ "$DRY_RUN" == 1 ]]; then
        printf '%q ' "$PYTHON_BIN" -m verl.trainer.main_ppo "${COMMON_ARGS[@]}" "$@"
        printf '\n'
    else
        "$PYTHON_BIN" -m verl.trainer.main_ppo "${COMMON_ARGS[@]}" "$@"
    fi
}

if [[ "$DRY_RUN" != 1 ]]; then
    "$PYTHON_BIN" "$REPO_DIR/quick_start/check_aligned_inputs.py" \
        "$METHOD" "$MODEL_PATH" "$TRAIN_FILE" "$TEST_FILE" \
        "$MAX_PROMPT_LENGTH" "$MAX_RESPONSE_LENGTH" "$MAX_MODEL_LEN"
fi

if [[ "$METHOD" == pag ]]; then
    run_ppo algorithm.use_kl_in_reward=False algorithm.kl_ctrl.kl_coef=0.0 \
        algorithm.norm_type=role reward_model.policy_rs=True \
        actor_rollout_ref.rollout.rollout_type=pag \
        "trainer.total_training_steps=$TOTAL_TRAINING_STEPS" \
        "trainer.val_before_train=${VAL_BEFORE_TRAIN:-True}" \
        "trainer.resume_mode=$RESUME_MODE" "$@"
else
    COMMON_ARGS+=(algorithm.norm_type=whole actor_rollout_ref.rollout.rollout_type=multiturn)
    if [[ "$SCORE_STAGE" != 2 ]]; then
        if [[ "${SAVE_FREQ:-50}" -le 0 ]]; then
            echo 'SCoRe stage 1 must save a checkpoint; require SAVE_FREQ > 0.' >&2
            exit 2
        fi
        run_ppo algorithm.use_kl_in_reward=True \
            algorithm.kl_ctrl.kl_coef_turn1=0.1 algorithm.kl_ctrl.kl_coef_turn2=0 \
            algorithm.kl_ctrl.multiturn_kl=True reward_model.policy_rs=False \
            "trainer.total_training_steps=$SCORE_STAGE1_STEPS" \
            "trainer.val_before_train=${VAL_BEFORE_TRAIN:-True}" \
            "trainer.resume_mode=$RESUME_MODE" "$@"
    fi
    if [[ "$SCORE_STAGE" != 1 ]]; then
        STAGE1_CKPT="${SCORE_STAGE1_CKPT:-$CKPTS_DIR/global_step_$SCORE_STAGE1_STEPS}"
        if [[ "$DRY_RUN" != 1 ]] && { [[ ! -d "$STAGE1_CKPT/actor" ]] || [[ ! -d "$STAGE1_CKPT/critic" ]]; }; then
            echo "Missing SCoRe stage 1 actor/critic checkpoint: $STAGE1_CKPT" >&2
            exit 1
        fi
        run_ppo algorithm.use_kl_in_reward=False \
            algorithm.kl_ctrl.kl_coef_turn1=0 algorithm.kl_ctrl.kl_coef_turn2=0 \
            algorithm.kl_ctrl.multiturn_kl=False reward_model.policy_rs=True \
            "trainer.total_training_steps=$TOTAL_TRAINING_STEPS" \
            trainer.val_before_train=False trainer.resume_mode=resume_path \
            "trainer.resume_from_path=$STAGE1_CKPT" "$@"
    fi
fi
