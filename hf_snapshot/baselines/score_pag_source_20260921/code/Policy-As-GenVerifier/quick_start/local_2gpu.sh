#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash quick_start/local_2gpu.sh pag [HYDRA_OVERRIDE ...]
  bash quick_start/local_2gpu.sh score [HYDRA_OVERRIDE ...]

Examples:
  CUDA_VISIBLE_DEVICES=0,1 bash quick_start/local_2gpu.sh pag
  CUDA_VISIBLE_DEVICES=0,1 bash quick_start/local_2gpu.sh score
  DRY_RUN=1 bash quick_start/local_2gpu.sh pag trainer.total_epochs=1

Environment overrides:
  PYTHON_BIN, MODEL_PATH, CKPT_ROOT, N_GPUS, TRAIN_BATCH_SIZE,
  PPO_MINI_BATCH_SIZE, MAX_TOKENS_PER_GPU, GPU_MEMORY_UTILIZATION,
  ROLLOUT_N, VAL_N, OFFLOAD, LOGGER, SCORE_STAGE, DRY_RUN

SCORE_STAGE applies only to SCoRe and accepts: both (default), 1, or 2.
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 2
fi

METHOD="${1,,}"
shift
if [[ "$METHOD" != "pag" && "$METHOD" != "score" ]]; then
    usage
    exit 2
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/mnt/data/conda_envs/pag-score/bin/python}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/models/Qwen2.5-1.5B-Instruct}"
CKPT_ROOT="${CKPT_ROOT:-/mnt/data/checkpoints/pag-score}"
N_GPUS="${N_GPUS:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-128}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.20}"
ROLLOUT_N="${ROLLOUT_N:-4}"
VAL_N="${VAL_N:-8}"
OFFLOAD="${OFFLOAD:-false}"
LOGGER="${LOGGER:-['console']}"
SCORE_STAGE="${SCORE_STAGE:-both}"
DRY_RUN="${DRY_RUN:-0}"

TRAIN_DATA="$REPO_DIR/datasets/math7500.parquet"
VAL_DATA="$REPO_DIR/datasets/math500.parquet"
PROJECT_NAME="PAG"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
    echo "Model is missing or incomplete: $MODEL_PATH" >&2
    exit 1
fi

export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/mnt/data/cache/huggingface}"
export TMPDIR="${TMPDIR:-/mnt/data/tmp/pag-score}"
export RAY_TMPDIR="${RAY_TMPDIR:-$TMPDIR/ray}"
export TOKENIZERS_PARALLELISM=true
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=0
mkdir -p "$CKPT_ROOT" "$TMPDIR" "$RAY_TMPDIR"
cd "$REPO_DIR"

COMMON_ARGS=(
    algorithm.adv_estimator=gae
    "data.train_files=['$TRAIN_DATA']"
    "data.val_files=['$VAL_DATA']"
    data.filter_overlong_prompts=true
    "data.train_batch_size=$TRAIN_BATCH_SIZE"
    data.max_prompt_length=1024
    data.max_response_length=2048
    "actor_rollout_ref.model.path=$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=true
    actor_rollout_ref.actor.use_dynamic_bsz=true
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_TOKENS_PER_GPU"
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0
    "actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE"
    "actor_rollout_ref.actor.fsdp_config.param_offload=$OFFLOAD"
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OFFLOAD"
    actor_rollout_ref.actor.use_kl_loss=false
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.clip_ratio_high=0.28
    actor_rollout_ref.actor.clip_ratio_low=0.2
    actor_rollout_ref.ref.fsdp_config.param_offload=true
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    "actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_TOKENS_PER_GPU"
    actor_rollout_ref.rollout.enforce_eager=true
    actor_rollout_ref.rollout.free_cache_engine=false
    "actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
    "actor_rollout_ref.rollout.n=$ROLLOUT_N"
    actor_rollout_ref.rollout.top_k=10000
    actor_rollout_ref.rollout.num_turns=2
    "actor_rollout_ref.rollout.val_kwargs.n=$VAL_N"
    actor_rollout_ref.rollout.val_kwargs.do_sample=true
    actor_rollout_ref.rollout.val_kwargs.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6
    actor_rollout_ref.rollout.val_kwargs.num_turns=2
    reward_model.rs_coef=1.0
    critic.optim.lr=2e-6
    critic.use_dynamic_bsz=true
    critic.model.use_remove_padding=true
    "critic.model.path=$MODEL_PATH"
    "critic.model.fsdp_config.param_offload=$OFFLOAD"
    "critic.model.fsdp_config.optimizer_offload=$OFFLOAD"
    "critic.ppo_max_token_len_per_gpu=$MAX_TOKENS_PER_GPU"
    "trainer.logger=$LOGGER"
    "trainer.project_name=$PROJECT_NAME"
    "trainer.n_gpus_per_node=$N_GPUS"
    trainer.nnodes=1
    trainer.save_freq=10
    trainer.test_freq=10
    trainer.max_actor_ckpt_to_keep=1
    trainer.max_critic_ckpt_to_keep=1
    trainer.resume_mode=auto
    trainer.log_val_generations=2
)

run_ppo() {
    local args=("$PYTHON_BIN" -m verl.trainer.main_ppo "$@")
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '%q ' "${args[@]}"
        printf '\n'
    else
        "${args[@]}"
    fi
}

if [[ "$METHOD" == "pag" ]]; then
    EXPERIMENT_NAME="qwen1p5b_pag_2gpu"
    run_ppo \
        "${COMMON_ARGS[@]}" \
        algorithm.use_kl_in_reward=false \
        algorithm.norm_type=role \
        reward_model.policy_rs=true \
        actor_rollout_ref.rollout.rollout_type=pag \
        "trainer.experiment_name=$EXPERIMENT_NAME" \
        trainer.total_epochs=40 \
        "trainer.default_local_dir=$CKPT_ROOT/$PROJECT_NAME/$EXPERIMENT_NAME" \
        trainer.val_before_train=true \
        "$@"
else
    EXPERIMENT_NAME="qwen1p5b_score_2gpu"
    SCORE_DIR="$CKPT_ROOT/$PROJECT_NAME/$EXPERIMENT_NAME"

    if [[ "$SCORE_STAGE" != "both" && "$SCORE_STAGE" != "1" && "$SCORE_STAGE" != "2" ]]; then
        echo "Invalid SCORE_STAGE: $SCORE_STAGE (expected both, 1, or 2)" >&2
        exit 2
    fi

    if [[ "$SCORE_STAGE" == "both" || "$SCORE_STAGE" == "1" ]]; then
        echo "SCoRe stage 1/2: decouple turn 1 and turn 2"
        run_ppo \
            "${COMMON_ARGS[@]}" \
            algorithm.use_kl_in_reward=true \
            algorithm.kl_ctrl.kl_coef_turn1=0.1 \
            algorithm.kl_ctrl.kl_coef_turn2=0 \
            algorithm.kl_ctrl.multiturn_kl=true \
            algorithm.norm_type=whole \
            reward_model.policy_rs=false \
            actor_rollout_ref.rollout.rollout_type=multiturn \
            "trainer.experiment_name=$EXPERIMENT_NAME" \
            trainer.total_epochs=20 \
            "trainer.default_local_dir=$SCORE_DIR" \
            trainer.val_before_train=true \
            "$@"
    fi

    if [[ "$SCORE_STAGE" == "both" || "$SCORE_STAGE" == "2" ]]; then
        echo "SCoRe stage 2/2: jointly train both turns (auto-resumes stage 1)"
        run_ppo \
            "${COMMON_ARGS[@]}" \
            algorithm.use_kl_in_reward=false \
            algorithm.kl_ctrl.kl_coef_turn1=0 \
            algorithm.kl_ctrl.kl_coef_turn2=0 \
            algorithm.kl_ctrl.multiturn_kl=false \
            algorithm.norm_type=whole \
            reward_model.policy_rs=true \
            actor_rollout_ref.rollout.rollout_type=multiturn \
            "trainer.experiment_name=$EXPERIMENT_NAME" \
            trainer.total_epochs=40 \
            "trainer.default_local_dir=$SCORE_DIR" \
            trainer.val_before_train=false \
            "$@"
    fi
fi
