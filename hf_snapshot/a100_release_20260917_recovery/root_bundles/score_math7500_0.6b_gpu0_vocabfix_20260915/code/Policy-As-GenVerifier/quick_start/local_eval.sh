#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: bash quick_start/local_eval.sh {pag|score} /path/to/global_step_N [HYDRA_OVERRIDE ...]" >&2
    exit 2
fi

METHOD="${1,,}"
RESUME_PATH="$2"
shift 2
if [[ "$METHOD" != "pag" && "$METHOD" != "score" ]]; then
    echo "Method must be 'pag' or 'score'." >&2
    exit 2
fi
if [[ "$(basename "$RESUME_PATH")" != global_step_* || ! -d "$RESUME_PATH/actor" ]]; then
    echo "Expected a sharded checkpoint directory ending in global_step_N: $RESUME_PATH" >&2
    exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/mnt/data/conda_envs/pag-score/bin/python}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/models/Qwen2.5-1.5B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/data/checkpoints/pag-score/eval}"
N_GPUS="${N_GPUS:-2}"
VAL_N="${VAL_N:-32}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.45}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-8192}"

if [[ "$METHOD" == "pag" ]]; then
    ROLLOUT_TYPE=pag
else
    ROLLOUT_TYPE=multiturn
fi

EXPERIMENT_NAME="eval_${METHOD}_qwen1p5b"
OUTPUT_DIR="$OUTPUT_ROOT/$EXPERIMENT_NAME"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/mnt/data/cache/huggingface}"
export TMPDIR="${TMPDIR:-/mnt/data/tmp/pag-score}"
export RAY_TMPDIR="${RAY_TMPDIR:-$TMPDIR/ray}"
export VLLM_USE_V1=0
export HYDRA_FULL_ERROR=1
mkdir -p "$OUTPUT_DIR" "$TMPDIR" "$RAY_TMPDIR"
cd "$REPO_DIR"

"$PYTHON_BIN" -m verl.trainer.main_ppo \
    trainer.resume_mode=resume_path \
    "trainer.resume_from_path=$RESUME_PATH" \
    trainer.val_before_train=true \
    trainer.val_only=true \
    algorithm.adv_estimator=grpo \
    "data.train_files=['$REPO_DIR/datasets/dapo17k.parquet']" \
    "data.val_files=['$REPO_DIR/datasets/math500.parquet']" \
    data.train_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=2028 \
    "actor_rollout_ref.model.path=$MODEL_PATH" \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    "actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_TOKENS_PER_GPU" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    "actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION" \
    actor_rollout_ref.rollout.num_turns=2 \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.free_cache_engine=false \
    "actor_rollout_ref.rollout.rollout_type=$ROLLOUT_TYPE" \
    "actor_rollout_ref.rollout.val_kwargs.n=$VAL_N" \
    actor_rollout_ref.rollout.val_kwargs.num_turns=2 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    trainer.project_name=PAG \
    "trainer.experiment_name=$EXPERIMENT_NAME" \
    "trainer.logger=['console']" \
    "trainer.n_gpus_per_node=$N_GPUS" \
    trainer.nnodes=1 \
    trainer.save_validation_results=true \
    "trainer.validation_results_path=$OUTPUT_DIR/validation_results.json" \
    "trainer.default_local_dir=$OUTPUT_DIR" \
    "$@"
