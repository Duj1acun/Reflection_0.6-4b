#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SIZE="${1:?Usage: evaluate_paired_qwen3.sh 0.6b|4b preflight|smoke|run gpu [base|phase2|both]}"
MODE="${2:-preflight}"
GPU="${3:-0}"
VARIANT="${4:-both}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export HF_DATASETS_CACHE=/mnt/data/cache/huggingface
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MATH500_PATH=/mnt/data/runs/Reflection_sft_data/rl_data/math500_val.jsonl
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
OUT="${EVAL_OUT:-/mnt/data/reflection_grpo/artifacts/paired_eval_pass4_20260912}"
mkdir -p "${OUT}"
exec 9>"${OUT}/qwen3-${SIZE}-${VARIANT}.lock"
flock -n 9 || { echo "Worker already running for ${SIZE}" >&2; exit 1; }
cd "${ROOT}"
exec /home/vipuser/miniconda3/envs/wrx_env/bin/python3 -u \
    evaluation/evaluators/eval_paired_qwen3.py --size "${SIZE}" --variant "${VARIANT}" --mode "${MODE}" \
    --out "${OUT}" --seed 1234 --max-iters 2 --max-model-len 20480
