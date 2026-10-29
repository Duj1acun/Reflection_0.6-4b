#!/usr/bin/env bash
set -euo pipefail

# GPU0-only wrapper for the continuation-focused reflection GRPO ablation.
# It leaves the base launcher unchanged and only overrides env knobs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="2"

export EXP_NAME="trajectory-grpo-qwen2.5-math-1.5b-dg-cont-only-gpu0-5-20"
export RESUME_MODE="disable"
export RESUME_FROM_PATH="null"

# Train only the last reflection segment and its continuation. Earlier
# reasoning / earlier reflection rounds are masked out.
export TRAJ_GRPO_TRAIN_FULL_CHAIN="0"
export TRAJ_GRPO_MASK_LAST_REFLECTION_ONLY="1"
export TRAJ_GRPO_WRONG_REFLECTION_FULL_CHAIN_MASK="0"

# Continuation-only training already masks the earlier chain, so do not tax
# correct trajectories for using extra reflection rounds. This makes:
#   correct direct / +1 reflection / +2 reflections -> reward 1.0
export TRAJ_GRPO_USE_EFF_REWARD="0"
export TRAJ_GRPO_REFLECTION_FREQ_PENALTY="0.0"
export TRAJ_GRPO_WRONG_REFLECTION_FREQ_PENALTY="0.0"

# Lower training sampling temperature 1.0 -> 0.7 (val stays greedy via
# val_kwargs.temperature=0 in the base launcher).
export TEMPERATURE="0.7"

# Disable the reflect-advantage floor completely (both clamp modes):
#   DG=0          -> per-prompt difficulty-gated floor OFF
#   DECAY_STEPS=0 -> legacy time-decayed floor OFF
# decoupled_advantage_dg.py:157-173 then forces floor_w = 0, so the
# torch.maximum(a, floor0) clamp is never blended in. Direction-2 (per-prompt
# mu_D baseline) is still active via TRAJ_GRPO_DECOUPLED_BASELINE=1 in the
# base launcher; only the clamp is gone.
export TRAJ_GRPO_REFLECT_FLOOR_DIFFICULTY_GATED="0"
export TRAJ_GRPO_REFLECT_ADV_FLOOR_DECAY_STEPS="0"
# floor0 is irrelevant with both modes off; set very negative as a guard.
export TRAJ_GRPO_REFLECT_ADV_FLOOR="-1000.0"

# Keep this experiment isolated from the existing 5-20 run.
# Keep this short: Ray's AF_UNIX socket paths must stay under 107 bytes.
export RAY_TMPDIR="/ssd/data/wangruxin/ray_cg0"
export CKPTS_DIR="/ssd/data/wangruxin/R_grpo_sft/RewardFlow/outputs/trajectory_grpo/${EXP_NAME}"
export VAL_DUMP_DIR="/ssd/data/wangruxin/outputs/val_result/${EXP_NAME}"
export LOG_DIR="/ssd/data/wangruxin/outputs/train_logs"

# Safety gate: do not start while GPU0 has any compute process, and require
# enough free memory for the original single-GPU configuration.
MIN_FREE_MIB="${MIN_FREE_MIB:-76000}"
WAIT_INTERVAL_SECONDS="${WAIT_INTERVAL_SECONDS:-60}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-0}"  # 0 means wait indefinitely.

gpu0_compute_apps() {
    nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" \
        --query-compute-apps=pid,process_name,used_gpu_memory \
        --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d'
}

gpu0_free_mib() {
    nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" \
        --query-gpu=memory.free \
        --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -dc '0-9'
}

mkdir -p "${LOG_DIR}" "${RAY_TMPDIR}" "${CKPTS_DIR}" "${VAL_DUMP_DIR}"

echo "[INFO] Continuation-only trajectory GRPO wrapper"
echo "[INFO] EXP_NAME=${EXP_NAME}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] Waiting for GPU0 to be empty and have >= ${MIN_FREE_MIB} MiB free"

start_ts="$(date +%s)"
while true; do
    apps="$(gpu0_compute_apps || true)"
    free_mib="$(gpu0_free_mib || true)"

    if [[ -z "${apps}" && -n "${free_mib}" ]] && (( free_mib >= MIN_FREE_MIB )); then
        echo "[INFO] GPU0 ready: free_mib=${free_mib}, compute_apps=0"
        break
    fi

    now_ts="$(date +%s)"
    elapsed=$(( now_ts - start_ts ))
    if (( MAX_WAIT_SECONDS > 0 && elapsed >= MAX_WAIT_SECONDS )); then
        echo "[ERROR] Timed out waiting for GPU0 after ${elapsed}s" >&2
        echo "[ERROR] free_mib=${free_mib:-unknown}, required=${MIN_FREE_MIB}" >&2
        if [[ -n "${apps}" ]]; then
            echo "[ERROR] Current GPU0 compute apps:" >&2
            echo "${apps}" | sed 's/^/[ERROR]   /' >&2
        fi
        exit 1
    fi

    echo "[WAIT] GPU0 not ready after ${elapsed}s: free_mib=${free_mib:-unknown}, required=${MIN_FREE_MIB}"
    if [[ -n "${apps}" ]]; then
        echo "[WAIT] GPU0 compute apps:"
        echo "${apps}" | sed 's/^/[WAIT]   /'
    fi
    sleep "${WAIT_INTERVAL_SECONDS}"
done

exec bash "${SCRIPT_DIR}/run_trajectory_grpo_5_20.sh"
