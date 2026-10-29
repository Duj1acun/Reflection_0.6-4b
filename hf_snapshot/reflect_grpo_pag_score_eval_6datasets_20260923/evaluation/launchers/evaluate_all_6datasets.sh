#!/usr/bin/env bash
# Re-run the paper evaluation for all five sampling seeds.
# Requires five GPU ids by default; override GPUS and EVAL_SEEDS together.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -a
    source "${PROJECT_ROOT}/.env"
    set +a
fi
REPO="${REPO:-${PROJECT_ROOT}}"
EVAL_ROOT="${REPO}/evaluation"
PY="${PYTHON:-python3}"
CONVERTED="${CONVERTED_DIR:-${REPO}/artifacts/models}"
PAPER_RESULTS="${PAPER_RESULTS_DIR:-${REPO}/artifacts/evaluation/seed_1234}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO}/artifacts/evaluation}"
LOG_DIR="${RESULTS_ROOT}/logs"

DATASETS="${EVAL_DATASETS:-math500,amc23,gsm8k,logiqa,gpqa,mmlu_pro}"
N="${EVAL_N:-32}"
TEMP="${EVAL_TEMP:-1.0}"
TOP_P="${EVAL_TOP_P:-1.0}"
MAX_ITERS="${EVAL_MAX_ITERS:-3}"
USE_LARGE_VLLM_BATCH="${USE_LARGE_VLLM_BATCH:-1}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-0}"
if [[ -n "${EVAL_MAX_NUM_BATCHED_TOKENS+x}" ]]; then
    MAX_NUM_BATCHED_TOKENS="${EVAL_MAX_NUM_BATCHED_TOKENS}"
elif [[ "${USE_LARGE_VLLM_BATCH}" == "1" ]]; then
    MAX_NUM_BATCHED_TOKENS="131072"
else
    MAX_NUM_BATCHED_TOKENS=""
fi
if [[ -n "${EVAL_MAX_NUM_SEQS+x}" ]]; then
    MAX_NUM_SEQS="${EVAL_MAX_NUM_SEQS}"
elif [[ "${USE_LARGE_VLLM_BATCH}" == "1" ]]; then
    MAX_NUM_SEQS="512"
else
    MAX_NUM_SEQS=""
fi
NEW_SEEDS_STR="${EVAL_SEEDS:-1234 2026 3407 4519 8675}"
ORIG_SEED="${ORIG_SEED:-1234}"
GPUS_STR="${GPUS:-0,1,2,3,4}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_CONVERT="${SKIP_CONVERT:-1}"
SKIP_GPU_PREFLIGHT="${SKIP_GPU_PREFLIGHT:-0}"
MIN_FREE_MIB="${MIN_FREE_MIB:-35000}"

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export HYBRID_V2_MATH_SCORE_TIMEOUT="${HYBRID_V2_MATH_SCORE_TIMEOUT:-5}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${REPO}/data/hf_datasets_cache}"
export HF_HOME="${HF_HOME:-${REPO}/.cache/huggingface}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

IFS=',' read -r -a GPUS <<< "${GPUS_STR}"
read -r -a NEW_SEEDS <<< "${NEW_SEEDS_STR}"

if [[ "${#GPUS[@]}" -ne "${#NEW_SEEDS[@]}" ]]; then
    echo "[ERROR] GPUS (${GPUS_STR}) count must equal NEW_SEEDS (${NEW_SEEDS_STR}) count." >&2
    exit 1
fi

mkdir -p "${RESULTS_ROOT}" "${LOG_DIR}"

log() {
    printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"
}

model_path() {
    case "$1" in
        base)
            printf '%s\n' "${BASE_MODEL_PATH:-${REPO}/models/Qwen2.5-Math-1.5B}"
            ;;
        sft)         printf '%s\n' "${SFT_MODEL_PATH:-${REPO}/artifacts/sft/phase2_merged_hf}" ;;
        grpo)        printf '%s\n' "${CONVERTED}/grpo_step400" ;;
        refl_dg520)  printf '%s\n' "${CONVERTED}/refl_dg_step200" ;;
        *)           echo "[ERROR] unknown model: $1" >&2; return 1 ;;
    esac
}

prompt_kind() {
    case "$1" in
        base|grpo)       printf '%s\n' "qwen_math" ;;
        sft|refl_dg520)  printf '%s\n' "reflection" ;;
        *)               echo "[ERROR] unknown model: $1" >&2; return 1 ;;
    esac
}

gpu_mem_util() {
    case "$1" in
        base|grpo)       printf '%s\n' "${QWEN_GPU_MEM_UTIL:-0.90}" ;;
        sft|refl_dg520)  printf '%s\n' "${REFL_GPU_MEM_UTIL:-0.90}" ;;
        *)               echo "[ERROR] unknown model: $1" >&2; return 1 ;;
    esac
}

extra_model_args() {
    case "$1" in
        base|grpo)       printf '%s\n' "--max-tokens 3072" ;;
        sft|refl_dg520)  printf '%s\n' "" ;;
        *)               echo "[ERROR] unknown model: $1" >&2; return 1 ;;
    esac
}

check_model_paths() {
    local m path
    for m in base sft grpo refl_dg520; do
        path="$(model_path "${m}")"
        if [[ ! -f "${path}/config.json" ]]; then
            echo "[ERROR] ${m} missing config.json at ${path}" >&2
            exit 1
        fi
    done
}

check_gpus() {
    [[ "${SKIP_GPU_PREFLIGHT}" == "1" ]] && return 0
    local gpu free
    for gpu in "${GPUS[@]}"; do
        free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
            -i "${gpu}" 2>/dev/null | head -1 | tr -dc '0-9' || true)"
        if [[ -z "${free}" ]]; then
            log "[warn] could not read free memory for GPU ${gpu}; continuing"
            continue
        fi
        if (( free < MIN_FREE_MIB )); then
            echo "[ERROR] GPU ${gpu} has only ${free} MiB free (< ${MIN_FREE_MIB})." >&2
            echo "[ERROR] Set SKIP_GPU_PREFLIGHT=1 to override intentionally." >&2
            exit 1
        fi
        log "GPU ${gpu}: ${free} MiB free"
    done
}

run_one_model() {
    local gpu="$1" seed="$2" model="$3" out_dir="$4"
    local path kind mem extra_str
    path="$(model_path "${model}")"
    kind="$(prompt_kind "${model}")"
    mem="$(gpu_mem_util "${model}")"
    extra_str="$(extra_model_args "${model}")"
    # Skip only when all expected per-dataset summaries already exist.
    if [[ "${FORCE}" != "1" ]]; then
        local all_done=1 ds_file ds
        IFS=',' read -r -a _datasets <<< "${DATASETS}"
        for ds in "${_datasets[@]}"; do
            ds_file="${out_dir}/${model}__${ds}.summary.json"
            [[ -f "${ds_file}" ]] || all_done=0
        done
        if [[ "${all_done}" == "1" ]]; then
            log "GPU${gpu} seed=${seed} ${model}: skip existing summaries"
            return 0
        fi
    fi

    local cmd=(
        "${PY}" "${EVAL_ROOT}/evaluators/eval_reflection.py"
        --model-path "${path}"
        --model-name "${model}"
        --prompt-kind "${kind}"
        --datasets "${DATASETS}"
        --n "${N}"
        --temp "${TEMP}"
        --top-p "${TOP_P}"
        --max-iters "${MAX_ITERS}"
        --gpu-mem-util "${mem}"
        --seed "${seed}"
        --out "${out_dir}"
    )
    if [[ -n "${MAX_NUM_BATCHED_TOKENS}" && "${MAX_NUM_BATCHED_TOKENS}" != "none" ]]; then
        cmd+=( --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" )
    fi
    if [[ -n "${MAX_NUM_SEQS}" && "${MAX_NUM_SEQS}" != "none" ]]; then
        cmd+=( --max-num-seqs "${MAX_NUM_SEQS}" )
    fi
    if [[ "${ENABLE_PREFIX_CACHING}" == "1" ]]; then
        cmd+=( --enable-prefix-caching )
    fi
    if [[ "${ENABLE_CHUNKED_PREFILL}" == "1" ]]; then
        cmd+=( --enable-chunked-prefill )
    fi
    if [[ -n "${extra_str}" ]]; then
        # shellcheck disable=SC2206
        local extra=( ${extra_str} )
        cmd+=( "${extra[@]}" )
    fi

    if [[ "${DRY_RUN}" == "1" ]]; then
        printf '[dry-run][GPU%s][seed=%s][%s] ' "${gpu}" "${seed}" "${model}"
        printf '%q ' CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}"
        printf '\n'
        return 0
    fi

    log "GPU${gpu} seed=${seed}: START ${model}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}"
    log "GPU${gpu} seed=${seed}: DONE ${model}"
}

run_seed_worker() {
    local gpu="$1" seed="$2"
    local out_dir="${RESULTS_ROOT}/seed_${seed}"
    if [[ "${DRY_RUN}" != "1" ]]; then
        mkdir -p "${out_dir}"
    fi
    log "GPU${gpu} seed=${seed}: worker output -> ${out_dir}"

    # Heavier reflection models first, so if a worker fails it fails early.
    run_one_model "${gpu}" "${seed}" refl_dg520 "${out_dir}"
    run_one_model "${gpu}" "${seed}" sft        "${out_dir}"
    run_one_model "${gpu}" "${seed}" grpo       "${out_dir}"
    run_one_model "${gpu}" "${seed}" base       "${out_dir}"
}

aggregate() {
    "${PY}" - "${PAPER_RESULTS}" "${RESULTS_ROOT}" "${ORIG_SEED}" "${NEW_SEEDS_STR}" "${DATASETS}" <<'PY'
import json
import os
import statistics
import sys

paper_results, results_root, orig_seed, new_seeds_str, datasets_csv = sys.argv[1:]
models = ["base", "sft", "grpo", "refl_dg520"]
datasets = [d for d in datasets_csv.split(",") if d]
seeds = new_seeds_str.split()
metrics = ["mean_at_n", "pass_at_n", "maj_at_n", "reflection_trigger_rate", "avg_n_iters"]

def source_dir(seed: str) -> str:
    return os.path.join(results_root, f"seed_{seed}")

def load_summary(seed: str, model: str, dataset: str):
    path = os.path.join(source_dir(seed), f"{model}__{dataset}.summary.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)

def stat(values):
    if not values:
        return {"n": 0, "mean": None, "std": None}
    if len(values) == 1:
        return {"n": 1, "mean": values[0], "std": 0.0}
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "std": statistics.stdev(values),
    }

def fmt(values, scale=100.0):
    if len(values) != len(seeds):
        return f"MISSING {len(values)}/{len(seeds)}"
    s = stat(values)
    return f"{s['mean'] * scale:.2f} +/- {s['std'] * scale:.2f}"

all_summaries = {}
missing = []
for seed in seeds:
    for model in models:
        for dataset in datasets:
            summary = load_summary(seed, model, dataset)
            if summary is None:
                missing.append({"seed": seed, "model": model, "dataset": dataset})
            else:
                all_summaries[(seed, model, dataset)] = summary

tables = {}
for metric in metrics:
    tables[metric] = {}
    for model in models:
        tables[metric][model] = {}
        for dataset in datasets:
            vals = [
                all_summaries[(seed, model, dataset)][metric]
                for seed in seeds
                if (seed, model, dataset) in all_summaries
                   and all_summaries[(seed, model, dataset)].get(metric) is not None
            ]
            tables[metric][model][dataset] = stat(vals)

overall = {}
for metric in ["mean_at_n", "pass_at_n", "maj_at_n"]:
    overall[metric] = {}
    for model in models:
        vals = []
        for seed in seeds:
            seed_vals = []
            ok = True
            for dataset in datasets:
                s = all_summaries.get((seed, model, dataset))
                if s is None or s.get(metric) is None:
                    ok = False
                    break
                seed_vals.append(s[metric])
            if ok:
                vals.append(stat(seed_vals)["mean"])
        overall[metric][model] = stat(vals)

deltas = {}
for metric in ["mean_at_n", "pass_at_n", "maj_at_n"]:
    vals = []
    for seed in seeds:
        refl_vals = []
        grpo_vals = []
        ok = True
        for dataset in datasets:
            refl = all_summaries.get((seed, "refl_dg520", dataset))
            grpo = all_summaries.get((seed, "grpo", dataset))
            if refl is None or grpo is None:
                ok = False
                break
            refl_vals.append(refl[metric])
            grpo_vals.append(grpo[metric])
        if ok:
            vals.append(stat(refl_vals)["mean"] - stat(grpo_vals)["mean"])
    deltas[metric] = stat(vals)

summary_json = {
    "seeds": seeds,
    "datasets": datasets,
    "models": models,
    "metrics": tables,
    "overall": overall,
    "deltas_refl_dg520_minus_grpo": deltas,
    "missing": missing,
}
json_path = os.path.join(results_root, "summary_5seed.json")
with open(json_path, "w") as f:
    json.dump(summary_json, f, indent=2)

def fmt_stat_obj(obj, scale=100.0):
    if obj["n"] != len(seeds):
        return f"MISSING {obj['n']}/{len(seeds)}"
    return f"{obj['mean'] * scale:.2f} +/- {obj['std'] * scale:.2f}"

lines = []
lines.append("# 5-Seed Evaluation Summary")
lines.append("")
lines.append(f"Seeds: {', '.join(seeds)}")
lines.append(f"Datasets: {', '.join(datasets)}")
lines.append("Models: base, sft, grpo, refl_dg520")
lines.append("Accuracy/trigger values are mean +/- sample std across seeds, in percentage points; avg_n_iters is reported in raw iteration count.")
lines.append("")
if missing:
    lines.append("## Missing Results")
    lines.append("")
    lines.append("| seed | model | dataset |")
    lines.append("|---|---|---|")
    for m in missing:
        lines.append(f"| {m['seed']} | {m['model']} | {m['dataset']} |")
    lines.append("")
else:
    lines.append("No missing results.")
    lines.append("")

for metric in ["mean_at_n", "pass_at_n", "maj_at_n"]:
    lines.append(f"## {metric}")
    lines.append("")
    lines.append("| model | " + " | ".join(datasets) + " |")
    lines.append("|" + "---|" * (len(datasets) + 1))
    for model in models:
        cells = [fmt_stat_obj(tables[metric][model][dataset]) for dataset in datasets]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    lines.append("")

lines.append("## Overall Across 5 Datasets")
lines.append("")
lines.append("| metric | base | sft | grpo | refl_dg520 | refl_dg520 - grpo |")
lines.append("|---|---|---|---|---|---|")
for metric in ["mean_at_n", "pass_at_n", "maj_at_n"]:
    cells = [fmt_stat_obj(overall[metric][model]) for model in models]
    delta = fmt_stat_obj(deltas[metric])
    lines.append(f"| {metric} | " + " | ".join(cells) + f" | {delta} |")
lines.append("")

lines.append("## Reflection Stats")
lines.append("")
for metric in ["reflection_trigger_rate", "avg_n_iters"]:
    scale = 100.0 if metric == "reflection_trigger_rate" else 1.0
    lines.append(f"### {metric}")
    lines.append("")
    lines.append("| model | " + " | ".join(datasets) + " |")
    lines.append("|" + "---|" * (len(datasets) + 1))
    for model in models:
        cells = [fmt_stat_obj(tables[metric][model][dataset], scale=scale) for dataset in datasets]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    lines.append("")

md_path = os.path.join(results_root, "summary_5seed.md")
with open(md_path, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"[aggregate] wrote {md_path}")
print(f"[aggregate] wrote {json_path}")
if missing:
    print(f"[aggregate] WARNING: {len(missing)} missing summaries")
PY
}

log "config: GPUS=${GPUS_STR} NEW_SEEDS=${NEW_SEEDS_STR} DATASETS=${DATASETS} N=${N} TEMP=${TEMP} max_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-default} max_num_seqs=${MAX_NUM_SEQS:-default} prefix_cache=${ENABLE_PREFIX_CACHING} chunked_prefill=${ENABLE_CHUNKED_PREFILL}"

if [[ "${DRY_RUN}" == "1" ]]; then
    log "DRY_RUN=1: not converting checkpoints or launching jobs"
else
    if [[ "${SKIP_CONVERT}" != "1" ]]; then
        log "converting checkpoints if needed"
        echo "[ERROR] Convert FSDP checkpoints first with scripts/checkpoints/model_merger.py."; exit 1
    fi
    check_model_paths
    check_gpus
fi

pids=()
for i in "${!NEW_SEEDS[@]}"; do
    seed="${NEW_SEEDS[$i]}"
    gpu="${GPUS[$i]}"
    worker_log="${LOG_DIR}/seed_${seed}_gpu${gpu}.log"
    if [[ "${DRY_RUN}" == "1" ]]; then
        run_seed_worker "${gpu}" "${seed}"
    else
        log "launch seed=${seed} on GPU${gpu}; log=${worker_log}"
        (
            set -euo pipefail
            run_seed_worker "${gpu}" "${seed}"
        ) >"${worker_log}" 2>&1 &
        pids+=("$!")
    fi
done

if [[ "${DRY_RUN}" == "1" ]]; then
    log "dry run complete"
    exit 0
fi

failed=0
for pid in "${pids[@]}"; do
    if wait "${pid}"; then
        log "worker pid=${pid} OK"
    else
        rc=$?
        log "worker pid=${pid} FAILED rc=${rc}"
        failed=$((failed + 1))
    fi
done

if (( failed > 0 )); then
    echo "[ERROR] ${failed} worker(s) failed. Check ${LOG_DIR}." >&2
    exit 1
fi

log "all workers completed; aggregating"
aggregate
log "done"
