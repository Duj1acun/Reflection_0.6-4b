#!/usr/bin/env bash
# Default action validates all data; formal training requires an explicit action.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
project_dir="$PWD"
model_key="${1:-}"
action="${2:-preflight}"
case "$model_key" in
  0.6b|qwen3-0.6b) model_key=qwen3-0.6b; model_name=Qwen3-0.6B; default_gpu=0 ;;
  4b|qwen3-4b) model_key=qwen3-4b; model_name=Qwen3-4B; default_gpu=1 ;;
  *) echo 'Usage: bash training/sft/train_paired_sft.sh 0.6b|4b [preflight|smoke|train|phase1|phase2] [gpu_index]' >&2; exit 2 ;;
esac
case "$action" in preflight|smoke|train|phase1|phase2) ;; *) echo 'Invalid action' >&2; exit 2 ;; esac
python_bin=/home/vipuser/miniconda3/envs/wrx_env/bin/python3
export PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${3:-$default_gpu}"
export MODEL_ID="/mnt/data/models/$model_name"
export OUTPUT_ROOT="/mnt/data/reflection_grpo/artifacts/paired_sft_20260912/$model_key"
data_dir="$project_dir/outputs/paired_reflection_perfect_sft_20260912/$model_key"
source "$data_dir/data.env"
export PHASE1_CKPT="$OUTPUT_ROOT/phase1/final"
export SEED=42 LORA_R=16 LORA_ALPHA=32 LORA_TARGETS=all
export PER_DEVICE_TRAIN_BATCH_SIZE=1 PER_DEVICE_EVAL_BATCH_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=16 GRADIENT_CHECKPOINTING=true
export ATTN_IMPLEMENTATION=flash_attention_2 WARMUP_RATIO=0.05
export ENFORCE_PAPER_MIX=1 REPLAY_RATIO=0.3
export COMPLIANCE_SAMPLES=20 COMPLIANCE_MAX_NEW_TOKENS=768
mkdir -p "$OUTPUT_ROOT"
exec 9>"$OUTPUT_ROOT/.launch.lock"
flock -n 9 || { echo 'Another process is using this model run directory' >&2; exit 2; }
"$python_bin" - "$project_dir/outputs/paired_reflection_perfect_sft_20260912/manifest.json" "$model_key" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
m=json.loads(Path(sys.argv[1]).read_text())['models'][sys.argv[2]]
for env in ['DATA_PATH','PERFECT_PATH']:
    p=Path(os.environ[env])
    if hashlib.sha256(p.read_bytes()).hexdigest()!=m['output_sha256'][p.name]:
        raise SystemExit('Dataset fingerprint mismatch: '+str(p))
print('Dataset fingerprints verified; model='+os.environ['MODEL_ID'])
PY
if [[ "$action" == smoke ]]; then
  "$python_bin" training/sft/smoke_paired_sft.py 2>&1 | tee "$OUTPUT_ROOT/gpu_smoke.log"
  exit
fi
run_stage() {
  local stage="$1"
  export OUTPUT_DIR="$OUTPUT_ROOT/phase$stage"
  export LOG_DIR="$OUTPUT_ROOT/logs/phase$stage"
  mkdir -p "$LOG_DIR"
  if [[ "$stage" == 1 ]]; then
    export NUM_TRAIN_EPOCHS=3 LEARNING_RATE=2e-4 SAVE_STEPS=100 EVAL_STEPS=100
  else
    export NUM_TRAIN_EPOCHS=2 LEARNING_RATE=5e-5 SAVE_STEPS=100 EVAL_STEPS=100
  fi
  local opts=()
  if [[ "$action" == preflight ]]; then
    opts+=(--dry-run)
  elif [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
    [[ "$action" != train ]] || { echo 'Resume requires phase1 or phase2 action' >&2; exit 2; }
    opts+=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
  elif [[ -d "$OUTPUT_DIR/final" ]] || compgen -G "$OUTPUT_DIR/checkpoint-*" >/dev/null; then
    echo "Existing weights under $OUTPUT_DIR; use explicit RESUME_FROM_CHECKPOINT with phase1/phase2" >&2
    exit 2
  fi
  "$python_bin" - "$stage" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
keys=['MODEL_ID','DATA_PATH','PERFECT_PATH','PHASE1_CKPT','MAX_LEN','SEED',
 'NUM_TRAIN_EPOCHS','LEARNING_RATE','PER_DEVICE_TRAIN_BATCH_SIZE','PER_DEVICE_EVAL_BATCH_SIZE',
 'GRADIENT_ACCUMULATION_STEPS','GRADIENT_CHECKPOINTING','ATTN_IMPLEMENTATION',
 'ENFORCE_PAPER_MIX','LORA_R','LORA_ALPHA','LORA_TARGETS','PERFECT_OVERSAMPLE',
 'CHAT_TEMPLATE_ENABLE_THINKING','SAVE_STEPS','EVAL_STEPS','CUDA_VISIBLE_DEVICES']
cfg={k:os.environ.get(k) for k in keys}
cfg['stage']=sys.argv[1]
cfg['code_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
 [Path('training/sft/train_phase'+sys.argv[1]+'.py'),Path('rgrpo/sft_training.py')]}
out=Path(os.environ['LOG_DIR'])/'resolved_config.json'
out.write_text(json.dumps(cfg,indent=2))
PY
  "$python_bin" "training/sft/train_phase$stage.py" "${opts[@]}" 2>&1 | tee -a "$LOG_DIR/$action.log"
}
case "$action" in
  preflight|train) run_stage 1; run_stage 2 ;;
  phase1) run_stage 1 ;;
  phase2) run_stage 2 ;;
esac
