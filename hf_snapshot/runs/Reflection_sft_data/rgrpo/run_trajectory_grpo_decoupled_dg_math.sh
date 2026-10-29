#!/usr/bin/env bash
# Decoupled-baseline + DIFFICULTY-GATED-FLOOR trajectory GRPO (Direction 2 + DG).
#
# SEPARATE from run_trajectory_grpo_math.sh, run_trajectory_grpo_decoupled_math.sh
# and Hybrid-GRPO v2. The reward is UNCHANGED (R = R_task * R_eff); only the
# ADVANTAGE is changed at runtime via a non-invasive patch installed by the
# reward-plugin module rgrpo.core_trajectory_grpo_decoupled_dg (no verl /
# old-code edits; the legacy decoupled modules are NOT touched):
#   - Direction 2: per-prompt counterfactual baseline -- reflect trajectories
#     are scored against the SAME prompt's direct sub-group mean (mu_D), so the
#     gradient tracks "was reflecting better than NOT reflecting on THIS
#     problem". Direct sub-group stays zero-sum (ability/efficiency protected).
#   - DIFFICULTY-GATED floor on the reflect sub-group only: the floor weight is
#     the PER-PROMPT difficulty clip(mu_D,0,1), NOT a function of training step.
#     Easy prompt (mu_D~=1) -> floor on, a failed reflection clamped to floor0
#     (keeps a residual NEGATIVE -> "don't reflect on easy" muted not zeroed ->
#     no white-knight) instead of -1/sigma drowning the whole reflect gradient.
#     Hard prompt (mu_D~=0) -> floor off, a genuine rescue keeps full +1/sigma,
#     no free positive. No-direct group -> floor off (white-knight guard).
#     This decouples "reflection dies early" (legacy short time-floor) from
#     "white-knight -> direct collapse" (legacy long/persistent time-floor):
#     run_trajectory_grpo_decoupled_math.sh:107-122 documents that NO global
#     time-floor value is safe (<=100 dies, >=500/1000 collapses direct).
#
# Diagnosed on run data-5-19 (..._20260519_042140.log): WRONG_NO_REFLECT=-0.25
# only DELAYED collapse; trigger fell into 0.10-0.15 right after the time-floor
# decayed out at step ~100, and the reflect negative advantage is largely
# DESERVED (reflection fires on easy already-solved prompts and breaks them).
# DG floor stops easy-prompt failed reflection from drowning the gradient and
# preserves full credit for rare hard-prompt rescues, WITHOUT forcing reflection
# anywhere (rollout untouched -> reflection stays purely emergent). Honest
# ceiling: it halts the active killing of emergent reflection but cannot
# manufacture good reflection if the model rarely reflects on hard prompts.

set -xeo pipefail

# ===================================================================
# ISOLATION FROM A CONCURRENT RUN (e.g. run_trajectory_grpo_math.sh on
# GPUs 6,7 with RAY_TMPDIR=/ssd/data/wangruxin/ray_tr_grpo). Everything
# below guarantees this run shares NO GPU, NO Ray cluster/temp dir/
# sockets/ports, and no checkpoint/log paths with that run, so neither
# process can crash the other.
# ===================================================================
# Pin to GPU 3. PCI_BUS_ID ordering makes CUDA_VISIBLE_DEVICES indices match
# nvidia-smi exactly. GPU 3 is the user's chosen card: as of 2026-05-19 it ran
# PID 2037737 -- a DIFFERENT experiment (grpo/run_grpo_math_trajectory_aligned.sh,
# launcher PID 2033796 / trainer PID 2033804, RAY_TMPDIR=/ssd/.../rg_2033796),
# which the user will kill. It is NOT the data-5-19 run (that one is on GPU 4
# with RAY_TMPDIR=/ssd/.../ray_tr_grpo_5_19 and must keep running). The
# preflight guard below is INTENTIONALLY kept: it refuses to launch until
# GPU 3 actually has >= MIN_FREE_MIB free, so a launch racing ahead of the
# kill cannot OOM. Override CUDA_VISIBLE_DEVICES=<n> for any other card.
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
# Live preflight: refuse to launch if the chosen card is not actually free
# enough (peak need ~45-70 GB; these runs burst). Best-effort: skips the
# check only if nvidia-smi cannot be parsed. Override threshold via
# MIN_FREE_MIB, or set SKIP_GPU_PREFLIGHT=1 to bypass entirely (not advised).
if [[ "${SKIP_GPU_PREFLIGHT:-0}" != "1" ]]; then
    _free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
        -i "${CUDA_VISIBLE_DEVICES}" 2>/dev/null | head -1 | tr -dc '0-9')"
    if [[ -n "${_free_mib}" ]] && (( _free_mib < ${MIN_FREE_MIB:-45000} )); then
        echo "[ERROR] GPU ${CUDA_VISIBLE_DEVICES} has only ${_free_mib} MiB free" >&2
        echo "[ERROR] (< ${MIN_FREE_MIB:-45000}). Refusing to launch: this run peaks" >&2
        echo "[ERROR] ~45-70 GB and would OOM / crash a co-resident process." >&2
        echo "[ERROR] Pick a freer card, wait, or override MIN_FREE_MIB." >&2
        exit 1
    fi
fi
export PYTHONPATH=/ssd/data/wangruxin/Relf_grpo:${PYTHONPATH:-}
export VLLM_USE_V1=0
# Distinct Ray temp dir => own GCS/raylet/plasma sockets + session,
# auto-selected free ports, separate object store. MUST differ from EVERY
# concurrent run: run_trajectory_grpo_math.sh (/ssd/.../ray_tr_grpo), the
# data-5-19 decoupled run (/ssd/.../ray_tr_grpo_5_19) and any older
# decoupled-baseline run (/ssd/.../ray_tr_grpo_decoupled). Sharing a temp dir
# with a live cluster collides on its sockets/session and can land this run's
# actors on the wrong GPU. This dir is namespaced to the DG experiment so it
# cannot clash with any other run.
export RAY_TMPDIR="${RAY_TMPDIR:-/ssd/data/wangruxin/ray_5_19}"
# Never auto-join an existing Ray cluster (would land actors on GPUs 6,7).
# Force a fresh local cluster; keep the dashboard off (avoids the
# concurrent run's webui port 8265).
unset RAY_ADDRESS RAY_INIT_ADDRESS || true
export RAY_INCLUDE_DASHBOARD="${RAY_INCLUDE_DASHBOARD:-false}"
mkdir -p "${RAY_TMPDIR}"

if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
    unset PYTORCH_CUDA_ALLOC_CONF
fi

# Use the robust math checker from core_hybrid_grpo_v2 inside core_trajectory_grpo.
export HYBRID_V2_MATH_SCORE_BACKEND="${HYBRID_V2_MATH_SCORE_BACKEND:-process}"
export HYBRID_V2_MATH_SCORE_TIMEOUT="${HYBRID_V2_MATH_SCORE_TIMEOUT:-30}"
export TRAJ_GRPO_REWARD_PHI="${TRAJ_GRPO_REWARD_PHI:-4}"
export TRAJ_GRPO_REWARD_STYLE="${TRAJ_GRPO_REWARD_STYLE:-efficiency}"
# First reflection FREE (=1). Empirically (run 20260518_222449) free=0 made
# correct+reflect=0.9375 < direct=1.0; that 0.0625 first-reflection tax fed a
# persistently negative reflect group-relative advantage and reflection
# collapsed (trigger 0.32->0.09, accuracy/reflection 0.39->0.13). The earlier
# "don't reflect when unnecessary" rationale is now carried by the direct-wrong
# negative reward below, not by taxing the first reflection. See memory
# no-free-first-reflection (the sanctioned current decision is =1).
export TRAJ_GRPO_EFF_FREE_REFLECTIONS="${TRAJ_GRPO_EFF_FREE_REFLECTIONS:-1}"
export TRAJ_GRPO_BINARY_DIRECT_WRONG_REWARD="${TRAJ_GRPO_BINARY_DIRECT_WRONG_REWARD:--0.25}"
export TRAJ_GRPO_BINARY_WRONG_REFLECTION_PENALTY="${TRAJ_GRPO_BINARY_WRONG_REFLECTION_PENALTY:-1}"
export TRAJ_GRPO_WRONG_REFL_PENALTY_A="${TRAJ_GRPO_WRONG_REFL_PENALTY_A:-0.05}"
export TRAJ_GRPO_WRONG_REFL_PENALTY_B="${TRAJ_GRPO_WRONG_REFL_PENALTY_B:-0.15}"
export TRAJ_GRPO_WRONG_REFL_PENALTY_C="${TRAJ_GRPO_WRONG_REFL_PENALTY_C:-0.25}"
# Reflection structure stays enforced (load-bearing: continuation context is
# only [question]+[reflection block]).
export TRAJ_GRPO_REQUIRE_REFLECTION_FORMAT="${TRAJ_GRPO_REQUIRE_REFLECTION_FORMAT:-1}"
export TRAJ_GRPO_REFLECTION_FREQ_PENALTY="${TRAJ_GRPO_REFLECTION_FREQ_PENALTY:-0.0}"
export TRAJ_GRPO_WRONG_REFLECTION_FREQ_PENALTY="${TRAJ_GRPO_WRONG_REFLECTION_FREQ_PENALTY:-0.0}"
# DIRECT-WRONG NEGATIVE REWARD (=1, the main lever): "wrong & num_reflections==0"
# (i.e. solved directly but wrong) gets -((N-1)/phi)^2. This lowers mu_D, which
# under Direction 2 raises every reflect sample's relative advantage
# (r_reflect - mu_D)/sigma -- strongest exactly on hard prompts where direct is
# all-wrong (mu_D~0 -> a successful reflection goes (1-0)/s -> (1+p)/s).
# Direction 2 alone was empirically insufficient (run 20260518_222449 collapsed
# trigger 0.32->0.09). Keep N=2 -> p = (1/4)^2 = -0.0625: small, and the
# cosmetic leak = |penalty| stays +0.0625 (the script's own sanctioned ablation
# magnitude; NOT N=4 which leaks +0.5625). Stronger dial N=3 -> -0.25 only if
# -0.0625 proves too weak, watching the white-knight guardrails.
export TRAJ_GRPO_WRONG_NO_REFLECT_PENALTY="${TRAJ_GRPO_WRONG_NO_REFLECT_PENALTY:-1}"
export TRAJ_GRPO_WRONG_NO_REFLECT_N="${TRAJ_GRPO_WRONG_NO_REFLECT_N:-3}"
export TRAJ_GRPO_MASK_LAST_REFLECTION_ONLY="${TRAJ_GRPO_MASK_LAST_REFLECTION_ONLY:-0}"
# Whole generated chain trained (incl. pre-<reflection> reasoning) -- kept as
# in the prior run so "when to pivot" gets gradient.
export TRAJ_GRPO_TRAIN_FULL_CHAIN="${TRAJ_GRPO_TRAIN_FULL_CHAIN:-1}"
export TRAJ_GRPO_WRONG_REFLECTION_FULL_CHAIN_MASK="${TRAJ_GRPO_WRONG_REFLECTION_FULL_CHAIN_MASK:-0}"

# === Direction 2 + 3 knobs (consumed by rgrpo.decoupled_patch) ===
# Activate the per-prompt counterfactual baseline. When 0 the patch is a strict
# no-op (delegates to original verl compute_advantage).
export TRAJ_GRPO_DECOUPLED_BASELINE="${TRAJ_GRPO_DECOUPLED_BASELINE:-1}"
# Lower bound clamp on REFLECT advantages (only clamps negatives; never lifts
# -> no cosmetic farming). DEEPENED -0.25 -> -0.5 after run
# ..._20260519_112119 (DG, floor0=-0.25): DG averted "reflection dies" but
# OVER-protected -> white-knight slide (trigger 0.27->0.6+, n_iters->1.6+,
# direct_solve 0.73->0.4, advantage/reflection_mean POSITIVE every step while
# advantage/direct_mean==0, while overall accuracy flat and refl-vs-direct gap
# still negative = mass shifted to the worse action). -0.5 keeps a stronger
# residual NEGATIVE on easy-prompt failed reflection so the reflect net
# advantage can go negative again (relieves the trigger pull). Watch on
# restart: if trigger instead slides back toward ~0 the floor is now too deep
# (reflection dies) -> ease toward -0.35.
export TRAJ_GRPO_REFLECT_ADV_FLOOR="${TRAJ_GRPO_REFLECT_ADV_FLOOR:--0.5}"
# Floor decay horizon. 0 => floor DISABLED => pure Direction 2 (DEFAULT).
# EMPIRICAL (run 20260518_035745, decay_steps=1000): Direction 2 itself works
# (accuracy/reflection 0.33->0.70, val acc 0.62->0.70, true-rescue 0.017->0.10+)
# but a long floor muted the "don't reflect on easy" negative signal for the
# whole run (floor_w@step1~=0.999), so advantage/reflection_mean was positive
# from step 1 while advantage/direct_mean==0 -> reflection white-knighted free
# positive advantage and DIRECT COLLAPSED (trigger 0.32->0.88, direct_solve
# 0.68->0.12, n_iters->1.88). Under Direction 2 reflection already gets a
# healthy positive signal from step 1 (per-prompt mu_D on hard prompts), so the
# floor's anti-starvation rationale does NOT apply here -> keep it OFF. Only if
# a future pure-Direction-2 run shows reflection dying early, add a VERY short
# floor (~30, must be << the ~20-40 step window in which direct began eroding).
# decay_steps=1000 is falsified; do not use.
# UPDATE (run 20260518_222449): reflection DID die early (trigger 0.32->0.09 by
# ~step 110). Per the prescription above, apply the VERY short ~30-step floor
# (FLOOR=-0.25 only clamps the negative side, no white-knight). 500 muted the
# "don't reflect on easy" signal too long, same family as the falsified 1000.
export TRAJ_GRPO_REFLECT_ADV_FLOOR_DECAY_STEPS="${TRAJ_GRPO_REFLECT_ADV_FLOOR_DECAY_STEPS:-100}"
# DIFFICULTY-GATED FLOOR (=1, the core of this run). When 1 the floor weight is
# the per-prompt difficulty clip(mu_D,0,1) and the time-decay above is IGNORED:
# easy prompt (mu_D~=1) -> floor on (failed reflection clamped to FLOOR=-0.25,
# residual negative kept -> no white-knight); hard prompt (mu_D~=0) -> floor
# off (full +1/sigma for genuine rescues, no free positive); no-direct group ->
# floor off. This is the only setting that escapes the legacy time-floor's
# dead end (run_trajectory_grpo_decoupled_math.sh:107-122: no global value is
# safe). =0 reproduces the legacy time-decayed Direction-3 floor exactly.
export TRAJ_GRPO_REFLECT_FLOOR_DIFFICULTY_GATED="${TRAJ_GRPO_REFLECT_FLOOR_DIFFICULTY_GATED:-1}"

project_name="${PROJECT_NAME:-Trajectory-GRPO-Math}"
exp_name="${EXP_NAME:-trajectory-grpo-qwen2.5-math-1.5b-dg-5-19}"
resume_mode="${RESUME_MODE:-auto}"
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
# Qwen2.5-Math-1.5B is configured with a 4k context locally. Reflection GRPO
# concatenates all generated rounds into one training response, so the safe
# upper bound is max_prompt_length + max_iters * max_response_length. With
# max_iters=3 the rollout hard cap is max_iters-1 = 2 reflections.
max_response_length="${MAX_RESPONSE_LENGTH:-1024}"
max_model_len="${MAX_MODEL_LEN:-4096}"
max_iters="${MAX_ITERS:-3}"

trajectory_max_len=$((max_prompt_length + max_iters * max_response_length))
if (( trajectory_max_len > max_model_len )) && [[ "${ALLOW_LONG_TRAJECTORY:-0}" != "1" ]]; then
    echo "[ERROR] max_prompt_length + max_iters * max_response_length = ${trajectory_max_len}, exceeds MAX_MODEL_LEN=${max_model_len}."
    echo "[ERROR] For the local 4k Qwen2.5-Math-1.5B, use e.g. MAX_RESPONSE_LENGTH=1024 MAX_ITERS=3, or set a real long-context MODEL_PATH/MAX_MODEL_LEN."
    echo "[ERROR] Set ALLOW_LONG_TRAJECTORY=1 only if the actor model truly supports the longer sequence."
    exit 1
fi

# Single-GPU defaults: halved prompt batch to fit one A100-80GB.
train_prompt_bsz="${TRAIN_PROMPT_BSZ:-64}"
# Larger group stabilises the per-prompt direct mean (mu_D); 16 is too heavy
# for one GPU so default 8 here (Direction 2 still holds, mu_D just noisier).
# Raise to 16/32 with more GPUs/VRAM via N_RESP_PER_PROMPT.
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
CKPTS_DIR="${CKPTS_DIR:-${RAY_DATA_HOME}/RewardFlow/outputs/trajectory_grpo/${exp_name}}"
TRAIN_FILE="${TRAIN_FILE:-${RAY_DATA_HOME}/data/math_lighteval_rgrpo.jsonl}"
TEST_FILE="${TEST_FILE:-${RAY_DATA_HOME}/data/math500_rgrpo_fixed.jsonl}"

echo "[INFO] Resolved model paths:"
echo "[INFO]   BASE_MODEL_PATH      = ${BASE_MODEL_PATH}"
echo "[INFO]   SFT_ADAPTER_PATH     = ${SFT_ADAPTER_PATH}"
echo "[INFO]   MERGED_SFT_MODEL_PATH= ${MERGED_SFT_MODEL_PATH}"
echo "[INFO]   MODEL_PATH           = ${MODEL_PATH}"
echo "[INFO] NOTE: the reflection SFT adapter was trained on Qwen2.5-3B-Instruct;"
echo "[INFO] BASE_MODEL_PATH MUST be that exact base. Override BASE_MODEL_PATH /"
echo "[INFO] SFT_ADAPTER_PATH if these defaults do not point at your real SFT."

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    if [[ "${MODEL_PATH}" == "${MERGED_SFT_MODEL_PATH}" ]] && [[ -f "${SFT_ADAPTER_PATH}/adapter_config.json" ]]; then
        echo "[INFO] Merging SFT adapter into base model..."
        BASE_MODEL_PATH="${BASE_MODEL_PATH}" SFT_ADAPTER_PATH="${SFT_ADAPTER_PATH}" MERGED_SFT_MODEL_PATH="${MERGED_SFT_MODEL_PATH}" python3 - <<'PY'
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
    # Guard: the LoRA adapter records the base it was trained on. Merging it
    # onto a different base (e.g. Qwen2.5-3B-Instruct adapter onto
    # Qwen2.5-Math-1.5B) silently produces a broken model. Fail loudly unless
    # explicitly overridden with ALLOW_SFT_BASE_MISMATCH=1.
    import json
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
        echo "[ERROR] MODEL_PATH=${MODEL_PATH} does not contain a HuggingFace model (missing config.json)."
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
# Single-GPU: offload actor params+optimizer to CPU to free VRAM (slower but
# avoids OOM on one card). Set OFFLOAD=False with >=2 GPUs for speed.
offload="${OFFLOAD:-True}"
actor_lr="${ACTOR_LR:-1e-6}"
actor_weight_decay="${ACTOR_WEIGHT_DECAY:-0.0}"
actor_grad_clip="${ACTOR_GRAD_CLIP:-1.0}"
actor_ppo_max_token_len_per_gpu="${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-32768}"
log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-32}"
log_prob_max_token_len_per_gpu="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-65536}"
# Single-GPU colocated: vLLM must leave room for FSDP actor+ref on the same
# card. 0.45 fits 1.5B on one A100-80GB; raise toward 0.75 with >=2 GPUs.
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
