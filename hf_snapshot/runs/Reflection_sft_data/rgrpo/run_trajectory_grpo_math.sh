#!/usr/bin/env bash
# Paper-style trajectory GRPO for reflection reasoning.
#
# This script is intentionally separate from Hybrid-GRPO v2. It aligns the RL
# objective with InftyThink+:
#   - one complete trajectory per rollout sample
#   - standard GRPO over G trajectories from the same prompt
#   - scalar reward R = R_task * R_eff
#   - token-level loss averaging
#
# Defaults mirror the paper where feasible:
#   train_batch_size=128 prompts, rollout.n=8, total_training_steps=1000,
#   ppo_mini_batch_size=64, lr=1e-6, temperature=1.0, top_p=1.0.

set -xeo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
export PYTHONPATH=/ssd/data/wangruxin/Relf_grpo:${PYTHONPATH:-}
export VLLM_USE_V1=0
export RAY_TMPDIR="${RAY_TMPDIR:-/ssd/data/wangruxin/ray_tr_grpo}"

if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
    unset PYTORCH_CUDA_ALLOC_CONF
fi

# Use the robust math checker from core_hybrid_grpo_v2 inside core_trajectory_grpo.
export HYBRID_V2_MATH_SCORE_BACKEND="${HYBRID_V2_MATH_SCORE_BACKEND:-process}"
export HYBRID_V2_MATH_SCORE_TIMEOUT="${HYBRID_V2_MATH_SCORE_TIMEOUT:-30}"
export TRAJ_GRPO_REWARD_PHI="${TRAJ_GRPO_REWARD_PHI:-4}"
# efficiency style applies R_eff to *correct* trajectories, so the GRPO
# group-relative advantage suppresses over-reflection per problem difficulty
# (the paper-native mechanism). binary style flat-rewards correct=1.0 and
# cannot do this. See plan 4.2/4.3.
export TRAJ_GRPO_REWARD_STYLE="${TRAJ_GRPO_REWARD_STYLE:-efficiency}"
# Reflections the efficiency reward charges nothing for. Set to 1 per the
# user's directive: the FIRST reflection is free, the 2nd+ get the quadratic
# frequency penalty 1-((n_iters-1-free)/phi)^2. Rationale for this run: an
# earlier full-MATH run with free=0 collapsed reflection to ~0% because
# reflect was a dominated action and the small first-reflection tax fed the
# negative group-relative advantage; making the first reflection free removes
# that tax. KNOWN RISK (was the original reason this was 0): on easy problems
# direct(1.0) == reflect-once(1.0) gives zero GRPO advantage, so gratuitous
# single reflections are no longer suppressed by the efficiency term alone --
# adaptive timing now leans entirely on the under-reflection penalty +
# full-chain credit. Watch the conditional trigger rate / over-reflection
# metrics; set =0 to recover the paper-original 1-((n_iters-1)/phi)^2.
export TRAJ_GRPO_EFF_FREE_REFLECTIONS="${TRAJ_GRPO_EFF_FREE_REFLECTIONS:-1}"
export TRAJ_GRPO_BINARY_DIRECT_WRONG_REWARD="${TRAJ_GRPO_BINARY_DIRECT_WRONG_REWARD:--0.25}"
export TRAJ_GRPO_BINARY_WRONG_REFLECTION_PENALTY="${TRAJ_GRPO_BINARY_WRONG_REFLECTION_PENALTY:-1}"
export TRAJ_GRPO_WRONG_REFL_PENALTY_A="${TRAJ_GRPO_WRONG_REFL_PENALTY_A:-0.05}"
export TRAJ_GRPO_WRONG_REFL_PENALTY_B="${TRAJ_GRPO_WRONG_REFL_PENALTY_B:-0.15}"
export TRAJ_GRPO_WRONG_REFL_PENALTY_C="${TRAJ_GRPO_WRONG_REFL_PENALTY_C:-0.25}"
# Enforce reflection structure. The continuation context is ONLY
# [question]+[reflection block] (pre-reflection chain is dropped), so the
# reflection must be a self-contained continuation seed: "So far I have X"
# (state to carry) + "However Y" (the pivot). This is load-bearing, not
# cosmetic; off-template degenerate reflections wreck continuation and game the
# under-reflection penalty. Paper analog: invalid format terminates the rollout.
# NOTE: currently a HARD gate (correct+malformed-reflection -> score 0). Soften
# to a graded multiplier next (see plan 4.4) so good-but-rephrased reflections
# are not equated with task failure.
export TRAJ_GRPO_REQUIRE_REFLECTION_FORMAT="${TRAJ_GRPO_REQUIRE_REFLECTION_FORMAT:-1}"
export TRAJ_GRPO_REFLECTION_FREQ_PENALTY="${TRAJ_GRPO_REFLECTION_FREQ_PENALTY:-0.0}"
export TRAJ_GRPO_WRONG_REFLECTION_FREQ_PENALTY="${TRAJ_GRPO_WRONG_REFLECTION_FREQ_PENALTY:-0.0}"
# under-reflection (wrong AND never reflected) penalty = -((N-1)/phi)^2.
# This is the PRIMARY knob for lifting the (too low ~14%) trigger rate, but a
# large |penalty| also fuels "insurance"/cosmetic reflection (wrong+reflect=0.0
# regardless of quality, so the bigger the penalty the bigger the free delta).
# N=4 -> -0.5625 is the default: used as the on-policy SUPPLY knob to keep
# reflection samples alive (N=2 -> -0.0625 collapsed to ~0.05 trigger over two
# runs). This is a supply mechanism, NOT the learning fix; under plain GRPO it
# raises trigger rate but the extra samples are largely cosmetic. Calibration
# target: find the N that holds trigger_rate ~0.15-0.25 without collapse, then
# feed that N into the decoupled-baseline (Direction 2) config. Scale (phi=4):
# N=3 -> -0.25, N=4 -> -0.5625, N=5 -> -1.0; N>=7 -> cosmetic blow-up, avoid.
# Watch trigger_rate (should stabilize) vs accuracy/reflection (cosmetic guard).
export TRAJ_GRPO_WRONG_NO_REFLECT_PENALTY="${TRAJ_GRPO_WRONG_NO_REFLECT_PENALTY:-1}"
export TRAJ_GRPO_WRONG_NO_REFLECT_N="${TRAJ_GRPO_WRONG_NO_REFLECT_N:-3}"
# 0 = train every reflection trigger/tag plus the final non-reflecting
# continuation. Ordinary reasoning before a later <reflection> is still masked,
# so failed intermediate reasoning is not reinforced; the opener token is kept
# trainable by reflection_start_tok boundary correction in reflection_grpo_loop.
# (Overridden by TRAJ_GRPO_TRAIN_FULL_CHAIN below when that is 1.)
export TRAJ_GRPO_MASK_LAST_REFLECTION_ONLY="${TRAJ_GRPO_MASK_LAST_REFLECTION_ONLY:-0}"
# Train the ENTIRE generated chain, including the pre-<reflection> reasoning
# (iter-1: the error chain that triggered the reflection; iter-2+: the prior
# continuation). =1 per the user's directive: learning *when* to pivot needs
# gradient on the tokens that lead into the reflection. This neutralises both
# the unconditional pre-reflection mask and MASK_LAST_REFLECTION_ONLY.
# KNOWN RISK (was the original reason the pre-chain was masked): on a CORRECT
# trajectory this gives positive gradient to the wrong reasoning that preceded
# the reflection, i.e. it can reinforce the very error -- a known 学偏 source
# and why SFT phase1 masks the wrong prefix. Watch for the model emitting a
# wrong chain on purpose just to reflect ("manufactured-error" / cosmetic
# reflection). Set =0 to revert to the paper-original masked behaviour.
export TRAJ_GRPO_TRAIN_FULL_CHAIN="${TRAJ_GRPO_TRAIN_FULL_CHAIN:-1}"
export TRAJ_GRPO_WRONG_REFLECTION_FULL_CHAIN_MASK="${TRAJ_GRPO_WRONG_REFLECTION_FULL_CHAIN_MASK:-0}"

project_name="${PROJECT_NAME:-Trajectory-GRPO-Math}"
exp_name="${EXP_NAME:-trajectory-grpo-qwen2.5-math-1.5b-with-last-turn-trained-and-sample01-reward}"
resume_mode="${RESUME_MODE:-auto}"
resume_from_path="${RESUME_FROM_PATH:-null}"

adv_estimator=grpo
grpo_skip_std_for_uniform_outcome="${GRPO_SKIP_STD_FOR_UNIFORM_OUTCOME:-True}"
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef="${KL_LOSS_COEF:-0.01}"

clip_ratio_low="${CLIP_RATIO_LOW:-0.20}"
clip_ratio_high="${CLIP_RATIO_HIGH:-0.26}"

max_prompt_length="${MAX_PROMPT_LENGTH:-1024}"
# Qwen2.5-Math-1.5B is configured with a 4k context locally. Reflection GRPO
# concatenates all generated rounds into one training response, so the safe
# upper bound is max_prompt_length + max_iters * max_response_length. The reward
# penalty horizon is controlled separately by TRAJ_GRPO_REWARD_PHI.
# Reflection count is unchanged from the original run (only the first
# reflection is now free, and the full chain is trained). With max_iters=3 the
# rollout's hard cap is remaining_reflections = max_iters-1 = 2 reflections.
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
    custom_reward_function.path=rgrpo.core_trajectory_grpo \
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
