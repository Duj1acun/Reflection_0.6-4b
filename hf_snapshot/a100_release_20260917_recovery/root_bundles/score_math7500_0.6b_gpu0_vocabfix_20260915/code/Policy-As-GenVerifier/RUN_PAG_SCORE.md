# Running PAG and SCoRe

## Qwen3-0.6B 四方法统一对照（2026-09-13）

新的 GRPO / REFLECT+ / PAG / SCoRe 对照实验入口、固定参数和评测说明见
[统一实验说明](../reflection_grpo_clean/experiments/four_methods/README.md)。
该实验使用清洗后的 DAPO17K、MATH500 + AMC23 验证、双卡 FSDP；
每 50 步评测 pass@4，温度 0.7。准备/预检不会启动正式训练。
以下内容保留为原 Qwen2.5-1.5B 实验的运行说明。

This checkout is pinned at commit `42fa41d8a2a6e820063bc1cd0aeaafe525d6af18`.

## Prepared local paths

- Environment: `/mnt/data/conda_envs/pag-score`
- Base model: `/mnt/data/models/Qwen2.5-1.5B-Instruct`
- Training data: `datasets/math7500.parquet` (7,500 rows)
- Validation data: `datasets/math500.parquet` (500 rows)
- Checkpoints: `/mnt/data/checkpoints/pag-score`

The local environment follows the repository's vLLM 0.8 Docker image: Python
3.10, PyTorch 2.6.0/CUDA 12.4, vLLM 0.8.2, Transformers 4.51.1 and
FlashAttention 2.7.4.post1. The historical `xgrammar==0.1.16` wheel is no
longer present on PyPI, so this environment uses 0.1.18. PAG's plain-text
generation path does not use xgrammar's constrained-decoding feature.

## Preflight

Run all commands from the repository root:

```bash
cd /root/Policy-As-GenVerifier
export PATH=/mnt/data/conda_envs/pag-score/bin:$PATH
export HF_HOME=/mnt/data/cache/huggingface

python - <<'PY'
import torch, vllm, transformers, flash_attn
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("vllm", vllm.__version__)
print("transformers", transformers.__version__)
print("flash_attn", flash_attn.__version__)
print("GPUs", torch.cuda.device_count())
PY

nvidia-smi
```

Do not start training until both local A800 GPUs are free. The launcher uses
both by default.

After the model download is complete, each method's rollout and rule-based
reward path can be checked on one free GPU:

```bash
cd /root/Policy-As-GenVerifier
export PATH=/mnt/data/conda_envs/pag-score/bin:$PATH
export VLLM_USE_V1=0

CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc-per-node=1 \
  quick_start/local_rollout_smoke.py pag
CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc-per-node=1 \
  quick_start/local_rollout_smoke.py score
```

## Local 2 x 40 GB run

The helper keeps the official global batch size, rollout count, learning rates,
two-turn setup and epoch schedule. It changes only memory/host details: two GPUs,
FSDP-sharded parameters and optimizer states kept on GPU, eager vLLM execution,
an 8,192-token per-GPU packing limit, 0.20 vLLM memory utilization,
console-only logging, big-disk checkpoints and one retained actor/critic
checkpoint per process. CPU offload is deliberately disabled: on this host,
two offloaded critic workers exceeded the 62 GiB RAM limit, while the tested
GPU-resident two-card configuration used about 29.7 GB allocated memory per GPU.

PAG:

```bash
cd /root/Policy-As-GenVerifier
set -o pipefail
CUDA_VISIBLE_DEVICES=0,1 \
  bash quick_start/local_2gpu.sh pag \
  2>&1 | tee /mnt/data/checkpoints/pag-score/pag.log
```

SCoRe (the launcher runs stage 1 and then auto-resumes it for stage 2):

```bash
cd /root/Policy-As-GenVerifier
set -o pipefail
CUDA_VISIBLE_DEVICES=0,1 \
  bash quick_start/local_2gpu.sh score \
  2>&1 | tee /mnt/data/checkpoints/pag-score/score.log
```

SCoRe runs both stages by default. To run or resume one stage explicitly, set
`SCORE_STAGE=1` or `SCORE_STAGE=2`; both use the same checkpoint directory:

```bash
CUDA_VISIBLE_DEVICES=0,1 SCORE_STAGE=1 bash quick_start/local_2gpu.sh score
CUDA_VISIBLE_DEVICES=0,1 SCORE_STAGE=2 bash quick_start/local_2gpu.sh score
```

To inspect the resolved commands without launching training:

```bash
DRY_RUN=1 bash quick_start/local_2gpu.sh pag
DRY_RUN=1 bash quick_start/local_2gpu.sh score
```

Extra arguments are appended as Hydra overrides. For a one-step integration
check, use a separate checkpoint directory so that it cannot be auto-resumed as
a real experiment:

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0,1 \
CKPT_ROOT=/mnt/data/checkpoints/pag-score-smoke \
TRAIN_BATCH_SIZE=4 \
PPO_MINI_BATCH_SIZE=2 \
MAX_TOKENS_PER_GPU=2048 \
GPU_MEMORY_UTILIZATION=0.20 \
ROLLOUT_N=1 \
VAL_N=1 \
  bash quick_start/local_2gpu.sh pag \
  trainer.total_training_steps=1 \
  trainer.val_before_train=false \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  data.max_prompt_length=512 \
  data.max_response_length=192 \
  2>&1 | tee /mnt/data/checkpoints/pag-score-smoke/pag_2gpu_ppo_1step.log
```

Replace `pag` with `score` and add `SCORE_STAGE=1` or `SCORE_STAGE=2` to test
the SCoRe stages separately.

## Verified scope

The following checks completed with exit code 0 on this machine:

- PAG and SCoRe real two-turn vLLM rollout plus rule-based reward smoke tests.
- PAG one full PPO step on one GPU and two consecutive PPO steps on 2 x A800,
  including critic and actor parameter updates.
- SCoRe stage 1 one full PPO step on one GPU and two consecutive PPO steps on
  2 x A800, including the reference-model turn-1 KL path.
- SCoRe stage 2 one full PPO step on one GPU, including joint two-turn reward,
  critic update and actor update.

The full 40-epoch experiments have not been run; the checks above establish
that the code paths and local resource configuration work, not that the final
paper metrics have already been reproduced. Integration-test logs are under
`/mnt/data/checkpoints/pag-score-smoke/`.

## Official 8-GPU reproduction

The untouched upstream scripts encode the paper run:

```bash
cd /root/Policy-As-GenVerifier
export PATH=/mnt/data/conda_envs/pag-score/bin:$PATH
export HF_HOME=/mnt/data/cache/huggingface
export VLLM_USE_V1=0

# Change MODEL_PATH and CKPT_PATH in the scripts first, or provide equivalent
# absolute paths in copies of the scripts.
bash quick_start/qwen1p5b_pag.sh
bash quick_start/qwen1p5b_SCoRe.sh
```

Both upstream scripts specify `trainer.n_gpus_per_node=8`. PAG trains for 40
epochs. SCoRe stage 1 trains to 20 epochs with a turn-1 KL penalty; stage 2 uses
the same checkpoint directory and `resume_mode=auto`, then continues to the
40-epoch total with both turns optimized.

## Evaluation

The repository evaluation script requires a *sharded training checkpoint*
whose directory name ends in `global_step_<N>`, not a merged Hugging Face model.
Use the local evaluator with a checkpoint from the matching method:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash quick_start/local_eval.sh pag \
  /mnt/data/checkpoints/pag-score/PAG/qwen1p5b_pag_2gpu/global_step_<N>

CUDA_VISIBLE_DEVICES=0,1 bash quick_start/local_eval.sh score \
  /mnt/data/checkpoints/pag-score/PAG/qwen1p5b_score_2gpu/global_step_<N>
```

The helper selects `rollout_type=pag` for PAG and `rollout_type=multiturn` for
SCoRe. The official evaluation samples 32 outputs per MATH500 question. For a
quick sanity evaluation, prefix the command with `VAL_N=1`.

To merge only the actor checkpoint into Hugging Face format:

```bash
python scripts/model_merger.py \
  --backend fsdp \
  --hf_model_path /mnt/data/models/Qwen2.5-1.5B-Instruct \
  --local_dir /mnt/data/checkpoints/pag-score/PAG/qwen1p5b_pag_2gpu/global_step_<N>/actor \
  --target_dir /mnt/data/models/pag-qwen1p5b-hf
```

The merged directory is useful for external inference harnesses. The supplied
`evaluation.sh` should continue to use the unmerged `global_step_<N>` directory.
