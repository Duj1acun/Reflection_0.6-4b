# MATH7500 三个 baseline 参数核对

配置 SHA256：`0d693ec7b1e888c1eb60c1e04b74bb100d7716601495942259db538403b78afa`。仅核对配置，不启动正式训练。

当前范围：GRPO → PAG → SCoRe，每个方法依次使用 GPU 0、1 两张 A800 40GB，Qwen3-0.6B Base 全参数训练。
四列包含 SCoRe 两个阶段；配置来自实际 build() 展开，当前为长度扩展后的新配置；旧GPU预检不适用于此配置，需重做。

## 阅读时必须区分的运行语义

- 每步64题×8轨迹=512轨迹。ppo_mini_batch_size=64会乘rollout.n再除双卡，变为每卡256条轨迹/次更新；动态微批按每卡16384 tokens组批，micro_batch_size=2不是固定实际微批大小。
- 每个方法500累计训练步，约4.27轮题目曝光；每epoch保留117个完整batch（7488题），末尾8题丢弃，下一轮重新洗牌。total_epochs=100000只是避免epoch上限提前截停，由total_training_steps限制。
- SCoRe stage 1到250步，stage 2从global_step_250恢复完整状态到累计500步。不是从第一阶段best恢复，也不是额外500步。
- 三个baseline每次完整解题上限4096：GRPO4096；PAG4096+2048验证+4096条件重解（总10240）；SCoRe4096+4096（总8192）。训练与MATH500/AMC23验证相同，未用额度不转移。输入题目上限2048，含历史上下文上限16384。
- PAG验证判断correct则停，wrong或未解析到correct则重解；SCoRe固定第二轮修正。标准答案只交给奖励函数，不放入模型的反馈prompt。
- actor为AdamW，lr1e-6、betas=(0.9,0.999)、默认eps=1e-8、weight_decay=0；critic为AdamW，lr2e-6、相同betas/eps、weight_decay=.01。学习率恒定，无warmup。
- FSDP为strategy=fsdp（不是fsdp2），在两张GPU分片；混合精度param为BF16、reduce/buffer为FP32。GPU内保留参数/优化器状态，不做CPU offload。
- GRPO不用critic、reference、独立reward model，其对应YAML默认路径和学习率不生效；PAG/SCoRe使用critic，只有SCoRe第一阶段启用冻结Base reference。三者均不用独立神经reward model或API。
- actor损失只作用于生成token：GRPO答案、PAG答案及验证文本、SCoRe两轮答案。原题及注入的反馈指令不计actor损失。SCoRe第一阶段也优化两轮生成token，首轮KL=.1，次轮KL=0。
- r1/r2为首轮/末轮数学正确性0或1：GRPO奖励r1；PAG首轮r1、验证判断正确得1否则0，重解奖励r2+(r2-r1)；SCoRe第一阶段各轮r1/r2另加首轮KL惩罚，第二阶段首轮r1、次轮r2+(r2-r1)。
- GRPO同题8回答内标准化优势，不过滤全对/全错组；PAG GAE按role标准化；SCoRe GAE按whole标准化；gamma=lambda=1。
- 每50步验证和保存，含训练前step0验证。540题各4条完整轨迹=2160轨迹，按最后一轮解题答案评分，不做跨轮oracle择优。
- best按(MATH500 pass@4+AMC23 pass@4)/2选，平局按宏平均mean@4，再平局取较早step。保留best和latest，SCoRe额外保留250；不把step0列为训练ckpt候选。
- 两个验证集同时用于选ckpt，因此最佳验证分数不是独立未参与选择的测试结果。各方法共享单次解题上限，轨迹总预算不同，必须同时报告实际生成token和时间成本。
- 下方完整配置包含未启用模块的默认值；warmup_style=constant时min_lr_ratio/num_cycles不影响学习率。runtime还会写入optimizer总步数，缩放mini-batch，并按上述规则管理checkpoint。

## Prompt 原文

三个方法首轮共同system prompt

```text
Please reason step by step, and put your final answer within \boxed{}.
```

PAG验证user反馈

```text
Check the math solution step by step. If you find a mistake, state the wrong step, explain why it is wrong, and end with exactly 'The answer is wrong.' Otherwise, end with exactly 'The answer is correct.'
```

PAG重解user反馈

```text
Review the previous solution and verification, then provide a corrected solution. Put your final answer within \boxed{}.
```

SCoRe第二轮user反馈

```text
There might be an error in the solution above because of lack of understanding of the question. Please correct the error, if any, and rewrite the solution. Put your final answer within \boxed{}.
```


## 实验输入配置

```json
{
  "experiment": "four_methods_math7500_0.6b_20260914",
  "output": "/mnt/data/reflection_grpo/artifacts/four_methods_math7500_0.6b_20260914",
  "python": "/home/vipuser/miniconda3/envs/wrx_env/bin/python",
  "reflection_repo": "/root/reflection_grpo_clean",
  "pag_repo": "/root/Policy-As-GenVerifier",
  "base_model": "/mnt/data/models/Qwen3-0.6B",
  "sft_adapter": "/mnt/data/reflection_grpo/artifacts/paired_sft_20260912/qwen3-0.6b/phase2/checkpoint-268",
  "train_source": "/root/Policy-As-GenVerifier/datasets/math7500.parquet",
  "math500_source": "/root/Policy-As-GenVerifier/datasets/math500.parquet",
  "amc23_source": "/mnt/data/reflection_grpo/artifacts/paired_eval_pass4_20260912/qwen3-0.6b-base__amc23.dataset.jsonl",
  "seed": 1234,
  "steps": 500,
  "score_stage1_steps": 250,
  "train_batch_size": 64,
  "train_n": 8,
  "ppo_mini_batch_size": 64,
  "ppo_micro_batch_size_per_gpu": 2,
  "actor_lr": 1e-06,
  "critic_lr": 2e-06,
  "max_prompt_length": 2048,
  "max_model_len": 16384,
  "total_generated_tokens": {
    "grpo": 4096,
    "reflect": 4096,
    "pag": 10240,
    "score": 8192
  },
  "stage_tokens": {
    "grpo": [
      4096
    ],
    "reflect": [
      2048,
      2048
    ],
    "pag": [
      4096,
      2048,
      4096
    ],
    "score": [
      4096,
      4096
    ]
  },
  "train_sampling": {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": -1
  },
  "validation": {
    "frequency": 50,
    "n": 4,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "seed": 1234,
    "batch_size": 128
  },
  "gpu_memory_utilization": 0.5,
  "max_num_seqs": 32,
  "max_num_batched_tokens": 8192,
  "tokens_per_gpu": 16384,
  "gpus": "0,1",
  "train_dataset": "math7500",
  "expected_train_questions": 7496,
  "verified_overlap_source_ids": {
    "304": "test/algebra/170.json"
  },
  "scorer_version": "fourway-math-v5-20260914",
  "train_source_sha256": "d10fad2d7d8b30dda2649ffc6a1ac6c9f5e047e02d9253d6c06fba9b4edcba03",
  "enforce_eager": false
}
```

## 全量框架配置（包括未启用模块默认项）

| 配置键 | GRPO | PAG | SCoRe stage 1 | SCoRe stage 2 |
|---|---|---|---|---|
| `actor_rollout_ref.actor.checkpoint.contents` | `["model", "optimizer", "extra"]` | `["model", "optimizer", "extra"]` | `["model", "optimizer", "extra"]` | `["model", "optimizer", "extra"]` |
| `actor_rollout_ref.actor.clip_ratio` | `0.2` | `0.2` | `0.2` | `0.2` |
| `actor_rollout_ref.actor.clip_ratio_c` | `3.0` | `3.0` | `3.0` | `3.0` |
| `actor_rollout_ref.actor.clip_ratio_high` | `0.26` | `0.26` | `0.26` | `0.26` |
| `actor_rollout_ref.actor.clip_ratio_low` | `0.2` | `0.2` | `0.2` | `0.2` |
| `actor_rollout_ref.actor.entropy_coeff` | `0.0` | `0.0` | `0.0` | `0.0` |
| `actor_rollout_ref.actor.fsdp_config.fsdp_size` | `-1` | `-1` | `-1` | `-1` |
| `actor_rollout_ref.actor.fsdp_config.offload_policy` | `false` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `actor_rollout_ref.actor.fsdp_config.optimizer_offload` | `false` | `false` | `false` | `false` |
| `actor_rollout_ref.actor.fsdp_config.param_offload` | `false` | `false` | `false` | `false` |
| `actor_rollout_ref.actor.fsdp_config.reshard_after_forward` | `true` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `actor_rollout_ref.actor.fsdp_config.wrap_policy.min_num_params` | `0` | `0` | `0` | `0` |
| `actor_rollout_ref.actor.grad_clip` | `1.0` | `1.0` | `1.0` | `1.0` |
| `actor_rollout_ref.actor.invalid_action_penalty_coef` | `0.1` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `actor_rollout_ref.actor.kl_loss_coef` | `0.0` | `0.0` | `0.0` | `0.0` |
| `actor_rollout_ref.actor.kl_loss_type` | `"low_var_kl"` | `"low_var_kl"` | `"low_var_kl"` | `"low_var_kl"` |
| `actor_rollout_ref.actor.loss_agg_mode` | `"seq-mean-token-mean"` | `"seq-mean-token-mean"` | `"seq-mean-token-mean"` | `"seq-mean-token-mean"` |
| `actor_rollout_ref.actor.optim.betas` | `[0.9, 0.999]` | `[0.9, 0.999]` | `[0.9, 0.999]` | `[0.9, 0.999]` |
| `actor_rollout_ref.actor.optim.lr` | `1e-06` | `1e-06` | `1e-06` | `1e-06` |
| `actor_rollout_ref.actor.optim.lr_warmup_steps` | `0` | `0` | `0` | `0` |
| `actor_rollout_ref.actor.optim.lr_warmup_steps_ratio` | `0.0` | `0.0` | `0.0` | `0.0` |
| `actor_rollout_ref.actor.optim.min_lr_ratio` | `0.0` | `null` | `null` | `null` |
| `actor_rollout_ref.actor.optim.num_cycles` | `0.5` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `actor_rollout_ref.actor.optim.total_training_steps` | `-1` | `-1` | `-1` | `-1` |
| `actor_rollout_ref.actor.optim.warmup_style` | `"constant"` | `"constant"` | `"constant"` | `"constant"` |
| `actor_rollout_ref.actor.optim.weight_decay` | `0.0` | `0.0` | `0.0` | `0.0` |
| `actor_rollout_ref.actor.policy_loss.loss_mode` | `"vanilla"` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `actor_rollout_ref.actor.ppo_epochs` | `1` | `1` | `1` | `1` |
| `actor_rollout_ref.actor.ppo_max_token_len_per_gpu` | `16384` | `16384` | `16384` | `16384` |
| `actor_rollout_ref.actor.ppo_micro_batch_size` | `null` | `null` | `null` | `null` |
| `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` | `2` | `2` | `2` | `2` |
| `actor_rollout_ref.actor.ppo_mini_batch_size` | `64` | `64` | `64` | `64` |
| `actor_rollout_ref.actor.shuffle` | `false` | `false` | `false` | `false` |
| `actor_rollout_ref.actor.strategy` | `"fsdp"` | `"fsdp"` | `"fsdp"` | `"fsdp"` |
| `actor_rollout_ref.actor.ulysses_sequence_parallel_size` | `1` | `1` | `1` | `1` |
| `actor_rollout_ref.actor.use_dynamic_bsz` | `true` | `true` | `true` | `true` |
| `actor_rollout_ref.actor.use_invalid_action_penalty` | `false` | `false` | `false` | `false` |
| `actor_rollout_ref.actor.use_kl_loss` | `false` | `false` | `false` | `false` |
| `actor_rollout_ref.actor.use_multiturn_loss` | —（该版本无此键） | `false` | `false` | `false` |
| `actor_rollout_ref.actor.use_torch_compile` | `true` | `true` | `true` | `true` |
| `actor_rollout_ref.hybrid_engine` | `true` | `true` | `true` | `true` |
| `actor_rollout_ref.model.enable_activation_offload` | `false` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `actor_rollout_ref.model.enable_gradient_checkpointing` | `true` | `true` | `true` | `true` |
| `actor_rollout_ref.model.external_lib` | `null` | `null` | `null` | `null` |
| `actor_rollout_ref.model.lora_alpha` | `16` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `actor_rollout_ref.model.lora_rank` | `0` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `actor_rollout_ref.model.override_config` | `{}` | `{}` | `{}` | `{}` |
| `actor_rollout_ref.model.path` | `"/mnt/data/models/Qwen3-0.6B"` | `"/mnt/data/models/Qwen3-0.6B"` | `"/mnt/data/models/Qwen3-0.6B"` | `"/mnt/data/models/Qwen3-0.6B"` |
| `actor_rollout_ref.model.target_modules` | `"all-linear"` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `actor_rollout_ref.model.trust_remote_code` | `false` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `actor_rollout_ref.model.use_fused_kernels` | `false` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `actor_rollout_ref.model.use_liger` | `false` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `actor_rollout_ref.model.use_remove_padding` | `true` | `true` | `true` | `true` |
| `actor_rollout_ref.model.use_shm` | `false` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `actor_rollout_ref.ref.fsdp_config.param_offload` | `false` | `false` | `false` | `false` |
| `actor_rollout_ref.ref.fsdp_config.reshard_after_forward` | `true` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `actor_rollout_ref.ref.fsdp_config.wrap_policy.min_num_params` | `0` | `0` | `0` | `0` |
| `actor_rollout_ref.ref.log_prob_max_token_len_per_gpu` | `16384` | `16384` | `16384` | `16384` |
| `actor_rollout_ref.ref.log_prob_micro_batch_size` | `null` | `null` | `null` | `null` |
| `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu` | `2` | `2` | `2` | `2` |
| `actor_rollout_ref.ref.log_prob_use_dynamic_bsz` | `true` | `true` | `true` | `true` |
| `actor_rollout_ref.ref.strategy` | `"fsdp"` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `actor_rollout_ref.ref.ulysses_sequence_parallel_size` | `1` | `1` | `1` | `1` |
| `actor_rollout_ref.ref.use_torch_compile` | `true` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `actor_rollout_ref.rollout.bad_words` | `["<think>", "</think>"]` | `["<think>", "</think>"]` | `["<think>", "</think>"]` | `["<think>", "</think>"]` |
| `actor_rollout_ref.rollout.chat_scheduler` | `null` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `actor_rollout_ref.rollout.disable_log_stats` | `true` | `true` | `true` | `true` |
| `actor_rollout_ref.rollout.do_sample` | `true` | `true` | `true` | `true` |
| `actor_rollout_ref.rollout.dtype` | `"bfloat16"` | `"bfloat16"` | `"bfloat16"` | `"bfloat16"` |
| `actor_rollout_ref.rollout.enable_chunked_prefill` | `true` | `true` | `true` | `true` |
| `actor_rollout_ref.rollout.end_with_verifer` | —（该版本无此键） | `false` | `false` | `false` |
| `actor_rollout_ref.rollout.enforce_eager` | `false` | `false` | `false` | `false` |
| `actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend` | `null` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `actor_rollout_ref.rollout.engine_kwargs.vllm.max_num_seqs` | `32` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `actor_rollout_ref.rollout.engine_kwargs.vllm.swap_space` | `null` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `actor_rollout_ref.rollout.free_cache_engine` | `false` | `false` | `false` | `false` |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | `0.5` | `0.5` | `0.5` | `0.5` |
| `actor_rollout_ref.rollout.ignore_eos` | `false` | `false` | `false` | `false` |
| `actor_rollout_ref.rollout.is_golden_generation` | —（该版本无此键） | `false` | `false` | `false` |
| `actor_rollout_ref.rollout.is_only_genrm` | —（该版本无此键） | `false` | `false` | `false` |
| `actor_rollout_ref.rollout.layered_summon` | `false` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `actor_rollout_ref.rollout.load_format` | `"dummy_dtensor"` | `"dummy_dtensor"` | `"dummy_dtensor"` | `"dummy_dtensor"` |
| `actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu` | `16384` | `16384` | `16384` | `16384` |
| `actor_rollout_ref.rollout.log_prob_micro_batch_size` | `null` | `null` | `null` | `null` |
| `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu` | `2` | `2` | `2` | `2` |
| `actor_rollout_ref.rollout.log_prob_use_dynamic_bsz` | `true` | `true` | `true` | `true` |
| `actor_rollout_ref.rollout.max_model_len` | `16384` | `16384` | `16384` | `16384` |
| `actor_rollout_ref.rollout.max_num_batched_tokens` | `8192` | `8192` | `8192` | `8192` |
| `actor_rollout_ref.rollout.max_num_seqs` | `32` | `32` | `32` | `32` |
| `actor_rollout_ref.rollout.mode` | `"sync"` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `actor_rollout_ref.rollout.multi_turn.enable` | `false` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `actor_rollout_ref.rollout.multi_turn.format` | `"chatml"` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `actor_rollout_ref.rollout.multi_turn.max_turns` | `null` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `actor_rollout_ref.rollout.multi_turn.tool_config_path` | `null` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `actor_rollout_ref.rollout.n` | `8` | `8` | `8` | `8` |
| `actor_rollout_ref.rollout.name` | `"vllm"` | `"vllm"` | `"vllm"` | `"vllm"` |
| `actor_rollout_ref.rollout.num_turns` | —（该版本无此键） | `2` | `2` | `2` |
| `actor_rollout_ref.rollout.prompt_length` | `2048` | `2048` | `2048` | `2048` |
| `actor_rollout_ref.rollout.response_length` | `4096` | `4096` | `4096` | `4096` |
| `actor_rollout_ref.rollout.rollout_type` | —（该版本无此键） | `"pag"` | `"multiturn"` | `"multiturn"` |
| `actor_rollout_ref.rollout.seed` | `1234` | `1234` | `1234` | `1234` |
| `actor_rollout_ref.rollout.temperature` | `1.0` | `1.0` | `1.0` | `1.0` |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | `1` | `1` | `1` | `1` |
| `actor_rollout_ref.rollout.top_k` | `-1` | `-1` | `-1` | `-1` |
| `actor_rollout_ref.rollout.top_p` | `1.0` | `1.0` | `1.0` | `1.0` |
| `actor_rollout_ref.rollout.use_fire_sampling` | `false` | `false` | `false` | `false` |
| `actor_rollout_ref.rollout.val_kwargs.do_sample` | `true` | `true` | `true` | `true` |
| `actor_rollout_ref.rollout.val_kwargs.n` | `4` | `4` | `4` | `4` |
| `actor_rollout_ref.rollout.val_kwargs.num_turns` | —（该版本无此键） | `2` | `2` | `2` |
| `actor_rollout_ref.rollout.val_kwargs.temperature` | `0.7` | `0.7` | `0.7` | `0.7` |
| `actor_rollout_ref.rollout.val_kwargs.top_k` | `20` | `20` | `20` | `20` |
| `actor_rollout_ref.rollout.val_kwargs.top_p` | `0.8` | `0.8` | `0.8` | `0.8` |
| `algorithm.adv_estimator` | `"grpo"` | `"gae"` | `"gae"` | `"gae"` |
| `algorithm.filter_groups.enable` | `false` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `algorithm.filter_groups.max_num_gen_batches` | `10` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `algorithm.gamma` | `1.0` | `1.0` | `1.0` | `1.0` |
| `algorithm.gigpo.enable_similarity` | `false` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `algorithm.gigpo.mode` | `"mean_norm"` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `algorithm.gigpo.similarity_thresh` | `0.95` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `algorithm.gigpo.step_advantage_w` | `1.0` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `algorithm.grpo_skip_std_for_uniform_outcome` | `false` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `algorithm.kl_ctrl.horizon` | `10000` | `10000` | `10000` | `10000` |
| `algorithm.kl_ctrl.kl_coef` | `0.0` | `0.0` | `0.0` | `0.0` |
| `algorithm.kl_ctrl.kl_coef_turn1` | —（该版本无此键） | `0.0` | `0.1` | `0.0` |
| `algorithm.kl_ctrl.kl_coef_turn2` | —（该版本无此键） | `0.0` | `0.0` | `0.0` |
| `algorithm.kl_ctrl.multiturn_kl` | —（该版本无此键） | `false` | `true` | `false` |
| `algorithm.kl_ctrl.target_kl` | `0.1` | `0.1` | `0.1` | `0.1` |
| `algorithm.kl_ctrl.type` | `"fixed"` | `"fixed"` | `"fixed"` | `"fixed"` |
| `algorithm.kl_penalty` | `"kl"` | `"kl"` | `"kl"` | `"kl"` |
| `algorithm.lam` | `1.0` | `1.0` | `1.0` | `1.0` |
| `algorithm.norm_adv_by_std_in_grpo` | `true` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `algorithm.norm_type` | —（该版本无此键） | `"role"` | `"whole"` | `"whole"` |
| `algorithm.pf_ppo.reweight_method` | `"pow"` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `algorithm.pf_ppo.weight_pow` | `2.0` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `algorithm.use_kl_in_reward` | `false` | `false` | `true` | `false` |
| `algorithm.use_pf_ppo` | `false` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `critic.checkpoint.contents` | `["model", "optimizer", "extra"]` | `["model", "optimizer", "extra"]` | `["model", "optimizer", "extra"]` | `["model", "optimizer", "extra"]` |
| `critic.cliprange_value` | `0.5` | `0.5` | `0.5` | `0.5` |
| `critic.forward_max_token_len_per_gpu` | `32768` | `16384` | `16384` | `16384` |
| `critic.forward_micro_batch_size` | `null` | `null` | `null` | `null` |
| `critic.forward_micro_batch_size_per_gpu` | `null` | `2` | `2` | `2` |
| `critic.grad_clip` | `1.0` | `1.0` | `1.0` | `1.0` |
| `critic.loss_agg_mode` | `"seq-mean-token-mean"` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `critic.model.enable_activation_offload` | `false` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `critic.model.enable_gradient_checkpointing` | `true` | `true` | `true` | `true` |
| `critic.model.external_lib` | `null` | `null` | `null` | `null` |
| `critic.model.fsdp_config.fsdp_size` | `-1` | `-1` | `-1` | `-1` |
| `critic.model.fsdp_config.offload_policy` | `false` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `critic.model.fsdp_config.optimizer_offload` | `false` | `false` | `false` | `false` |
| `critic.model.fsdp_config.param_offload` | `false` | `false` | `false` | `false` |
| `critic.model.fsdp_config.reshard_after_forward` | `true` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `critic.model.fsdp_config.wrap_policy.min_num_params` | `0` | `0` | `0` | `0` |
| `critic.model.lora_alpha` | `16` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `critic.model.lora_rank` | `0` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `critic.model.override_config` | `{}` | `{}` | `{}` | `{}` |
| `critic.model.path` | `"~/models/deepseek-llm-7b-chat"` | `"/mnt/data/models/Qwen3-0.6B"` | `"/mnt/data/models/Qwen3-0.6B"` | `"/mnt/data/models/Qwen3-0.6B"` |
| `critic.model.target_modules` | `"all-linear"` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `critic.model.tokenizer_path` | `"/mnt/data/models/Qwen3-0.6B"` | `"/mnt/data/models/Qwen3-0.6B"` | `"/mnt/data/models/Qwen3-0.6B"` | `"/mnt/data/models/Qwen3-0.6B"` |
| `critic.model.trust_remote_code` | `false` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `critic.model.use_remove_padding` | `false` | `true` | `true` | `true` |
| `critic.model.use_shm` | `false` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `critic.optim.lr` | `1e-05` | `2e-06` | `2e-06` | `2e-06` |
| `critic.optim.lr_warmup_steps` | —（该版本无此键） | `0` | `0` | `0` |
| `critic.optim.lr_warmup_steps_ratio` | `0.0` | `0.0` | `0.0` | `0.0` |
| `critic.optim.min_lr_ratio` | `null` | `null` | `null` | `null` |
| `critic.optim.total_training_steps` | `-1` | `-1` | `-1` | `-1` |
| `critic.optim.warmup_style` | `"constant"` | `"constant"` | `"constant"` | `"constant"` |
| `critic.optim.weight_decay` | `0.01` | `0.01` | `0.01` | `0.01` |
| `critic.ppo_epochs` | `1` | `1` | `1` | `1` |
| `critic.ppo_max_token_len_per_gpu` | `32768` | `16384` | `16384` | `16384` |
| `critic.ppo_micro_batch_size` | `null` | `null` | `null` | `null` |
| `critic.ppo_micro_batch_size_per_gpu` | `null` | `2` | `2` | `2` |
| `critic.ppo_mini_batch_size` | `64` | `64` | `64` | `64` |
| `critic.rollout_n` | `8` | `8` | `8` | `8` |
| `critic.shuffle` | `false` | `false` | `false` | `false` |
| `critic.strategy` | `"fsdp"` | `"fsdp"` | `"fsdp"` | `"fsdp"` |
| `critic.ulysses_sequence_parallel_size` | `1` | `1` | `1` | `1` |
| `critic.use_dynamic_bsz` | `true` | `true` | `true` | `true` |
| `custom_reward_function.name` | `"compute_math_score"` | `"compute_score"` | `"compute_score"` | `"compute_score"` |
| `custom_reward_function.path` | `"rgrpo.core_hybrid_grpo_v2"` | `null` | `null` | `null` |
| `data.custom_cls.name` | `null` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `data.custom_cls.path` | `null` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `data.dataloader_num_workers` | `0` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `data.filter_overlong_prompts` | `false` | `false` | `false` | `false` |
| `data.filter_overlong_prompts_workers` | `1` | `1` | `1` | `1` |
| `data.image_key` | `"images"` | `"images"` | `"images"` | `"images"` |
| `data.max_prompt_length` | `2048` | `2048` | `2048` | `2048` |
| `data.max_response_length` | `4096` | `4096` | `4096` | `4096` |
| `data.prompt_key` | `"prompt"` | `"prompt"` | `"prompt"` | `"prompt"` |
| `data.return_full_prompt` | `false` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `data.return_raw_chat` | `true` | `true` | `true` | `true` |
| `data.return_raw_input_ids` | `false` | `false` | `false` | `false` |
| `data.reward_fn_key` | `"data_source"` | `"data_source"` | `"data_source"` | `"data_source"` |
| `data.seed` | `1234` | `1234` | `1234` | `1234` |
| `data.shuffle` | `true` | `true` | `true` | `true` |
| `data.tokenizer` | `null` | `null` | `null` | `null` |
| `data.train_batch_size` | `64` | `64` | `64` | `64` |
| `data.train_files` | `["/mnt/data/reflection_grpo/artifacts/four_methods_math7500_0.6b_20260914/data/grpo.train.parquet"]` | `["/mnt/data/reflection_grpo/artifacts/four_methods_math7500_0.6b_20260914/data/pag.train.parquet"]` | `["/mnt/data/reflection_grpo/artifacts/four_methods_math7500_0.6b_20260914/data/score.train.parquet"]` | `["/mnt/data/reflection_grpo/artifacts/four_methods_math7500_0.6b_20260914/data/score.train.parquet"]` |
| `data.truncation` | `"error"` | `"error"` | `"error"` | `"error"` |
| `data.trust_remote_code` | `false` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `data.use_shm` | `false` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `data.val_batch_size` | `128` | `128` | `128` | `128` |
| `data.val_files` | `["/mnt/data/reflection_grpo/artifacts/four_methods_math7500_0.6b_20260914/data/grpo.validation.parquet"]` | `["/mnt/data/reflection_grpo/artifacts/four_methods_math7500_0.6b_20260914/data/pag.validation.parquet"]` | `["/mnt/data/reflection_grpo/artifacts/four_methods_math7500_0.6b_20260914/data/score.validation.parquet"]` | `["/mnt/data/reflection_grpo/artifacts/four_methods_math7500_0.6b_20260914/data/score.validation.parquet"]` |
| `data.video_key` | `"videos"` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `env.alfworld.eval_dataset` | `"eval_in_distribution"` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `env.env_name` | `"none"` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `env.history_length` | `2` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `env.max_steps` | `25` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `env.resources_per_worker.num_cpus` | `0.1` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `env.resources_per_worker.num_gpus` | `0` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `env.rollout.n` | `1` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `env.search.log_requests` | `false` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `env.search.search_url` | `"http://127.0.0.1:8000/retrieve"` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `env.search.timeout` | `60` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `env.search.topk` | `3` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `env.seed` | `0` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `env.sokoban.dim_room` | `[6, 6]` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `env.sokoban.mode` | `"tiny_rgb_array"` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `env.sokoban.num_boxes` | `1` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `env.sokoban.search_depth` | `30` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `env.webshop.human_goals` | `false` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `env.webshop.use_small` | `true` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `fourway.method` | `"grpo"` | `"pag"` | `"score"` | `"score"` |
| `fourway.output` | `"/mnt/data/reflection_grpo/artifacts/four_methods_math7500_0.6b_20260914"` | `"/mnt/data/reflection_grpo/artifacts/four_methods_math7500_0.6b_20260914"` | `"/mnt/data/reflection_grpo/artifacts/four_methods_math7500_0.6b_20260914"` | `"/mnt/data/reflection_grpo/artifacts/four_methods_math7500_0.6b_20260914"` |
| `fourway.probe` | `false` | `false` | `false` | `false` |
| `fourway.stage` | `1` | `1` | `1` | `2` |
| `ray_init._temp_dir` | `"/mnt/data/tmp/fw"` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `ray_init.num_cpus` | `16` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `ray_init.object_store_memory` | `2147483648` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `reward_model.compute_score` | —（该版本无此键） | `null` | `null` | `null` |
| `reward_model.enable` | `false` | `false` | `false` | `false` |
| `reward_model.end_with_verifer` | —（该版本无此键） | `false` | `false` | `false` |
| `reward_model.forward_max_token_len_per_gpu` | `32768` | `16384` | `16384` | `16384` |
| `reward_model.is_golden_generation` | —（该版本无此键） | `false` | `false` | `false` |
| `reward_model.is_only_genrm` | —（该版本无此键） | `false` | `false` | `false` |
| `reward_model.launch_reward_fn_async` | `false` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `reward_model.max_length` | `null` | `null` | `null` | `null` |
| `reward_model.micro_batch_size` | `null` | `null` | `null` | `null` |
| `reward_model.micro_batch_size_per_gpu` | `null` | `null` | `null` | `null` |
| `reward_model.model.external_lib` | `null` | `null` | `null` | `null` |
| `reward_model.model.fsdp_config.fsdp_size` | `-1` | `-1` | `-1` | `-1` |
| `reward_model.model.fsdp_config.param_offload` | `false` | `false` | `false` | `false` |
| `reward_model.model.fsdp_config.reshard_after_forward` | `true` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `reward_model.model.fsdp_config.wrap_policy.min_num_params` | `0` | `0` | `0` | `0` |
| `reward_model.model.input_tokenizer` | `"/mnt/data/models/Qwen3-0.6B"` | `"/mnt/data/models/Qwen3-0.6B"` | `"/mnt/data/models/Qwen3-0.6B"` | `"/mnt/data/models/Qwen3-0.6B"` |
| `reward_model.model.path` | `"~/models/FsfairX-LLaMA3-RM-v0.1"` | `"~/models/FsfairX-LLaMA3-RM-v0.1"` | `"~/models/FsfairX-LLaMA3-RM-v0.1"` | `"~/models/FsfairX-LLaMA3-RM-v0.1"` |
| `reward_model.model.trust_remote_code` | `false` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `reward_model.model.use_fused_kernels` | `false` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `reward_model.model.use_remove_padding` | `false` | `false` | `false` | `false` |
| `reward_model.model.use_shm` | `false` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `reward_model.num_turns` | —（该版本无此键） | `2` | `2` | `2` |
| `reward_model.policy_rs` | —（该版本无此键） | `true` | `false` | `true` |
| `reward_model.reward_manager` | `"naive"` | `"pag"` | `"multiturn"` | `"multiturn"` |
| `reward_model.rs_coef` | —（该版本无此键） | `1.0` | `1.0` | `1.0` |
| `reward_model.sandbox_fusion.max_concurrent` | `64` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `reward_model.sandbox_fusion.url` | `null` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `reward_model.strategy` | `"fsdp"` | `"fsdp"` | `"fsdp"` | `"fsdp"` |
| `reward_model.ulysses_sequence_parallel_size` | `1` | `1` | `1` | `1` |
| `reward_model.use_dynamic_bsz` | `true` | `true` | `true` | `true` |
| `trainer.balance_batch` | `true` | `true` | `true` | `true` |
| `trainer.critic_warmup` | `0` | `0` | `0` | `0` |
| `trainer.default_hdfs_dir` | `null` | `null` | `null` | `null` |
| `trainer.default_local_dir` | `"/mnt/data/reflection_grpo/artifacts/four_methods_math7500_0.6b_20260914/runs/grpo"` | `"/mnt/data/reflection_grpo/artifacts/four_methods_math7500_0.6b_20260914/runs/pag"` | `"/mnt/data/reflection_grpo/artifacts/four_methods_math7500_0.6b_20260914/runs/score"` | `"/mnt/data/reflection_grpo/artifacts/four_methods_math7500_0.6b_20260914/runs/score"` |
| `trainer.del_local_ckpt_after_load` | `false` | `false` | `false` | `false` |
| `trainer.device` | `"cuda"` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `trainer.experiment_name` | `"grpo"` | `"pag"` | `"score"` | `"score"` |
| `trainer.log_val_generations` | `0` | `0` | `0` | `0` |
| `trainer.logger` | `["console"]` | `["console"]` | `["console"]` | `["console"]` |
| `trainer.max_actor_ckpt_to_keep` | `null` | `null` | `null` | `null` |
| `trainer.max_critic_ckpt_to_keep` | `null` | `null` | `null` | `null` |
| `trainer.n_gpus_per_node` | `2` | `2` | `2` | `2` |
| `trainer.nnodes` | `1` | `1` | `1` | `1` |
| `trainer.project_name` | `"four_methods_math7500_0.6b_20260914"` | `"four_methods_math7500_0.6b_20260914"` | `"four_methods_math7500_0.6b_20260914"` | `"four_methods_math7500_0.6b_20260914"` |
| `trainer.ray_wait_register_center_timeout` | `300` | `300` | `300` | `300` |
| `trainer.resume_from_path` | `null` | `null` | `null` | `"/mnt/data/reflection_grpo/artifacts/four_methods_math7500_0.6b_20260914/runs/score/global_step_250"` |
| `trainer.resume_mode` | `"disable"` | `"disable"` | `"disable"` | `"resume_path"` |
| `trainer.rollout_data_dir` | `null` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `trainer.save_freq` | `50` | `50` | `50` | `50` |
| `trainer.save_validation_results` | —（该版本无此键） | `false` | `false` | `false` |
| `trainer.seed` | `1234` | `1234` | `1234` | `1234` |
| `trainer.test_freq` | `50` | `50` | `50` | `50` |
| `trainer.total_epochs` | `100000` | `100000` | `100000` | `100000` |
| `trainer.total_training_steps` | `500` | `500` | `250` | `500` |
| `trainer.val_before_train` | `true` | `true` | `true` | `false` |
| `trainer.val_only` | `false` | `false` | `false` | `false` |
| `trainer.validation_data_dir` | `null` | —（该版本无此键） | —（该版本无此键） | —（该版本无此键） |
| `trainer.validation_results_path` | —（该版本无此键） | `"validation_results.json"` | `"validation_results.json"` | `"validation_results.json"` |
