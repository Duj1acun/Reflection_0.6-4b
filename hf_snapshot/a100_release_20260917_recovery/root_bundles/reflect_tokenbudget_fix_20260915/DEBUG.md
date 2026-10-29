# REFLECT token 预算修复与 step 50 续训

## 故障

旧 REFLECT 在 2026-09-15 11:43（北京时间）第 62 步退出，最后完成第 61 步。具体错误来自 `compute_log_prob → rearrange_micro_batches`：

```text
AssertionError: max_token_len must be greater than the sequence length.
Got max_token_len=8192 and max_seq_len=8710
```

`save-frequency transition failed` 是旧启动控制器对训练子进程退出的外层报告。保存间隔已经成功改为 50；故障发生在训练序列的计算预算检查。

## 修复

- 新远端 bundle：`/root/reflect_tokenbudget_fix_20260915`，独立代码快照、Ray 目录和后台 session，GPU 1。
- `tokens_per_gpu` 从 8192 改为 16384，映射到 actor PPO 更新、rollout log-prob、reference log-prob 三项预算。
- 解析后的学习配置差异检查通过：其余学习配置与原 REFLECT 相同。总计 500 步，每 50 步保存和验证，actor LR 1e-6，64 题 × 8 初始采样，每次反思分支 4 个子节点，每轮生成 4096、三轮生成总预算 12288。
- 远端源代码增加了仅在 `FOURWAY_PROBE=1` 且 `FOURWAY_LONG_SEQUENCE_PROBE=1` 时启用的长度回归钩子。正式训练不启用该钩子。
- 输出仍为 `/data/reflection_grpo/experiments/reflect_math7500_0.6b_gpu1_branch4_stable_20260915`。旧正式训练日志及解析配置归档到 `runs/reflect/before_tokenbudget_fix_20260915/`；旧预检证据保存在新 bundle 的 `preparation.before/` 与 `probe.before/`。

## 验证

- CPU 环境、数据完整性、500 步数据日程预检通过。
- step 50 的模型、优化器、调度器/RNG、数据读取状态和 judge 状态文件齐全；附加状态和数据状态可正常反序列化。
- 在实际初始化的 native actor/FSDP/vLLM 环境中，8710 和 16384 token 均用旧 8192 预算复现断言，用新预算通过 log-prob、熵计算、前向、反向与有限梯度检查。
- 完整 64 × 8 训练批次的 native 反向预检通过；所有预检禁用 AdamW 参数更新，同时分配优化器状态和更新所需临时空间。
- 预检峰值分配约 69.7 GiB，峰值保留约 76.2 GiB；见 `GPU_VERIFIED.json`。

## 续训

使用 `resume_mode=resume_path` 从 `runs/reflect/global_step_50` 恢复，保留该 checkpoint，下一次更新为 step 51。关闭重复初始验证，下一次计划保存/验证为 step 100。

2026-09-15 15:44（北京时间）启动正式续训，launcher PID 194266。加载日志确认模型、优化器和附加状态路径均来自 step 50。15:56:42 检查确认已完成第 51 步真实参数更新：`branch/optimizer_steps=1.000`、`actor/grad_norm=0.324`、单步 674.069 秒；峰值显存分配 70.947 GiB、保留 77.992 GiB。GPU 1 继续活动，正式日志无异常。正式 worker 的 `FOURWAY_PROBE=0` 且未设置长序列测试开关。详见 `LAUNCH_VERIFIED.json`。同期 SCORE 已完成第 31 步并继续运行。

检查命令：

```bash
python3 /mnt/data/reflection_grpo/transfers/reflect_tokenbudget_fix_20260915/check.py
python3 /mnt/data/reflection_grpo/transfers/score_vocab_fix_20260915/brief_progress.py
```

后台已有正式训练流程，请勿重复启动。SCORE 继续使用其自身 GPU 0 和独立配置。
