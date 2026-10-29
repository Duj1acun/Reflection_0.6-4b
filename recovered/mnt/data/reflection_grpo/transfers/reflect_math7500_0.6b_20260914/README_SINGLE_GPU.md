# REFLECT 主实验：单卡、最多三轮

远端入口：`/root/reflect_math7500_0.6b_20260914/run_reflect.py`
远端当前配置：同目录 `config.json`。
环境：`/data/reflection_grpo/envs/reflect_math7500/bin/python`，入口自动选择。

- GPU 0，seed 1234；GPU 1 留给后续消融。
- 初始化：Qwen3-0.6B SFT phase2 checkpoint-268 合并权重。
- 全局 64 题 × 8 条轨迹 = 512；PPO minibatch 64（进入 worker 后乘 8），PPO epochs 1，每步一次优化器更新。
- Actor lr 1e-6，500 次更新；动态 token 微批及梯度累积，不因单卡缩小全局 batch。
- 每轮 4096 tokens，最多三轮（初始解题 + 最多两次反思续写），总生成上限 12288，上下文 16384。
- 当前反思奖励、优势计算和禁用 thinking 设置保留。
- 训练集 7496 题；MATH500 500 + AMC23 40；初始验证及每 50 步验证 pass@4，温度 0.7。
- 单个 Ray 实例配置 12 CPUs；Ray 临时目录按输出目录隔离，GPU 锁共用以避免同卡重复任务。

输出：`/data/reflection_grpo/experiments/reflect_math7500_0.6b_gpu0_main_3rounds`。
包内 `results` 链接指向当前主实验输出。
正式日志：`results/runs/reflect/stage1.log`；启动日志：`results/launcher.log`；PID：`results/launcher.pid`。
本文件描述配置；实际是否已启动请查看 `SERVER_STATUS.json` 和远端进程。

需要手动启动时（仅限尚未启动，不可重复执行）：

```bash
cd /root/reflect_math7500_0.6b_20260914
python3 run_reflect.py preflight --gpu
nohup python3 -u run_reflect.py train > results/launcher.log 2>&1 < /dev/null &
echo $! > results/launcher.pid
```

第二个实验可使用 `--config /绝对路径/消融配置.json` 指定独立配置，必须设置不同 GPU、输出目录和 Ray 临时目录；共享代码在训练期间不可覆盖修改。
主实验三轮生成预算比先前三个 baseline 更大，结果需要同时报告 token 成本；本实验不属于生成预算严格相等的对照。
