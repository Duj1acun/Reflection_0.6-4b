# GPU 1：对齐后的 REFLECT Branch GRPO

当前配置：`/root/reflect_branch_gpu1_20260914/config.json`。
当前代码独立保存在本目录 `code/reflection_grpo_clean`，不会修改 GPU 0 的运行代码。
训练入口：`code/reflection_grpo_clean/training/rl/train_reflect_branch_grpo.sh`。
在此独立副本中，该脚本增加 FOURWAY_CONFIG 路径，使用统一数据、判分、验证及 checkpoint 选择。

与 GPU 0 相同：
- Qwen3-0.6B SFT phase2 checkpoint-268 合并权重（共享同一只读权重）。
- 7496 道训练题，完全相同的预处理数据和 500 步题目顺序；seed 1234。
- 每步 64 题，每题 8 个初始样本，共 512 个根样本；PPO minibatch 64，epochs 1，每步一次优化器更新。
- Actor lr 1e-6，weight decay 0，clip low/high 0.20/0.26，grad clip 1，entropy 0，KL 0。
- 最多三轮，每轮 4096 tokens；单条根到叶路径上限 12288，上下文 16384。
- 相同反思 prompt、奖励公式、格式要求、禁用 thinking、更新后的数学判分及故障处理。
- 初始验证和每 50 步验证 MATH500 + AMC23，pass@4 / mean@4；温度 .7、top_p .8、top_k 20。
- 验证不分叉、不按标准答案挑分支；两数据集宏平均 pass@4 选最佳 checkpoint。
- 每个实验单卡，Ray 12 CPUs，vLLM memory .5、max_num_seqs 32、batched tokens 8192，actor 动态 token 预算 16384。

方法差异：
- 训练每个反思点采样 4 条续推，下一轮可再次分叉；每个根最多 16 个叶子。
- 每个根总权重为 1；每次分叉权重除以 4。叶子的优势按同题加权均值和总体方差计算。
- 不使用主实验的直接回答均值 baseline 和难度相关 advantage floor。
- PPO loss 按根权重平均，补齐叶子的权重为 0；优化器仍每步更新一次。
- 加权统计看 branch/accuracy、branch/reward；通用未加权训练指标可能包含补齐行。
- fourway/generated_tokens 记录实际生成量（共享前缀生成计一次、已丢弃尾部也计入），mean_generated_tokens 按根样本数平均。

这不是旧 Hybrid-GRPO 的每个反思节点内局部 z-score；也不是计算预算相等的对照。
代码整合中仅增加配置接入、准确的生成成本统计，以及零权重补齐行不参与 actor 前后向的优化。
分支更新梯度守恒等回归测试与完整批次 GPU 无更新预检记录保存在 results 下。

运行状态以 SERVER_STATUS.json 和远端进程为准，不要重复启动。
日志：`results/launcher.log`、`results/runs/reflect/stage1.log`。
输出实际位置：`/data/reflection_grpo/experiments/reflect_math7500_0.6b_gpu1_branch4_3rounds`。
