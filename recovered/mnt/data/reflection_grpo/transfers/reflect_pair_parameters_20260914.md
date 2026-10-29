> 当前版本更新：两个旧实验已停止，由 `_sftprompt_nogate` 新实验替代。训练及验证均使用 SFT 原始 system prompt（不另加 boxed 指令），`TRAJ_GRPO_REQUIRE_REFLECTION_FORMAT=0`。其他训练、采样和奖励参数保持不变。新配置分别为 `/root/reflect_main_sftprompt_20260914/config.json`、`/root/reflect_branch_sftprompt_20260914/config.json`。下方原始参数表保留作为历史记录。

# 两个 REFLECT 实验的实际配置

来自目标服务器两个 config.json 和实际解析的 preparation/reflect.stage1.resolved.json。
运行状态单独查看各实验 SERVER_STATUS.json；本清单不代表训练完成。

| 方法项 | GPU 0 REFLECT 主实验 | GPU 1 Branch GRPO |
|---|---|---|
| 物理 GPU | A100 80GB GPU 0 | A100 80GB GPU 1 |
| 初始根样本/步 | 64×8=512 | 64×8=512 |
| 每反思点续推数 | 1 | 4 |
| 最大解题轮次/反思续写次数 | 3 / 2 | 每条路径 3 / 2 |
| 每个初始样本最大叶子数 | 1 | 16 |
| 优势基线 | 同题直接回答奖励均值；没有直接回答时用全组均值 | 同题所有叶子的根权重加权均值 |
| 标准差 | 全组样本标准差 | 同题加权总体标准差 |
| 同题全对/全错 | 跳过标准差除法，保留中心化 | 同左，忽略零权重 padding |
| 难度相关反思优势保护 | 开启，floor0=-0.5 | 不使用 |
| PPO 损失权重 | 每条初始轨迹等权 | 根样本总权重 1，每次分叉除以 4 |
| 验证时分叉 | 不分叉 | 不分叉 |

| 共同项 | 设置 |
|---|---|
| 初始化 | Qwen3-0.6B，SFT phase2 checkpoint-268 合并权重 |
| 训练类型 | 全参数更新，LoRA rank=0 |
| 训练集 | MATH7500 清洗后 7496 题，数据与题序一致 |
| seed | 1234 |
| 训练步数 | 500 次参数更新 |
| 每步题数/每题初始采样数 | 64 / 8 |
| PPO minibatch | 配置 64（题数单位），worker 展开为 512 根样本 |
| PPO epochs | 1 |
| 优化器更新/step | 1 |
| Actor 优化器/学习率 | AdamW / 1e-6 |
| betas / weight decay | (0.9, 0.999) / 0 |
| 学习率计划 / warmup | constant / 0 |
| gradient clip | 1.0 |
| PPO clip low/high / dual clip | 0.20 / 0.26 / 3 |
| loss aggregation | seq-mean-token-mean |
| entropy coefficient | 0 |
| actor KL loss / reward KL | 均关闭，系数 0 |
| critic | 无 |
| 训练 temperature / top_p / top_k | 1.0 / 1.0 / -1 |
| thinking | 模板 enable_thinking=False，禁止生成 <think> 和 </think> |
| 最大 prompt / 每轮生成 | 2048 / 4096 tokens |
| 单条路径总生成上限 / 模型上下文 | 12288 / 16384 tokens |
| 奖励风格 / phi | efficiency / 4 |
| 免费反思次数 | 1 |
| 正确直接 / 正确一次反思 / 正确两次反思 | 正常结束且格式合法时 1 / 1 / 0.9375 |
| 错误直接 / 错误反思 | -0.25 / 0 |
| 反思格式要求 | So far I have ... However ... |
| 额外线性反思惩罚（正确/错误） | 0 / 0 |
| 训练完整保留轨迹 / 仅最后反思 | 开启 / 关闭 |
| 判分 | 同版 fourway-math-v5-20260914 实现 |
| 判分异常 | 重试后跳过未更新的整批；连续 3 次失败停止 |
| 验证集 | MATH500 500 题 + AMC23 40 题 |
| 验证时机 | step 0 及每 50 步 |
| 验证每题样本/指标 | 4 / pass@4、mean@4 |
| 验证 temperature / top_p / top_k / seed | 0.7 / 0.8 / 20 / 1234 |
| 验证 batch | 128 题 |
| checkpoint | 每 50 步保存；保留 best + latest |
| best 选择 | 两数据集等权宏平均 pass@4，平分看 mean@4，再看更早 step |
| FSDP / TP / SP | FSDP 单卡实际 NO_SHARD / 1 / 1 |
| 计算精度 | BF16 |
| gradient checkpointing / remove padding | 开启 / 开启 |
| 参数、优化器 CPU offload | 均关闭 |
| dynamic batch / 配置 microbatch | 开启 / 每卡 2 条（动态模式按 token 分批） |
| actor / logprob token budget | 每卡 16384 / 16384 |
| logprob microbatch | 每卡 2 |
| vLLM | 0.8.5，V0 |
| vLLM GPU memory utilization | 0.5 |
| vLLM max_num_seqs / max_num_batched_tokens | 32 / 8192 |
| CUDA graph / chunked prefill | 开启 / 开启 |
| Ray 资源 | 每实验 12 CPUs，object store 2 GiB |

GPU 1 的分叉数和生成成本不是 GPU 0 的等预算对照；12288 是单条根到叶路径上限，不是整棵树总 token 上限。
相同根数和优化器步数不能消除不同生成预算的影响，应同时报告实际生成 token 和 GPU 小时。

主实验输出：`/data/reflection_grpo/experiments/reflect_math7500_0.6b_gpu0_main_3rounds`。
分支实验输出：`/data/reflection_grpo/experiments/reflect_math7500_0.6b_gpu1_branch4_3rounds`。
