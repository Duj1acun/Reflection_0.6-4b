# 与 GRPO / REFLECT+ 对齐的 PAG、SCoRe 训练

`qwen1p5b_pag.sh` 和 `qwen1p5b_SCoRe.sh` 现在调用 `aligned_train.sh`。
参考设置来自 `/root/reflection_grpo_clean/training/baselines/train_grpo.sh`
和 `training/rl/train_reflect_plus.sh` 的默认值；运行时环境变量可以覆盖。
原有 `local_2gpu.sh` 未修改，仍是旧的 Instruct / MATH7500 配置，不能作为本次对齐入口。

| 项目 | 对齐后的默认值 |
|---|---|
| 初始模型 | 与 GRPO 相同的 Qwen2.5-Math-1.5B；不加载反思 SFT |
| 数据 | 从参考 GRPO 的 LIMO / MATH500 JSONL 转换，保留题目、答案、prompt、顺序 |
| 每步问题数 / 每题轨迹数 | 64 / 8 |
| PPO mini-batch / micro-batch | 64 / 每卡 4；动态 batch 开启 |
| Actor LR / weight decay / grad clip | 1e-6 / 0 / 1 |
| PPO epochs / clip low, high | 1 / 0.20, 0.26 |
| Actor loss 聚合 | seq-mean-token-mean |
| 训练采样 | temperature=1、top_p=1、top_k=-1 |
| 验证 | 每题 1 条 greedy 轨迹；保留各方法两轮流程 |
| Prompt / 模型上下文上限 | 1024 / 4096 |
| PAG 生成上限 | 1024 × 3 段：首次解题、验证、可能的重新解题 |
| SCoRe 生成上限 | 1536 × 2 段：首次解题、纠错解题 |
| PAG 训练步数 | 1000 |
| SCoRe 训练步数 | 第一阶段到 500；加载该 checkpoint，第二阶段继续到累计 1000 |
| 保存 / 验证频率 | 每 50 / 10 步 |
| 新实验 resume | disable；SCoRe 阶段切换显式加载第一阶段 checkpoint |
| 默认资源 | 单卡，actor / critic CPU offload，vLLM memory utilization=0.45 |

SCoRe 的 500/500 分配沿用原脚本两个阶段各占总训练进度一半的比例。
可用 `SCORE_STAGE1_STEPS` 调整；`TOTAL_TRAINING_STEPS` 始终表示两个阶段的累计终点。
第一阶段末尾会保存 checkpoint；第二阶段要求 actor 和 critic 都存在，不会静默从头训练。

## 数据与模型

本次检查时，参考目录下没有实际的 Math-1.5B 模型和 RL JSONL 文件。
PAG 目录现有的 Instruct 模型与 MATH7500 数据不会自动替代它们。
请使用你实际 GRPO 实验的模型和数据路径：

```bash
cd /root/Policy-As-GenVerifier
export PYTHON_BIN=/mnt/data/conda_envs/pag-score/bin/python

# 将路径替换为实际 GRPO 使用的 JSONL，输出默认在 datasets/ 下。
"$PYTHON_BIN" quick_start/prepare_aligned_data.py \
  --train /path/to/limo_grpo.jsonl \
  --val /path/to/math500_grpo.jsonl

export MODEL_PATH=/path/to/Qwen2.5-Math-1.5B
bash quick_start/qwen1p5b_pag.sh
bash quick_start/qwen1p5b_SCoRe.sh
```

转换器只适配格式，不筛选、不重排、不改写 prompt，不覆盖已有输出。
也可以直接设置 `TRAIN_FILE` / `TEST_FILE`，但必须是含有 `prompt`、
`reward_model.ground_truth`、`data_source` 的 parquet。

## 检查与覆盖

```bash
# 不加载模型、不建 Ray 集群、不启动训练。
DRY_RUN=1 bash quick_start/qwen1p5b_pag.sh
DRY_RUN=1 bash quick_start/qwen1p5b_SCoRe.sh

# 独立运行第二阶段：必须有匹配的第一阶段 checkpoint。
SCORE_STAGE=2 bash quick_start/qwen1p5b_SCoRe.sh

# 使用不同总步数，仍由两个阶段共享预算。
TOTAL_TRAINING_STEPS=200 SCORE_STAGE1_STEPS=100 \
  bash quick_start/qwen1p5b_SCoRe.sh
```

长度、路径、步数请使用环境变量覆盖，以便启动前检查与训练设置一致。
其他 Hydra 参数可以追加在命令末尾。不要用尾部 Hydra 参数另行覆盖阶段步数、
恢复路径、轮数、长度或输入路径，否则会绕过相应启动检查。

`check_aligned_inputs.py` 使用实际 tokenizer 和 rollout 中的反馈模板，核对
训练／验证数据的最长有效 prompt。多轮反馈模板会额外占用上下文，因此
1024 prompt + 3072 生成 token 并不一定能放进 4096；如果实际题目过长，检查会报错，
给出安全的每段生成长度，不会静默截掉历史。改变生成预算后应同步记录比较条件。

PAG、SCoRe 比 GRPO 多一个 critic，SCoRe 第一阶段还需要 reference model。
上述单卡资源设置只对齐配置，不代表这台机器能承载；尚未验证完整训练的显存／内存。
已有本机记录显示双卡全部 CPU offload 可能超过 RAM，使用双卡时可参考原有运行记录，
通过 `NGPUS_PER_NODE=2 CUDA_VISIBLE_DEVICES=0,1 OFFLOAD=False` 等调整资源设置。

## 保留的方法差异与验证范围

- 保留 GAE + critic；PAG 的 role normalization、验证奖励和 reward shaping；
  SCoRe 的两阶段流程、第一阶段 turn-1 KL=0.1、第二阶段 reward shaping。
- Critic LR=2e-6、critic weight decay=0.01 保留原实现，GRPO 无对应 critic 参数。
- 保留 PAG/SCoRe 原生数学判分后端，其答案提取和容错行为与 REFLECT+ 不完全相同。
  正式结果应另用统一判分器比较最终答案正确率，不能直接比较复合 reward。
- 生成 token 上限相同不等于计算成本相同；应报告实际生成 token、训练耗时，
  并说明 REFLECT+ 额外的 SFT 数据与成本。
- 两个仓库的随机数实现仍不同（PAG/SCoRe vLLM 使用 seed=42）；对齐参数不保证
  相同随机样本或相同数据顺序，正式实验应报告种子并考虑多次运行。
- 修复了 actor 对 `loss_agg_mode` 的漏传，使配置作用于策略损失；同时把
  `max_num_seqs` 传给两个 vLLM 引擎，使设定的并发上限实际生效。

已进行 shell 语法、Hydra 配置解析、策略损失数值和输入检查验证；未启动正式训练。
