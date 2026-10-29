# 四路反思分支 GRPO

启动（在与原 REFLECT+ 相同的 Python 环境中）：

```bash
bash /root/reflection_grpo_clean/training/rl/train_reflect_branch_grpo.sh
```

先检查实际命令，不启动 Ray/GPU、不合并模型、不创建训练输出：

```bash
bash /root/reflection_grpo_clean/training/rl/train_reflect_branch_grpo.sh --dry-run
```

模型、数据、GPU 等仍使用原脚本的环境变量，例如：

```bash
CUDA_VISIBLE_DEVICES=0 MODEL_PATH=/path/to/merged_sft \
TRAIN_FILE=/path/to/train.jsonl TEST_FILE=/path/to/test.jsonl \
bash /root/reflection_grpo_clean/training/rl/train_reflect_branch_grpo.sh
```

## 参数对齐

新脚本委托 `train_reflect_plus.sh` 组装命令，不复制另一套超参数。
两者读取相同的项目 `.env`（如存在）和环境变量覆盖。
普通调用原脚本仍走原来的 REFLECT+ 路径，原有 FOURWAY_CONFIG 分支保留。
新分支脚本拒绝 FOURWAY_CONFIG，以免误启动另一条实验流程。

| 参数 | 共享默认值 |
| --- | --- |
| 初始模型 | phase2_merged_hf SFT 模型 |
| 训练/验证数据 | limo_rgrpo.jsonl / math500_rgrpo.jsonl |
| 每步题数 / 每题初始样本 | 64 / 8，共 512 个根样本 |
| PPO mini / micro | 64 个题对应的根样本 / 每卡 4 条；dynamic batching 开启 |
| 学习率 / weight decay / grad clip | 1e-6 / 0 / 1 |
| PPO epochs / clip low-high | 1 / 0.20–0.26 |
| 损失聚合 | seq-mean-token-mean |
| 熵系数 | 继承共享 Hydra 默认值 0.001，分支损失按权重计算 |
| prompt / 每轮 response / max_iters | 1024 / 1024 / 3 |
| 采样 temperature / top_p / top_k | 1 / 1 / -1 |
| 训练步数 / 验证间隔 / 保存间隔 | 1000 / 10 / 50 |

算法差异：每次允许继续的反思后采样 `REFLECTION_BRANCHES=4` 个子分支；
使用根样本权重守恒的组内 GRPO，**不使用原版的 decoupled baseline 或 advantage floor**。
最终 reward 公式、格式要求和已有惩罚保持不变；没有暗中改为纯二元正确性奖励。
这是整条轨迹联合训练，不是反思/续推分阶段训练，也不是能力解耦方案。

默认实验名为 `reflect-branch-grpo-qwen2.5-math-1.5b`。
`BRANCH_EXP_NAME` 优先于 `EXP_NAME`；手动指定输出目录时请勿与正在运行的旧实验共用。

## 分组与权重

- 一次分叉时，8 个根中 m 个触发反思，实际叶子数为 `(8-m)+4*m`。
- 默认保留 `MAX_ITERS=3`，所以子分支也可能再次反思并分叉。
  每个根最多 16 个叶子；这不是固定只扩展一次。只要一次分叉可设置 `MAX_ITERS=2`，
  但这会改变与默认 REFLECT+ 的轮数对齐。
- 每个根初始权重为 1，每次分成 K 份后子分支权重为父权重除以 K。
  同一题全部叶子按此权重计算均值、总体方差以及 PPO/熵损失。
  全同正确性结果时是否跳过标准差，继承原脚本开关。
- PPO minibatch 仍按根样本数计数，不会因为叶子增多而增加每步 optimizer 更新次数。
  共享前缀仍会从多个后续结果获得梯度，但不会简单按叶子数放大根样本总权重。
  保留逐轨迹 token-mean，因此不同叶子长度也会影响具体 token 的系数。
- 不同根补齐相同叶子槽数供多卡分发；补齐行权重为 0，不参加优势统计、PPO 或熵损失。
  分发保留整个根的连续块；不允许普通 adjust_batch 随机复制或删除叶子。
- 各段 log probability 使用该段采样时的真实输入上下文。
  拼接整条路径仅用于 reward 和 token 位置映射，不作为续推段的策略输入。

## 验证与资源

验证仍使用原来的单路贪心反思流程，不分四路、不挑选最优答案。
模型路径必须存在，或原脚本能够根据 SFT adapter 合并模型；训练/验证数据也需自行准备。

超参数对齐 **不等于计算预算对齐**：分支增加生成 token、前后向工作量和显存占用。
保留原脚本 GPU 空闲显存检查；它不是对新分支模式峰值显存的保证。
当前实现支持原脚本的 FSDP、SP=1、vanilla PPO、seq-mean-token-mean，禁用 KL 惩罚。
修改 batch 或卡数后，需保证根样本 minibatch 和微批次的整除约束；不符合时会报错。

训练效果主要查看 `branch/accuracy`、`branch/reward`（根权重平均），
以及 `branch/root_count`、`branch/leaf_count`、`branch/padding_count`。
原训练器的通用 `critic/*`、`reflection_grpo/*` 等未加权统计可能包含补齐行，
不能用它们代替分支模式的根加权指标；泛化效果以未分支的验证指标为准。

## 测试

`tests/test_branch_grpo.py` 覆盖单层/递归分叉、叶子权重、实际采样上下文、
加权优势、补齐 reward、根块分发、四倍复制梯度不放大、熵项以及脚本参数一致性。
这些是小模型/模拟生成回归测试，不代表已完成真实 vLLM 多卡训练或效果验证。
