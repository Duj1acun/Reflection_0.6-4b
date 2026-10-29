# REFLECT+ 分段采样上下文修复

原来的反思续写使用 `原始 prompt + 当前反思标签 + 换行`，但 RL 把各段拼成
完整历史再计算概率，续写 token 因而看到了采样时不存在的旧推理文本。

修复后保留原有生成方式、轨迹奖励、GRPO 分组和解耦优势。每段记录实际使用的
prompt token IDs（包含换行和实际左截断结果），以及保留下来的生成 token IDs。
Actor 在计算 old/new log probability 时，把分段输入临时展开成独立序列，按各自
prompt、attention mask、position IDs 前向，然后把概率映射回原轨迹的 token 位置。

完整拼接序列仍供奖励判分、loss mask 和日志使用，不再作为反思轨迹的 policy
forward 上下文。每条轨迹仍只占一个 batch 样本，`seq-mean-token-mean` 仍按
整条轨迹的有效 token 平均；反思更多不会自动增加样本权重。首段错误推理与反思
标签仍按原有 `TRAIN_FULL_CHAIN` / mask 设置参与学习。将前缀 loss mask 设为零
也不会改变后续 token 的采样上下文。

辅助张量保持第一维为轨迹，支持 DataProto 重排、切分及 FSDP 分发。没有对应
反思段的槽位使用零损失的有效占位序列，避免全 padding 的 FlashAttention 输入，
并保持各 rank 的 forward/backward 调用结构一致。动态 batch 和负载均衡计入
每段重复的 prompt，防止按旧拼接长度低估计算 token 数。

原来的 `train_reflect_plus.sh` 可直接使用，无需新增开关。建议为修复后的正式
对照实验使用新的 `EXP_NAME`，从相同 SFT 初始化重新训练。旧 checkpoint 在格式上
仍可加载，但继续训练不能消除已经发生的旧梯度更新，也不能把旧实验标为修复后实验。
分段会重复处理 prompt，速度和显存占用可能与旧实现不同，应记录实际训练成本。

回归验证：

```bash
python -m pytest tests/test_trajectory_context.py -q

# 显式选择空闲 GPU，额外验证 FlashAttention/remove-padding 和一次 PPO 更新。
CUDA_VISIBLE_DEVICES=1 RUN_CUDA_CONTEXT_TEST=1 \
  python -m pytest tests/test_trajectory_context.py -q
```

测试包括原始 collector 的一至三轮生成、截断 prompt、full-chain 开关、被丢弃
的生成尾部、混合直接/反思轨迹、逐 token 概率、独立分段梯度参照、轨迹权重、
DataProto 切分重排、缺失上下文拒绝处理，以及实际 actor 更新。
验证使用随机初始化的小型 Qwen；不等于完成了全规模训练或多 GPU Ray/FSDP 集成验证。
