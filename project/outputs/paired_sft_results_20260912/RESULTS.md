# 两模型两阶段SFT结果

两个模型均正常完成，final adapter与各阶段验证loss最优checkpoint逐参数/文件核对一致，Phase2配置确认加载各自Phase1 final。未运行独立自由解题或自主反思benchmark。

| 模型 | 阶段 | 最优step | 最优eval_loss | 对应token accuracy | 最后step | Trainer耗时 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| qwen3-0.6b | phase1 | 300 | 0.529054 | 89.67% | 474 | 1.34小时 |
| qwen3-0.6b | phase2 | 268 | 0.499436 | 87.71% | 268 | 0.99小时 |
| qwen3-4b | phase1 | 400 | 0.387084 | 92.18% | 594 | 3.59小时 |
| qwen3-4b | phase2 | 290 | 0.413300 | 90.14% | 290 | 2.02小时 |

完成时间（北京时间）：0.6B为2026-09-12 14:14:12；4B为2026-09-12 17:33:02。两张GPU已释放。

Phase1配置3 epochs，Trainer实际约2.98/2.99；Phase2配置2 epochs，实际约1.99。梯度累积与步数取整导致不足整轮，不是异常中断。

## 观察

- Phase1后期验证损失回升：0.6B最优step300=0.529054，最后step474=0.534932；4B最优step400=0.387084，最后step594=0.403743。符合轻度过拟合/后期收益回退的表现，选择较早checkpoint避免了直接使用最后权重。
- Phase2两个模型最低验证loss均在最后一步，分别为0.499436和0.413300；本轮观测范围内仍有小幅改善。
- Phase1结束、加载最优权重后的反思格式抽检：两模型均15/20=75%。
- Phase2在step100和step200的反思格式抽检：两模型均17/20=85%。最后step268/290没有执行该格式抽检，不能把85%写成最终adapter的测量结果。
- 两阶段抽检样本顺序不同，不据75%到85%断言能力提升。该指标仅检查给定错误前缀后reflection标签闭合，不审核反思数学内容或自主触发。
- token accuracy是teacher forcing下的下一token准确率，不是题目答对率。两阶段的数据/监督目标不同，loss不能直接跨阶段比较；两模型验证题目集合也不同。
- 未测基座在同一任务上的loss或自由解题准确率，不能声称SFT提高了多少数学或自发反思能力。

## 产物

最终LoRA adapter目录：

- /mnt/data/reflection_grpo/artifacts/paired_sft_20260912/qwen3-0.6b/phase2/final
- /mnt/data/reflection_grpo/artifacts/paired_sft_20260912/qwen3-4b/phase2/final

加载时需要搭配对应原始Qwen3基座；这些是adapter而不是已合并的完整模型。本轮chat template使用enable_thinking=false，首轮对照评测应保持相同模板设置。

完整数字见summary.json，曲线见validation_loss.png。
