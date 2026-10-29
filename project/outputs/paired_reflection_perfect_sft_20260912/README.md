# 同题配对的反思 / perfect SFT 数据

构造日期：2026-09-12。全程离线，没有新增 API 调用，没有启动训练。保留原始批次，不覆盖原数据。

## 数量

| 模型 | split | 反思 | 同题 perfect | 四次全对 perfect | perfect 合计 |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | train | 1378 | 1096 | 67 | 1163 |
| Qwen3-0.6B | eval | 167 | 131 | 7 | 138 |
| Qwen3-4B | train | 1493 | 1484 | 200 | 1684 |
| Qwen3-4B | eval | 166 | 165 | 22 | 187 |

0.6B 总计反思1545、perfect1301、混合2846条；4B 总计反思1659、perfect1871、混合3530条。train/eval 沿用原始题目划分，无规范化同题跨集合，两个学生模型分别构造。

## 选样

1. 完整保留已经通过自动校验的反思记录，仅补充 type=reflection。
2. 优先取同题、同split的原学生正确回答，每题最多一条perfect；不要求正确轨迹与错误轨迹逐步对齐。
3. 对同题已有四次采样中的正常结束正确回答，按文本字符数从短到长尝试。必须重新通过本地数学答案匹配，且没有筛查规则命中的显式自我纠错表述、think/reflection标签或长段落三次重复。保留原文，不截短、改写或补造正确回答。
4. 同题配对后，0.6B 加入全部可用四次全对样本（train67/eval7）；4B 补充train200/eval22条四次全对样本，按source/domain比例分层、固定seed=42抽样。
5. 相对旧perfect导出，0.6B有286条、4B有812条使用了同题另一条原始正确轨迹。

质量筛查后，两模型train同题perfect均比原计划少4条；0.6B eval另少1条。未通过者没有用无关题目补齐，也没有删除对应反思记录。unpaired_reflection_questions.jsonl 列出未配对题目，包括原先就没有可用perfect的题目。

这些筛查是局部启发式和最终答案复核，不是完整数学证明审核，也不能保证不存在隐含的自我纠错或碰巧答对。保留semantic_reviewed=false。perfect_selection_audit.jsonl记录尝试过的候选和筛查结果；passed_candidate_screen表示通过候选筛查，不代表四次全对候选一定进入最终分层抽样结果。

## 文件

每个模型目录包含：

- reasoning_outputs.jsonl：完整反思集，含split字段。
- reflection_data_prepared_perfect.jsonl：本次选出的完整perfect集，含split字段。
- reflection_train.jsonl / reflection_eval.jsonl：拆分反思集。
- perfect_train.jsonl / perfect_eval.jsonl：拆分perfect集。
- phase1_mixed.jsonl：两类记录混合，固定seed打乱，保留type和split。
- data.env：指向该模型两份完整数据文件，并设MAX_LEN=4096、关闭Qwen3模板thinking、PERFECT_OVERSAMPLE=1；不设置模型权重或启动训练。
- summary.json / length_audit.jsonl：数量与逐条长度统计。
- perfect_selection_audit.jsonl / unpaired_reflection_questions.jsonl：筛查、缺失配对记录。

现有train_phase1.py/train_phase2.py分别读取DATA_PATH和PERFECT_PATH，应使用上述两份完整文件，不要将phase1_mixed.jsonl当作DATA_PATH再重复加载perfect。

在加载对应模型的原配置之后，再source本目录相应模型的data.env，确保新数据路径和长度设置生效。Phase2是否启用64:19:17混合由训练配置决定，本数据构造未改变该开关。

## 长度与监督量

使用各自本地Qwen3 tokenizer、实际训练system提示词、enable_thinking=false，分别检查Phase1反思、Phase2续推和perfect的完整拼接。

- 最大完整长度：0.6B为2920 tokens；4B为3366 tokens。
- MAX_LEN=4096时，本次所有检查序列完整保留，均有有效监督。
- MAX_LEN=2048时，0.6B有59个检查序列被截断，其中1个没有监督token；4B为278个、其中33个没有监督token。这里是按训练用途计算的序列数。
- Phase1训练集不截断时，perfect占有效标签token总量约78.8%（0.6B）和86.0%（4B）。这是标签数量比例，不是声称实际优化梯度占比相同。perfect整段解答通常比反思段长，因此样本数接近不代表训练权重接近。正式训练若需要平衡两种行为，应另行确定loss权重或采样策略；本次按用户确认配方保留数据。

## 重现与验证

构建器：scripts/data/build_paired_sft_sets.py。输入：outputs/formal_reflection_sft_cut_mixed_2000each_deepseek_pro_thinking32k_20260912。使用--input和--output参数，输出目录必须不存在。

manifest.json记录原数据、源采样、构建器与输出JSONL哈希；verification_summary.json记录最终检查。已核对反思正文未变、perfect原文对应源trajectory、模型归属正确、每题最多一条perfect、训练验证划分无交叉及4096下无截断。数学语义尚未全量人工审核。
