# 不提供参考答案：20 题匹配试验

## 范围与控制变量

与 teacher_simple_en_probe20_deepseek_v4_flash_20260910 使用完全相同的 20 个问题及学生回答（0.6B、4B 各 10 条）。只做两项提示词修改：删除 GOLD_ANSWER 区块，将 reasoning 定义中的 ending with the gold answer 改为 ending with the final answer。没有加入其他约束、system 消息或 verdict。

样本哈希：a1d0e75e8f38e24d81cfc85d27ef1df66d2adce5e5d3bcd6d8a6215bc7fdf142。已验证将每条记录的 answer 改成哨兵字符串不会改变请求内容。因此实际请求不依赖参考答案。samples.json 和 annotations.jsonl 中仍保留本地参考答案供事后对照，它没有发送给 API。

请求参数与前次相同：并行科技 DeepSeek-V4-Flash，thinking.type=enabled，reasoning_effort=high，top_p=0.95，max_tokens=16384，temperature=0.1（thinking 模式下不能视作有效低温控制），最高并发 20，无自动重试。

## 完成情况

- 实际生成请求 20 次，20/20 HTTP 200，20/20 返回非空 thinking，响应模型名均为 deepseek-v4-flash（忽略大小写）。
- 19 条 finish_reason=stop，输出直接 JSON，无围栏；其中 18 条严格符合四字段结构。
- qwen3-4b_k1_02 多出 final_answer 字段，原样保留，没有删字段。
- qwen3-4b_k2_04（三角形高度题）finish_reason=length：16,384 个输出 token 全部用于 reasoning，最终 content 为空，没有可用标注。不是题目输入过长；该请求输入为 2,553 tokens。没有追加请求、从 thinking 提取答案或修补输出。
- usage：prompt_tokens=21,978，completion_tokens=87,881，total_tokens=109,859。
- 未启动全量或 SFT，Perfect 请求为 0。annotations.jsonl 保留 20 个请求的记录，其中失败条目的 annotation 为 null，不应当作 20 条成功训练样本。

## 两个重点问题的对比

| 样本 | 有参考答案时 | 不给参考答案时 |
|---|---|---|
| qwen3-0.6b_k1_03 三次函数运算 | 推得 -148/7，又猜测题目中的 -4 应为 -3，以解释参考 -21 | 正确推得 144+7a=-4，明确结束于 -148/7，没有修改题目 |
| qwen3-0.6b_k3_01 平方根等于自身 | 求出 0、1 后只选择参考答案 0 | 回代检验 0、1，完整保留两个解 |

这两条本次均有改善，支持“去掉参考答案有助于避免迁就错误标签”的判断，但每个条件每题只有一次采样，不能据此估计总体合格率或声称所有问题均解决。

## 仍需注意的内容问题

1. qwen3-4b_k2_01 蚂蚁题：最终位移计算正确，答案 29；但 verified_states 声称有“20 complete 4-move cycles plus the final 41 cm North”。41 步应为 10 个四步循环加最后一步，20 是两步组数，不是四步循环数。错误事实进入反思，即使答案正确也不能原样当作高质量监督。
2. qwen3-4b_k1_02 五维超平面题：续推的 120×4=480 正确，但 error_reason 把“新超平面若穿过全部现有区域就使区域数翻倍”本身判为错误。对超平面排列形成的凸开放区域，若确实穿过每个区域内部，该条件命题成立；真正问题是原学生假设后续平行超平面仍穿过所有已经分割出的区域。这条需要修正诊断，不能只看答案 480。
3. qwen3-0.6b_k2_03 三平方和题：诊断与终值 -4 正确，但 verified_states 仍保留了首次错误之后才显式写出的第三个约束；对“错误前已建立”的来源边界仍未完全遵守。修正解答的线性求解过程也较简略。
4. qwen3-4b_k2_03 抽样题：仍假定采用系统抽样而得到 795。去掉参考答案并不会自动消除原题未说明抽样方法的问题。

另一个积极变化：qwen3-0.6b_k1_02 抛物线题在得到不同点条件下的 6 后，主动说明如果允许 M=N，则最小值为 0，未无条件掩盖该歧义。

本次没有因第一人称、提前给出终值或必要修正公式等用户未要求的规则判违规；上述问题是实际数学/诊断缺陷、题意歧义、来源语义或请求格式问题。

## 查看文件

- 英文提示词：/root/reflection_grpo_clean/configs/sft/teacher_simple/annotation.no_gold.user.txt。
- ANNOTATIONS.md：最终输出和学生原回答，不展示内部 thinking。
- annotations.jsonl：20 条请求的结构化检查记录，不是筛选后的生产训练集。
- requests.json 与 calls/*.request.json：实际发送的请求，无参考答案区块，无密钥。
- audit_summary.json：结构统计。本次跳过不适合无 boxed 自然语言答案的旧自动评分，不把解析失败当成数学错误率。
