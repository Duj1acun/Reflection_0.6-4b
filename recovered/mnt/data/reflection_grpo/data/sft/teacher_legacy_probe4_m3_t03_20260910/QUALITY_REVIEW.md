# 旧 teacher prompt 的匹配四题试验

结论：三个真正存在推理错误的回答都得到数学正确的修复；另一个未化简完成的回答也得到正确补全。这个简短 prompt 已能组织诊断和续推，但仍有字段语义、首错定位、最终答案提前出现和格式问题，不等于四条都可以原样导入 SFT。

## 调用与可追溯性

- MiniMax-M3，temperature=0.3，top_p=0.95，max_completion_tokens=16384，thinking.type=adaptive，reasoning_split=true。
- 地址 https://api.minimaxi.com/v1/chat/completions；实际 4 次请求，最多并发 4，首个返回确认接口后启动其余三个，无重试，无独立解题或二阶段 API 调用。
- 使用与 v2 四题试验完全相同的 samples.json；哈希为 3ed242f7340caf082f8139d99275b693434ab81000d3db2648c6768bdabbd805。
- 系统消息来自旧 command_teacher 的常量，用户消息直接调用现存 _teacher_prompt(row)；没有添加示例、schema、thinking 文字指令或其他约束。请求快照在 requests.json 和 calls/*.request.json，函数源代码快照在 manifest.json。
- 4/4 HTTP 200、M3、finish_reason=stop、非空 thinking，无截断。3 条直接 JSON，1 条仅外包 JSON 围栏；去除围栏后 4 条均满足旧版五字段结构。
- 原样 reasoning 的本地答案评分为 4/4；人工与独立符号/数值检查确认终值分别为 325、1、-2、1。固定点题的答案提取器返回一段等式而非纯数字，但本地评分通过；最终值 1 另经人工验证。
- 3/4 有 boxed，固定点题缺少 boxed；未追加答案、未修复 JSON、未从 thinking 中挖取答案。
- usage：prompt_tokens=4732，completion_tokens=6472，total_tokens=11204。
- 两个原采样数据文件哈希与冻结记录一致。Perfect API 请求 0；没有启动全量标注、学生推理或训练，没有导出生产训练集。

## 分题结果

| 样本 | 续推结果 | 主要结论 |
|---|---:|---|
| 0.6B 合并数字/糖果 | 325 | 诊断、平方和证明正确；方向末尾有无关的“最大化初始状态”措辞 |
| 0.6B 对数固定点 | 1 | 解法正确；漏掉更早的平移项丢失，把新坐标写成已知状态，缺 boxed |
| 4B 平移基底系数和 | -2 | 诊断与续推较好；正确令 x+2=1 |
| 4B 分式化简 | 1 | 正确指出未化简到底，没有捏造代数错误；方向直接给出最终值 1 |

旧 prompt 未禁止 new_direction 出现最终答案，未要求第一人称，也没有 incomplete 分类。因此这些是与预期 SFT 用途的适配问题，不能冒充本轮 prompt 的违规。必要的中间修正关系和公式本身不作为拒收理由。

固定点题的 verified_states 与首错定位则确实没有完全遵守旧 prompt 本身的定义。把新推导的坐标写入 new_direction 合理；写成“学生错误前已经建立”的状态则不准确。

## 与前次 v2 的公平比较

前次 v2 在同组三个真正错误样本上也给出了三个正确终值，另将分式题识别为 incomplete。本次旧 prompt 给出了第四题的完整化简，但不能据此宣称数学准确率从 75% 升到 100%。两版在错误样本的求解上均为 3/3；区别主要在分类、约束遵守、措辞和补全方式。这只是每题一次的四题试验，不足以判断哪版全量质量更高。

本次不输出“严格只通过几条”的单一数字；逐项区分数学、诊断、格式与可选训练约束，避免套用前次过严边界。若继续改进，建议以简短旧版为基础，只对自述人称、首错/状态语义和最终答案格式做小幅调整，保留有用修正公式。改动尚未实施。

## 长度

本地 tokenizer、no-thinking、当前训练模板下，Phase 1/Phase 2 长度分别为 748/547、1119/524、785/463、1735/664；全部在 2048 内，更未超过 4096。reasoning 本体仅 235、185、139、268 tokens。本次问题不是长度截断。

完整最终标注和学生原回答见 SAMPLES_REVIEWED.md；单独复核意见见 manual_review.json；只读数学与长度检查见 automatic_checks.json。原始响应独立保留。
