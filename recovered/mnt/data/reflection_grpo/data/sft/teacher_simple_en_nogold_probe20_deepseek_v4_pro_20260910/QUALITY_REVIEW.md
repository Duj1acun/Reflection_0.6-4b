# DeepSeek V4 Pro：无参考答案、20 题匹配对照

## 执行与控制变量

与上一轮 Flash 的 20 个问题和学生回答完全相同，0.6B/4B 各 10 条。已逐条断言：请求体只把 model 从 DeepSeek-V4-Flash 改为 DeepSeek-V4-Pro，没有其他差异。参考答案仍不发送，只有本地 samples/annotations 文件保存它供事后对照。

- 并行科技端点：https://llmapi.paratera.com/v1/chat/completions。
- model=DeepSeek-V4-Pro，thinking.type=enabled，reasoning_effort=high，max_tokens=16384，top_p=0.95，temperature=0.1（thinking 模式下不视为有效低温控制）。
- 最高并发 20，首条确认模型和 thinking 后发送其余 19 条，每题一次，无重试。
- 20/20 HTTP 200、20/20 finish_reason=stop、20/20 返回 thinking；网关响应名称均为 DeepSeek-V4-Pro。无截断，不独立断言后端权重版本。
- 19 条直接通过四字段 JSON 解析与结构检查，无多余字段、无代码围栏。剩下 1 条有完整最终文本，但 JSON 中反斜杠未正确转义，未通过解析。
- 输入 tokens=23,038，输出（含 thinking）=109,098，总计 132,136。Flash 对照为 109,859 总 tokens；两种模型的 token 数不等于同等费用，输入 tokenizer 计数也可能不同。
- 没有对原输出做 JSON 修复、字段补齐、答案改写或 thinking 提取；没有重试或启动训练。

## 关键对照结果

| 样本 | Flash 无参考答案 | Pro 无参考答案 |
|---|---|---|
| 蚂蚁位移 qwen3-4b_k2_01 | 最终值 29 正确，但 verified_states 错写 20 个四步循环 | 正确保留两步分组，续推使用 10 个四步循环，再加最后 41 cm，得到 29 |
| 五维超平面 qwen3-4b_k1_02 | 把成立的条件命题误判为错误；多一个 final_answer | 明确指出第一平面确实可使 120 变 240，错误在于后续平行平面不再穿过全部新区域；续推得到 480，但 JSON 转义失败 |
| 三角形高度 qwen3-4b_k2_04 | 16,384 个输出 token 全用于 thinking，最终 content 为空 | 7,492 个输出 token 内完成，准确定位正弦定理边角对应颠倒，完整续推得到 3 |

此前两道参考有问题的题，Pro 也给出了独立正确结果：三次函数运算得到 -148/7；sqrt(x)=x 保留 0、1 两个解。与无参考答案 Flash 相同，没有迁就旧标签。

## 五维超平面题的实际诊断

Pro 最终文本明确说：第一条 sum=0 平面穿过所有 120 个排序区域，所以确实会翻倍；加入它之后，sum=1 无法穿过 sum<0 的区域，不能继续把总数翻倍。这个诊断没有否定原本正确的条件命题。

其续推将空间分成 sum-zero 部分和总和方向，得到 120 个排序区域乘以四个总和区间，结果 480，数学上成立。

失败原因是纯格式：最终 content 使用了 JSON 不允许的未转义 LaTeX 反斜杠（例如字符串中的 \(），导致 JSONDecodeError。原始 content 已完整保留在 ANNOTATIONS.md 和 calls/qwen3-4b_k1_02.response.json；annotations.jsonl 中 annotation 为 null，不能因为人工读得懂就伪装成已通过 JSON 检查。

## 尚未解决的问题

- 三平方和题 qwen3-0.6b_k2_03 的诊断和续推正确，但 verified_states 仍纳入首次错误之后才显式写出的第三个约束。“事实正确”与“确实在错误前建立”还没有完全区分。
- 抽样题 qwen3-4b_k2_03 仍默认系统抽样得到 795，原题并未清楚给出该方法。换更强模型不能自动消除数据源的条件缺失。
- 种子题仍按明确提到的种子数求 4；原题询问拥有的总数与实际种植数之间的文字歧义依然存在。
- 抛物线题主动注明 M、N 必须不同，否则最小距离为 0；有条件的 6 不应脱离这个前提录入。

## 结论边界

这次固定 prompt、只换模型的试验，在蚂蚁状态、超平面诊断和三角形高度完成度这三个具体问题上，Pro 表现优于上次 Flash。这说明至少部分残余问题与模型输出可靠性有关，不能全部归因于提示词。

但每题每模型仅一次采样，不能据此宣称 Pro 的总体数学或标注合格率显著更高。19/20 是严格 JSON 结构通过数，不是数学准确率或最终可训练样本数。没有采用不适配自然语言结尾的旧自动评分来汇报正确率。

文件：ANNOTATIONS.md 为全部最终原文；annotations.jsonl 为全部 20 个请求的检查记录；requests.json 为实际无参考答案请求；audit_summary.json 为结构统计。没有导出筛选后的生产训练数据。
