# 用户指定的最简英文 prompt：20 题运行记录

已完成 20 次生成请求，每题一次，无重试。此次只使用用户给定提示词的英文翻译，不附加 system 消息、verdict、第一人称、答案泄露限制、boxed 要求或额外示例。历史试验与原入口没有被覆盖。

## 参数与结果

- 端点：https://llmapi.paratera.com/v1/chat/completions。
- 请求模型：DeepSeek-V4-Flash；鉴权模型列表明确提供此名称，20 条响应名称均对应该模型。这里只确认网关报告的模型名，不独立证明后端权重版本。
- thinking.type=enabled，reasoning_effort=high，top_p=0.95，max_tokens=16384。沿用已有 Flash 试验的 temperature=0.1，但 DeepSeek 官方说明 thinking 模式不使用 temperature，不能据此宣称低温生效。
- 最高并发 20；首条确认模型和 thinking 返回后发送其余 19 条。
- 样本与前面 20 题试验逐字节相同，0.6B、4B 各 10 条，共 20 个不同问题。samples.json SHA256：a1d0e75e8f38e24d81cfc85d27ef1df66d2adce5e5d3bcd6d8a6215bc7fdf142。
- 20/20 HTTP 200，20/20 finish_reason=stop，20/20 返回非空 thinking，没有截断。
- 20/20 最终 content 可直接解析为 JSON，没有代码围栏。19/20 严格符合四个字符串字段；qwen3-0.6b_k2_03 多出 final_answer 字段，原样保留，未删除或重试。
- usage：输入 22,123 tokens，输出（含 thinking）78,207 tokens，合计 100,330 tokens。
- Perfect API 请求 0，没有启动全量标注、学生采样或 SFT 训练。

## 内容检查发现

简短提示词仍能给出具体诊断。例如三角形高度题 qwen3-4b_k2_04 准确发现了更早的正弦定理边角对应错误，随后算出高度 3。五维超平面题正确用 120 个排序区域、每个被切成 4 部分得到 480，没有额外引入“必须一般位置”的错误条件。

但不能仅按“结尾出现标准答案”验收。以下两条尤需隔离：

1. qwen3-0.6b_k1_03：由题目可得 144+7a=-4，所以 a=-148/7，原参考是 -21。模型求出了 -148/7，随后又猜测题目输出应为 -3，以解释 -21。这段改变题目条件的猜测不应作为标准修正解答使用。
2. qwen3-0.6b_k3_01：sqrt(x)=x 同时有 0、1 两个解。模型推导出两解后，只选择参考答案 0，没有根据排除 1。此处反映了参考答案不完整和朝参考答案靠拢的问题。

另有原题条件歧义，如抛物线题是否明确 M、N 不同，以及抽样题是否明确为系统抽样。本次没有改动原题、标签或生成内容。正确前缀的来源也仍需审核：例如三平方和题 verified_states 保留了错误步骤之后才写出的第三个约束，不能仅因这个约束正确就认为满足“错误前已建立”的定义。

## 自动判分的限制

本次按用户要求未要求 boxed，20 条 reasoning 均没有 boxed。旧答案提取器抓取的是整段自然语言，而非干净终值，导致保存的原始自动分数均为 0。这不是数学正确率为 0%，也不能用这些分数比较模型。

automatic scorer 的原始数值保留在 annotations.jsonl 与 audit_summary.json 中，并添加了警告；没有通过补框、取最后一个数字或修改生成内容把分数变好。本报告不宣称整批语义合格率。

## 文件

- 英文模板：/root/reflection_grpo_clean/configs/sft/teacher_simple/annotation.user.txt。
- requests.json 与 calls/*.request.json：实际请求，无密钥。
- calls/*.response.json：API 原始响应；查看最终标注不需要读取内部 thinking。
- annotations.jsonl：20 条问题、原学生解答、标注与结构检查，额外字段原样保留；不是筛选后的生产训练集。
- ANNOTATIONS.md：可直接阅读的 20 条最终输出和原学生回答，不展示 thinking。

thinking 参数说明：https://api-docs.deepseek.com/zh-cn/guides/thinking_mode/。
