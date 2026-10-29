# DeepSeek V4 Pro：用户 reasoning error editor 提示词试跑

2026-09-12；3 个并发请求，无重试。原样使用用户提示词（去掉末尾作为粘贴分隔符的三引号），在末尾附加实际 Input JSON；包含 ground_truth。thinking enabled，reasoning_effort high，temperature 0.1，top_p 0.95，max_tokens 16384。未改动生产提示词，未导出训练数据。

## 接口与格式

- 3/3 HTTP 200，返回模型 DeepSeek-V4-Pro，thinking 非空，finish_reason=stop。
- 3/3 严格 JSON 解析成功，字段类型和结构正确。
- 3/3 cut_point 是学生原文的精确非空前缀；3/3 error_step 是原文精确子串。
- 超平面题 error_step 没有包含公式闭合标记，而 cut_point 包含该标记，因此机械 endswith=false；不算复制失败。
- 本批全为错误样本，未测试 perfect 分支，也未专门测试冗余检测能力。
- 总 usage：prompt_tokens=5523，completion_tokens=24713，total_tokens=30236；耗时约 110/124/191 秒。

## 内容审查

### 平方和（qwen3-0.6b_k2_03）

正确识别第二个方程右端把 -6 写成 +6 的符号错误。
但 verified_state 包含修正后的 a+2b-3c=-6，以及首错之后才写出的 2a-b+2c=2。两者数学上正确，但不是严格有效前缀中已经建立的子结论。

### 超平面（qwen3-4b_k1_02）

识别了错误的 Zaslavsky 计数公式，未误把最终的倍增错误作为首个错误。verified_state 中 10、3、13 个超平面均来自此前内容。
仍有粒度问题：更早的 “We are given three types of hyperplanes” 与后续实际两类不符，未被标记；若将此算作数学分类错误，所选位置仍不是严格首错。
new_direction 给了使用正确公式或逐次加超平面两种泛化方法，未给出该题更直接的坐标排序方向，可执行性一般。

### 蚂蚁（qwen3-4b_k2_01）

正确诊断 “- Odd-numbered moves: **North**”。不过紧前一句 “The ant alternates between **North** and **East-West** directions.” 已有同样的方向错误，严格逐句首错应更早。
verified_state 加入南北交替以及 N:1,5,9,... / S:3,7,11,... 等教师修正结论，不是错误前已有的学生子结论。

## 结论

本批原文复制可靠，主要数学诊断有用，但 2/3 样本出现严格前缀状态污染，不能原样满足此前的反思 SFT 目标。原因之一是当前提示词仅要求尽量枚举正确结论，没有明确禁止纠正、新推导和错误后结论。另有首错粒度、冗余界定问题。所有样本均为第三人称或客观语态；当前提示词未要求第一人称，也不生成完整续推，不能单独提供第二阶段 reasoning 标签。

建议后续仅在用户批准后调整：严格限定 verified_state 来自首错前显式建立的结论；禁止修正或新推导；若目标是数学反思，将冗余检测单独处理。本轮没有擅自改变提示词，也没有追加付费请求。

完整最终输出见 results.json 中 content；每题 .request.json 保存实际发送提示词，.response.json 保存接口原始响应。
