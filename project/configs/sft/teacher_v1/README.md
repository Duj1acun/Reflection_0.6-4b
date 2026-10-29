# SFT teacher prompt v1 草案

状态：已完成 4 条小样本实调用，共 6 次请求，均返回 thinking。当前没有完整记录通过全部收录要求；批量标注和训练未启动，旧 teacher 入口未替换。详细结论见本次试验目录中的 QUALITY_REVIEW.md。

2026-09-10 补充：按用户要求完成同样 4 条样本的 MiniMax-M3 对照，共 5 次请求，均返回 thinking。数学诊断有改善，1 条合理筛为 incomplete；但其余 3 条反思均泄露最终答案，仍不能原样训练。M3 试验及人工报告位于 `/mnt/data/reflection_grpo/data/sft/teacher_v1_probe4_m3_t03_20260910/QUALITY_REVIEW.md`。本轮未修改下面的 prompt/schema，也未将 M3 设为全量默认模型。

## 范围

本批反思候选为 Qwen3-0.6B 的 3,441 条和 Qwen3-4B 的 4,412 条。仅来自各学生模型 4 次采样中答对 1、2、3 次的题，每题最多两条完整错误回答。候选规模不是最终合格数；同题不同文本也不保证错误类型不同。

Perfect 不调用 teacher API，直接复用学生已采样且本地判分正确的回答，进行去重、长度和数据一致性检查。只有错误轨迹进入本次 teacher 试调用。

两阶段 SFT 对应：

- Phase 1：题目 + 完整错误回答 → 反思。
- Phase 2：题目 + 反思 → 正确续推。

Teacher 调用顺序是先诊断、再在新上下文里续推，不是先独立解题再诊断。两次调用均允许 thinking。

## Prompt 文件与消息组装

- 诊断 system：diagnosis.system.txt，末尾附上 diagnosis.schema.json 的完整 JSON 文本。
- 续推 system：continuation.system.txt，末尾附上 continuation.schema.json 的完整 JSON 文本。
- 两个 user 消息都使用标准 JSON 序列化，不用手工拼接未转义的原始题目。
- 两次请求是独立的新对话，不复用 messages 或 teacher thinking。
- schema 用于本地严格校验；未确认网关兼容前，不假定它支持 API 侧 json_schema 响应约束。

诊断 user 的字段仅为：

    {
      "problem": "题目原文",
      "reference_answer": "可能有误的参考答案",
      "student_steps": [
        {"step_id": "S0001", "text": "原回答中的第一段原文"},
        {"step_id": "S0002", "text": "原回答中的第二段原文"}
      ]
    }

程序按原文段落分段，保留字符和原文偏移并分配稳定编号。编号只用于定位，不代表严格的数学推理步。不得改写公式、翻译、压缩或 Unicode 归一化后再校验原文引用。

续推 user 的字段严格只有：

    {
      "problem": "题目原文",
      "reflection_text": "<reflection>...</reflection>"
    }

续推请求中不得出现原错误回答、参考答案、参考解法、首次调用的原始响应或 thinking。诊断模型看到的参考答案不能通过额外元数据传给续推模型。

## 诊断输出字段

只有 verdict=incorrect 才进入续推队列：

- first_error_step、first_error_quote、error_type 是审计字段，不作为学生的输出目标。
- verified_states、error_reason、new_direction 是学生要学习的反思内容。
- review_note 仅用于非 incorrect 记录的复核，不作为训练内容。

schema 之外还必须验证错误编号存在、引用确实位于对应原文段落，以及反思没有虚构中间结论。schema 检查不等于数学检查。

对于原本正确、仅格式错误、未完成、参考答案冲突和无法可靠诊断的样本，单独保存复核结果，不强制制造错误，也不自动更改原始数据标签。

## 反思的统一渲染

程序只取三个反思字段，按固定顺序渲染为：

    <reflection>
    So far I have: {verified_states}
    However, {error_reason}
    Next, {new_direction}
    </reflection>

让 teacher 字段自身包含完整标点，渲染器不重复追加句号。三个字段不得含有额外 reflection/think 标签、参考答案提示或 step ID 代称。error_reason 必须直接说明相关表达式和错误操作，因为续推模型看不到原回答。

不让 teacher 返回或复制 cut_point。Phase 1 的兼容字段 cut_point 由程序直接保存完整错误回答，不能截在错误发生之前。

## 示例（人为编写，不是当前数据原文）

诊断输入：

    {
      "problem": "Find the positive real number x satisfying (2x - 3)^2 = 25.",
      "reference_answer": "4",
      "student_steps": [
        {"step_id": "S0001", "text": "Taking square roots gives 2x - 3 = 5 or 2x - 3 = -5."},
        {"step_id": "S0002", "text": "Therefore 2x = 5 - 3 or 2x = -5 - 3."},
        {"step_id": "S0003", "text": "Thus x = 1 or x = -4, so the positive solution is 1."}
      ]
    }

期望诊断形态：

    {
      "verdict": "incorrect",
      "first_error_step": "S0002",
      "first_error_quote": "2x = 5 - 3 or 2x = -5 - 3",
      "error_type": "algebraic_transformation",
      "verified_states": "The constraint is x > 0. Taking square roots gives 2x - 3 = 5 or 2x - 3 = -5.",
      "error_reason": "I subtracted 3 when isolating 2x, but removing -3 from the left requires adding 3 to both sides.",
      "new_direction": "I should add 3 to both sides in each branch, solve for x, and then apply the positivity constraint.",
      "review_note": ""
    }

渲染后仅将题目与反思送入续推调用。期望续推形态：

    {
      "status": "ok",
      "reasoning": "Adding 3 to both sides gives 2x = 8 or 2x = -2. Dividing by 2 gives x = 4 or x = -1. The condition x > 0 leaves x = 4. Substitution gives (2*4 - 3)^2 = 25, so the answer is \\boxed{4}.",
      "review_note": ""
    }

## 启动前的参数建议

- Teacher 暂沿用旧入口默认的 MiniMax-M2.7，不自动改成其他模型；启动时明确记录实际 model ID 和网关地址。
- 诊断和续推均使用 temperature=0.3，top_p=0.95，每次请求只生成一个结果。按用户要求降低采样温度，thinking 保持不变；实际标注质量通过小批量试制验证，不预设低温一定更准确。
- max_completion_tokens=16384。thinking 与最终内容分开处理；不要把最终反思需要简短理解成要禁止充分思考。
- 总在途请求上限 64，由两个阶段共享；不是诊断 64 加续推 64。出现限流时降低实际并发并退避。
- 单请求超时先按 600 秒准备；HTTP 暂态错误最多 3 次尝试，尊重 Retry-After，记录每次消耗。不要重复叠加 SDK 与外层重试。
- 长度耗尽时不接受 thinking 草稿或半截 JSON。先记录并隔离；增加预算重跑应有明确上限，不无限重试。
- 学生聊天模板仍使用 enable_thinking=false。SFT 总序列长度建议 4096，按各自 tokenizer 检查两个阶段的完整拼接结果，不静默截断监督目标。

按 2026-09-10 核对的 [MiniMax 官方文档](https://platform.minimax.io/docs/api-reference/text-openai-api)，M2.x 的 thinking 无法关闭；reasoning_split 只控制返回格式。M3 可显式设置 thinking.type=adaptive。第三方网关是否按该规则转发，仍需小样本验证。

对 M2.7，额外请求体沿用 reasoning_split=true。如果后续明确改用 M3，则额外请求体可以为：

    {
      "thinking": {"type": "adaptive"},
      "reasoning_split": true
    }

## 实现与验收边界

这些资源不能直接用旧 teacher 入口运行。接入前仍需完成：

1. 选取具体轨迹，保留 student_model + trajectory_id；冻结输入和 prompt 哈希，断点续跑不能只按 problem_id。
2. 按 problem_id 在所有模型和所有样本类型之间统一划分 train/eval；标注可以覆盖两侧，但验证集绝不能参与训练。
3. 新增诊断和续推两个独立阶段，按上面的白名单组装输入。续推重试也不能追加参考答案或首次 teacher 的 thinking。
4. 只解析正常结束的最终 content 中的完整 JSON，拒绝重复 JSON 键、多个顶层对象、非有限数值和额外字段；reasoning_content/reasoning_details 仅作原始审计记录。
5. 本地校验 schema、原文引用、答案数学等价性和完整训练长度。答案一致只是必要条件，不足以证明推导正确或反思有用。
6. 对反思状态正确性、答案泄漏、续推是否衔接进行语义审核；发生争议的样本进入复核队列，不通过凑 gold 修复。
7. 仅 incorrect + ok + 所有验收通过的记录编译为训练数据；保留两阶段请求、usage、finish_reason、失败原因和版本信息。

## 当前获准的小样本试调用

先从每个模型选 2 条错误轨迹，总计 4 条；最多 8 次 HTTP 请求，不自动重试，不因拒收自动追加样本。0.6B 覆盖答对 1/4、2/4 的题，4B 覆盖答对 2/4、3/4 的题。选择按固定种子的哈希排序，不据 teacher 结果挑选成功案例。这只验证接口和标注行为，不能估计全库通过率。

入口为 scripts/data/probe_teacher_v1.py，分为 prepare 和 run 两个动作。prepare 不需要密钥，不调用网络 API；run 要求环境变量 MINIMAX_API_KEY，拒绝重复运行同一个试验目录。请求体和响应会留档，但 Authorization 和 API key 不写入文件。

本次已准备的目录：

    /mnt/data/reflection_grpo/data/sft/teacher_v1_probe4_t03_20260910

其中 samples.json 保存原始样本，manifest.json 冻结输入哈希、prompt 和准备参数。本次根据用户明确提供的地址，用 --base-url https://api.minimaxi.com/v1 覆盖准备配置中的旧网关，实际调用配置记录在 run_config.json。密钥经隐藏终端输入，不写入文件。

实际完成 4 次诊断、2 次续推。6 次请求均正常返回并提供非空 thinking；两份续推的数学终值分别为 325、-2，但都没有 boxed 格式。人工复核还发现反思方向不可靠、错误定位不准确、答案泄漏和错误恒等式等问题。完整记录 0/4 收录，不应直接启动全量。

数学等价性、完整训练长度和语义质量已进行小样本复核，结果见试验目录中的 QUALITY_REVIEW.md。原始响应未修改，不把人工修正当成 API 输出。

四条试调用审核后，可以另行扩大到每个模型 200 条分层试制，观察诊断准确性、续推质量和实际通过率，再决定全批次运行。不得把当前小样本授权视为启动完整批次的授权。

后续效果评估需同时比较 teacher 反思和学生自生成反思，并做相同生成预算下的无反思重答对照；不能把“强 teacher 续推答对”直接当成“反思有效”的证明。
