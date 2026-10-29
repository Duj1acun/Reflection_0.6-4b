# 单次调用生成反思与续推：teacher v2

20 条扩展试验已完成：两个学生模型各 10 条全新样本，每条一次 M3 请求，prompt/schema 与原四条试验完全一致。20 条结构校验通过，18 条与原参考答案一致；另 2 条经复核为参考答案问题。保守建议保留 7 条候选、13 条待复核，未启动全量。完整标注和报告位于 /mnt/data/reflection_grpo/data/sft/teacher_v2_single_probe20_m3_t03_20260910/。

本版本按用户要求，将诊断和续推合并为一次 API 调用。实际 system prompt 在 annotation.system.txt，使用英文；下面为中文说明。annotation.schema.json 只做本地校验，不再作为大段 schema 附加到模型提示词中。

## 当前状态

2026-09-10 使用 MiniMax-M3 对原来的同一组 4 条学生回答完成试调用，每条恰好调用一次，合计 4 次。三个错误样本均产生正确的最终数值，反思没有直接给出最终答案；另一个样本被合理识别为 incomplete，但没有遵守审核记录的空字段要求。

严格人工检查下，系数和题可作为合格示例；另外两条错误样本仍有“teacher 新结论混入学生已有状态”等问题。尚未启动全量标注、导出训练数据或修改旧入口。详细报告：`/mnt/data/reflection_grpo/data/sft/teacher_v2_single_probe4_m3_t03_20260910/QUALITY_REVIEW.md`。

## 核心 prompt 的中文说明

你是帮助学生修复错误解答的数学教师。允许充分 thinking；最终只输出一个 JSON 对象，使用英文训练文本，不输出内部思考记录。

输入是题目、学生完整回答以及可能有误的参考答案。先核实学生是否真的存在未纠正的数学错误，不强行对齐参考答案，也不把“等价但未化简完成”说成错误变形。

如果确有错误，依次完成：

1. verified_states：保留有用的题目条件、变量定义和错误前已得到的正确子结论；没有有效进展时只写已知条件。不要把 teacher 新发现的修正或不变量冒充学生已有结论。
2. error_reason：用自己的话解释首个实质性且未纠正的错误，指出误用的表达式、假设或规则。用第一人称，不要求编号或逐字引用原句。
3. new_direction：用 “I should” 开头给出下一步修复操作或方法。说清要做什么，不在这里执行完整计算或给出最终答案。
4. reasoning：从保留的状态出发执行上述方向，给出必要计算、条件检查和最终结论。只依赖题目和反思中可理解的信息，不需要读者再看学生的错误回答。不要重复反思或谈论标注工作。
5. final_answer：单独给出最终答案，与续推结论一致，不带美元符号或 boxed 命令。程序负责统一答案格式。

同时输出 verdict 与 review_note。只有 verdict=incorrect 才使用上述五个训练字段；其余 verdict 只保留审核说明，不强行构造反思。完整英文 prompt 还包含一个正负平方根移项错误的小例子，明确示范“反思只提操作、续推才实际计算”的边界。

## 请求与输出

user 消息为标准 JSON 序列化的三个字段：

    {"problem": "...", "student_response": "...", "reference_answer": "..."}

最终输出字段：

    {
      "verdict": "incorrect",
      "verified_states": "...",
      "error_reason": "I ...",
      "new_direction": "I should ...",
      "reasoning": "...",
      "final_answer": "...",
      "review_note": ""
    }

去掉了 first_error_step、first_error_quote 和 error_type；不再要求 teacher 逐字复制任何原始段落。

参数：MiniMax-M3，temperature=0.3，top_p=0.95，max_completion_tokens=16384，thinking.type=adaptive，reasoning_split=true。试验并发上限 4，无重试，最多 4 个请求。原计划的全量并发 64 尚未启动。

## 如何服务两阶段 SFT

程序把 verified_states、error_reason、new_direction 渲染为既有的三段 reflection_text，然后把 reasoning 与单独的 final_answer 拼成带 boxed 答案的续推目标。

- Phase 1：题目 + 学生完整失败回答 → reflection_text。
- Phase 2：题目 + reflection_text → reasoning + 程序生成的 boxed(final_answer)。

因此两个训练阶段保留，但构造一条候选只需要一次 teacher 请求。Perfect 仍不经过 API。

单次调用并不具备“两次独立上下文”的信息隔离：teacher 在生成续推时看过失败回答与参考答案。必须审核续推是否自洽、是否充分交代必要推导；不能把同一次调用的自检当成独立质量验证。

## 格式与质量边界

- 只解析正常结束的最终 content；独立 thinking 字段仅留档，不用作训练目标。
- 允许去掉一个完整 JSON 代码围栏，并记录 normalization；不修补内部 JSON，不改公式。
- final_answer 由程序补 boxed，答案分数仅说明终值与参考一致；人工仍需检查 reasoning 与 final_answer 是否一致、过程是否可靠。
- 状态来源、错误解释、反思提前执行修复、最终答案泄漏都需要语义复核，不能靠字段齐全来判定合格。
- 非 incorrect 记录绝不因包含额外的正确状态就转为反思数据。当前严格 schema 仍要求这些训练字段为空；本次 incomplete 的多余状态保留为审核证据，不擅自清空后冒充模型合规输出。
- 试验脚本为 scripts/data/probe_teacher_v2_single.py，默认仍为四条；准备和运行均显式传入 --num-samples 20 才允许二十条试验。拒绝重复付费运行，未替换生产数据构造入口。
