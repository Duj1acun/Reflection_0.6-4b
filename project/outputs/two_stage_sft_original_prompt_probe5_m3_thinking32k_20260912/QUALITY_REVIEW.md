# MiniMax M3 thinking、32768 输出预算：同五题试验

本轮只改变模型/网关及用户指定的 thinking、输出预算，未改提示词或候选。请求参数：MiniMax-M3，https://api.minimaxi.com/v1，thinking.type=adaptive（官方接口的开启方式），reasoning_split=true，max_completion_tokens=32768，temperature=0.1，top_p=0.95，并发 5，不重试。

5 次第一阶段请求均 HTTP 200、stop、有 thinking、最终内容非空，无截断。总 token 47859（prompt 9174，completion 38685）。椭圆题使用 22212 completion tokens，其中 reasoning 21075，本次较大预算确实容纳了超过旧 16384 上限的生成。

## 必须区分格式失败和内容失败

5 条最终内容都被包在 Markdown 的 ```json 围栏中，因此当前脚本的严格 json.loads 全部拒绝，未发起第二阶段请求。0/5 是当前自动管道通过数，不是数学准确率。

用户原始脚本 extract_json 会移除围栏；本次构造脚本的严格解析比原始脚本更严格。此差异需要明确披露，不能将“未续推”全归咎于模型能力。以下人工检查仅移除最外层围栏读取内容，不改写或覆盖原始返回：

| 题目 | 去围栏后的问题 |
| --- | --- |
| 三次方程 | 诊断优化变量问题基本合理，但复制时把弯引号改为直引号；error_step 也压缩成非原文的一行，违反逐字符要求。 |
| 三角形 | 符号诊断正确、verified_state 在首错前，但 cut_point 从 Step 1 开始，遗漏回答开头，非完整原文前缀。 |
| 椭圆 | 原文复制把 semi-minor 改成 semi-major，并改变引号；error_step 非逐字引用。漏掉此前“准线通过顶点”的错误；verified_state 包含真实首错之后写出的椭圆方程。new_direction 从 A、B 对称直接推出整条抛物线的轴是 y 轴，缺少依据。 |
| 四质数 | 复制改变引号，且保留 error_step 后的答案段。聚焦最终最小值断言，未定位更早的非精确等号/冗余问题；error_reason 给出最终答案 22 和具体可行组合，反思已泄漏解答。 |
| 三质数 | 去围栏后严格原文及字段检查通过；正确识别重复测试 (3,5,11)，此前排除结果也来自有效前缀。但 new_direction 已给出 (2,5,7)、验证过程与最终答案 78，违背只给一句下一步指令、不提供完整解答的要求。 |

因此并非只去掉围栏就能得到五条合格 SFT 数据：其中四条仍有截取/引用问题，余下一条存在完整答案泄漏。不得把这次结果与其他模型的续推正确率直接比较，因为本次没有续推调用。

完整第一阶段返回见 GENERATED_RESULTS.md / probe_results.jsonl；接口请求见 calls/*.request.json。没有修改原始输出、追加付费调用或导出训练集。之前 M3 无 thinking 的单条诊断保存在另一个目录，不计入本轮五次调用。
