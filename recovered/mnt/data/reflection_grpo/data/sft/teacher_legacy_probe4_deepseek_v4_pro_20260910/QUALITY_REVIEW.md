# DeepSeek-V4-Pro 四题试验

重要来源说明：本次使用 /root/reflection_grpo_clean/scripts/data/prepare_sft_data.py 的 _teacher_prompt()，即此前展示并测试的现存 clean 版本；并未确认这是 /root/reflection_grpo 当初造数时的原始 prompt。不能将本试验描述为原仓库历史 prompt 的复现。

并行科技模型列表鉴权成功，本 key 返回的列表没有 Qwen3.7-Plus，有 DeepSeek-V4-Pro。本次只测试 DeepSeek-V4-Pro，未替换为其他 Qwen 型号。

使用同样四题；system/user 消息与 M3 四题试验逐字一致。实际四次请求，每条一次，无重试，最高并发四；4/4 HTTP 200、返回模型 DeepSeek-V4-Pro、stop、非空 thinking。4/4 原始 JSON 直接解析成功，4/4 有 boxed，人工数学复核及本地判分均正确。没有导出训练集或追加调用。

请求 thinking.type=enabled、reasoning_effort=high、top_p=0.95、max_tokens=16384。发送 temperature=0.3，但 [DeepSeek 文档](https://api-docs.deepseek.com/zh-cn/guides/thinking_mode/) 说明 thinking 模式不支持 temperature、发送也不生效；平台实际路由行为不可仅凭请求体完全确认，因此不是严格同温度实验。实际 usage：输入 4458、输出 11024、总计 15482 tokens；不能据跨厂商 token 总数直接推算费用。

结论：同组样本中固定点题的首错与状态组织比 M3 更准确，四条格式也更整齐；但样本仅四条、每条一次，不能外推全量优劣。三条真正错误的修复与一条未完成化简的补全均数学正确。第三人称自述适配和最终答案提前出现在反思里仍需处理，未擅自修改 prompt。

## qwen3-0.6b_k1

主要诊断与组合计数证明正确，得到 325；方向没有给出最终答案。按合并至只剩一个数的自然理解成立。verified_states 中每次少一个数是直接操作推论，不按新增基础结论拒收。

题目：There are 26 ones written on a board. Each minute, Karlson erases any two numbers and writes their sum on the board, then eats the number of candies equal to the product of the two erased numbers. What is the maximum number of candies he could have eaten in 26 minutes?

参考答案：325

API 最终 JSON（仅标准序列化展示，字段未改写，不展示内部 thinking）：

```json
{
  "verdict": "incorrect",
  "verified_states": "The initial board has 26 ones; each operation chooses two numbers a,b, replaces them by a+b, and earns ab candies; the number of entries on the board decreases by one each time.",
  "error_reason": "The solution incorrectly assumes that because the board starts with 1s, every erased pair has product 1; after the first merge the board contains larger numbers such as 2, so later products can be larger.",
  "new_direction": "Model the process as a binary merge tree and count how many pairs of the original 26 ones are counted by each merge.",
  "reasoning": "Label the 26 original ones. When two numbers a and b are merged, suppose they are the sums of p and q original ones. The earned product is pq, which counts exactly the pairs consisting of one original one from the first group and one from the second. Every unordered pair of original ones is separated until the merge that combines their two groups, and at that merge it is counted exactly once. Therefore the total number of candies over the entire merging process is exactly the number of unordered pairs of original ones, independent of the order of merges. Thus the maximum is C(26,2)=26*25/2=325, so the answer is \\boxed{325}"
}
```

## qwen3-0.6b_k2

准确定位学生更早丢失 -8/9 的错误；verified_states 只保留原有坐标关系，修正坐标放在 new_direction，计算 b=-1 后得到 1。这比同组 M3 返回更符合首错和字段语义。

题目：Given the function $$y=\log_{a}(x+3)-\frac{8}{9}$$ (a>0, a≠1), its graph always passes through point A. If point A also lies on the graph of the function f(x)=3^x+b, then f(log\_3⁡2) equals to \_\_\_\_\_\_.

参考答案：1

API 最终 JSON（仅标准序列化展示，字段未改写，不展示内部 thinking）：

```json
{
  "verdict": "incorrect",
  "verified_states": "Point A can be written as (α,β), with β = log_a(α+3) - 8/9.",
  "error_reason": "The student drops the constant -8/9 and treats A as lying on y=log_a(x+3) instead of y=log_a(x+3)-8/9, while also failing to use that A is fixed for all a.",
  "new_direction": "Choose the x-coordinate that makes the logarithmic term independent of a, namely x+3=1, so A=(-2,-8/9), then use it to find b.",
  "reasoning": "Since the graph of y=log_a(x+3)-8/9 always passes through A for every a>0, a≠1, the term log_a(x+3) must be independent of a. This happens only when x+3=1, because log_a(1)=0 for all a. Thus x=-2 and y=0-8/9=-8/9, so A=(-2,-8/9). Since A also lies on f(x)=3^x+b, we have f(-2)=-8/9. Therefore 3^{-2}+b=1/9+b=-8/9, so b=-1. Then f(log_3 2)=3^{log_3 2}+b=2-1=1. Thus the answer is \\boxed{1}."
}
```

## qwen3-4b_k2

准确区分 x 与 x+2 的展开基底，代入 x=-1 得到 -2；诊断与续推均合理。

题目：Given the equation $(x^2+1)(2x+1)^9=a_0+a_1(x+2)+a_2(x+2)^2+\ldots+a_{11}(x+2)^{11}$, find the sum of coefficients $a_0+a_1+a_2+\ldots+a_{11}$. Express your answer as a single integer.

参考答案：-2

API 最终 JSON（仅标准序列化展示，字段未改写，不展示内部 thinking）：

```json
{
  "verdict": "incorrect",
  "verified_states": "The equation is a polynomial identity, and the right side is expressed in powers of (x+2).",
  "error_reason": "The student treats the sum of the coefficients a_k as P(1), but because the polynomial is expanded in powers of (x+2), the coefficient sum is obtained by setting x+2=1, i.e. x=-1, not x=1.",
  "new_direction": "Substitute x=-1 into the identity so that every (x+2)^k equals 1 and the right side becomes the desired coefficient sum.",
  "reasoning": "Let y=x+2. Then the right side is Q(y)=sum_{k=0}^{11} a_k y^k, so the desired sum is Q(1). Since y=x+2, Q(1) corresponds to x+2=1, i.e. x=-1. Evaluating the left side at x=-1 gives ((-1)^2+1)(2(-1)+1)^9 = (2)(-1)^9 = -2. Therefore a0+a1+...+a11 = -2. \\boxed{-2}"
}
```

## qwen3-4b_k3

正确指出学生停在等价但未完全化简的表达式，没有捏造错误代数变形。续推得到 1，但 error_reason 已提前说出最终值 1；这不违反该 prompt 明文限制，却可能不利于反思与续推的训练分工。

题目：Simplify the expression: \[ \frac{\left(p q^{-1}+1\right)^{2}}{p q^{-1}-p^{-1} q} \cdot \frac{p^{3} q^{-3}-1}{p^{2} q^{-2}+p q^{-1}+1} \div \frac{p^{3} q^{-3}+1}{p q^{-1}+p^{-1} q-1} \]

参考答案：1

API 最终 JSON（仅标准序列化展示，字段未改写，不展示内部 thinking）：

```json
{
  "verdict": "incorrect",
  "verified_states": "Up to the expression \\(\\frac{(a^2-1)(a+b-1)}{(a-b)(a^2-a+1)}\\), with \\(a=p/q\\) and \\(b=q/p\\), the substitutions, factoring, and cancellations are correct.",
  "error_reason": "The solution stops at an unsimplified equivalent expression and incorrectly treats it as the final simplified answer, instead of using \\(b=1/a\\) to reduce it further to 1.",
  "new_direction": "Substitute \\(b=1/a\\) into \\(\\frac{(a^2-1)(a+b-1)}{(a-b)(a^2-a+1)}\\) and cancel the common factor \\(a^2-a+1\\).",
  "reasoning": "Let \\(a=p/q\\), so \\(p^{-1}q=q/p=1/a\\). Since \\(p^3q^{-3}=a^3\\), the expression becomes \\(\\frac{(a+1)^2}{a-1/a}\\cdot\\frac{a^3-1}{a^2+a+1}\\div\\frac{a^3+1}{a+1/a-1}\\). Factor \\(a^3-1=(a-1)(a^2+a+1)\\) and \\(a^3+1=(a+1)(a^2-a+1)\\). Canceling \\(a^2+a+1\\) and rewriting the division gives \\(\\frac{(a+1)^2}{a-1/a}\\cdot(a-1)\\cdot\\frac{a+1/a-1}{(a+1)(a^2-a+1)}\\). Cancel one \\(a+1\\): \\(\\frac{(a+1)(a-1)(a+1/a-1)}{(a-1/a)(a^2-a+1)}\\). Now \\((a+1)(a-1)=a^2-1\\), \\(a+1/a-1=\\frac{a^2-a+1}{a}\\), and \\(a-1/a=\\frac{a^2-1}{a}\\). Therefore the expression equals \\(\\frac{(a^2-1)\\cdot\\frac{a^2-a+1}{a}}{\\frac{a^2-1}{a}\\cdot(a^2-a+1)}=1\\). Thus the simplified expression is \\(\\boxed{1}\\)."
}
```
