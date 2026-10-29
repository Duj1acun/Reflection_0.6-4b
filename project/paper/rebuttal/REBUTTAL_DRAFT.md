# Response to Reviewer sX1P

We sincerely thank the reviewer for the careful reading and constructive feedback. We are grateful that the reviewer recognized the motivation, the clean-slate structural fix, the ablation design, and the dedicated RL objective. We also fully understand the main concern: REFLECT+ should not be judged only by accuracy unless the paper also makes clear whether the gain comes from the clean-slate design or from a larger inference budget caused by extra tokens and multiple generation rounds. We agree that our current draft did not make this distinction sufficiently clear, and we will revise the paper accordingly.

## 1. Token Budget and Length Distribution

We agree that reporting only an overall average response length is not enough. In the revision, we will report per-benchmark response lengths together with the main accuracy/pass metrics. Importantly, the length is counted as the full generated response for each trajectory, including all clean-slate continuation rounds when reflection is triggered, rather than only the final round.

Using the completed evaluation table, REFLECT+ improves the overall mean accuracy/pass scores while reducing the overall average generated length:

| Dataset | Metric | GRPO score | REFLECT+ score | GRPO len. | REFLECT+ len. |
|---|---:|---:|---:|---:|---:|
| GSM8K | Avg@32 | 71.66 +- 0.29 | 80.07 +- 0.22 | 466.7 | 345.1 |
| AMC23 | Avg@32 | 43.36 +- 1.07 | 39.52 +- 1.02 | 990.0 | 868.9 |
| MATH500 | Avg@32 | 62.07 +- 0.30 | 66.43 +- 0.20 | 721.9 | 628.8 |
| LogiQA | Avg@32 | 22.35 +- 0.88 | 29.31 +- 0.33 | 665.7 | 505.3 |
| GPQA | Avg@32 | 10.73 +- 0.72 | 18.77 +- 1.02 | 995.5 | 960.6 |
| MMLU-Pro | Pass@1 | 13.76 | 22.54 | 611.2 | 485.0 |
| BBH | Avg@8 | 22.39 +- 0.75 | 28.76 +- 0.68 | 699.0 | 607.8 |
| DROP | Avg@8 | 27.27 +- 1.73 | 35.70 +- 0.88 | 514.3 | 373.9 |
| CLUTRR | Avg@8 | 2.58 +- 0.49 | 3.99 +- 0.21 | 569.7 | 734.2 |
| Overall | Mean Avg / Mean Pass | 32.80 / 68.57 | 37.82 / 72.83 | 692.7 | 612.2 |

This table also directly addresses the reviewer's concern about the high-reflection-trigger benchmarks. On LogiQA, GPQA, and MMLU-Pro, REFLECT+ improves accuracy while using fewer generated tokens than GRPO. We will also be careful not to overstate the length claim: CLUTRR is an exception where REFLECT+ is more accurate but longer. We will explicitly discuss this exception and report benchmark-level lengths rather than relying on a single aggregate number.

## 2. Whether the Gains Come from More Tokens or Multiple Rounds

We agree that REFLECT+ can require multiple model calls for samples that trigger reflection. However, we would like to draw the reviewer's attention to the core result in Table 5. Controlling for reflection content r, Clean-Slate reconstruction (x, r) significantly outperforms appended reconstruction (x, y^-, r). The latter yields even longer responses but suffers from performance degradation due to retaining the erroneous prefix. This directly proves that the gain comes from discarding the faulty prefix, not from stacking more tokens. Table 2 further rules out the "longer is better" explanation, as forced/random interruption performs far worse than the adaptive trigger.

Regarding multi-round costs, we acknowledge the increased system complexity. However, we note that: (1) reflections are capped at two events, providing a clear upper bound; (2) vLLM rolling engine supports efficient continuous batching, limiting latency increase; (3) on 8 of 9 benchmarks, the average response length is reduced by 3.5%-27.3% compared with GRPO, with CLUTRR as the exception (+28.9%). We will discuss this trade-off more explicitly in the "Limitation" and "Efficiency Discussion" sections.

## 3. Accuracy-Complexity Tradeoff

We agree that the accuracy-complexity trade-off should be discussed. REFLECT+ increases system complexity for samples that trigger reflection, but it addresses a structural flaw in long-CoT reasoning-anchor failure-where a single wrong assumption in the visible prefix permanently corrupts subsequent generations, a problem that simply extending the context or appending reflections cannot resolve. Clean-slate reflection trades bounded prompt rebuilds for removing this corrupted state. This design is forward-looking: as reasoning scales to o1-class models with chains spanning tens of thousands of tokens, discarding the erroneous prefix yields substantial memory and attention savings that outweigh the overhead of multi-round calls. The benchmark-level length reductions above provide practical efficiency evidence for this trade-off, while CLUTRR reminds us that the benefit is not uniform.

## 4. Model Scale and Task Scope

We agree that the current evidence is limited to one backbone scale. Whether clean-slate reflection remains beneficial for larger models is an empirical question: stronger models may self-correct better in context, but this does not necessarily eliminate anchor failure caused by erroneous prefixes, since REFLECT+ changes the conditioning context rather than merely increasing reasoning capability. Our current study therefore focuses on whether a small RL-trained reasoner, which may be more vulnerable to erroneous prefixes, can learn strategic self-correction through clean-slate state replacement.

At the same time, the evaluation is not restricted to mathematics. In addition to MATH500, AMC23, and GSM8K, the updated table includes LogiQA, GPQA, MMLU-Pro, BBH, DROP, and CLUTRR. REFLECT+ improves over GRPO on LogiQA, GPQA, MMLU-Pro, BBH, DROP, and CLUTRR, suggesting that the clean-slate mechanism is not only fitting the in-domain math benchmarks. We will still describe these as broader but not definitive evidence, because several of these tasks are multiple-choice or structured reasoning tasks and the backbone scale remains fixed.

To address the model-scale concern, we are now running REFLECT+ on Qwen3-4B-Base and Qwen2.5-Math-7B and will report the detailed comparison as early as possible. We will also revise the limitation section to explicitly state: one completed backbone scale, math-centric RL training, limited open-ended non-math evaluation, and the need for larger-model studies.

## 5. Significance and Variance

We appreciate this point. The original table could give the impression of a single-run result. In the revision, we will report mean and standard deviation over five evaluation seeds wherever available; MMLU-Pro currently has one seed under the test results and will be marked accordingly. As shown in the updated table, REFLECT+ improves the overall Mean Avg from 32.80 to 37.82 and Mean Pass from 68.57 to 72.83.

We also agree that AMC23 should be discussed carefully. REFLECT+ is lower than GRPO on AMC23 Avg@32 (39.52 +- 1.02 vs. 43.36 +- 1.07), although the response length is shorter. We will not hide this exception; instead, we will note that the gains are broad but not uniform. Finally, we will clarify that these are multiple evaluation seeds rather than independent RL retraining seeds, and we will list independent retraining runs as a limitation/future robustness check.

## 6. Reward Design and Hyperparameter Sensitivity

We agree that the reward design is more involved than vanilla GRPO, and we should better justify why each component is needed. In the revision, we will tie each component to a concrete failure mode:

- The format gate prevents malformed reflection blocks from receiving task reward.
- The wrong-no-reflect penalty mitigates under-triggering when direct solving fails.
- The reflection-count/efficiency term discourages repeated or unnecessary reflections.
- The counterfactual direct baseline evaluates reflection against direct attempts on the same problem.
- The difficulty-gated advantage floor prevents the reflection policy from being prematurely suppressed during early exploration.

We will also add the training-dynamics evidence from the logs. The policy does not collapse into always reflecting or never reflecting on the in-domain validation set: validation reflection rate remains selective, roughly 9%-18% over the monitored period, with 11.6% at step 200. The format-valid rate during training is high, mostly about 96.5%-99.4%, indicating that the format gate does not lead to frequent invalid outputs. Validation accuracy improves from 62.4% at step 0 to 73.2% at step 200.

For hyperparameters, we will clarify the design rationale and cite the prior work that inspired these settings. The values were chosen to encode qualitative constraints in our reflection setting: the first reflection should be nearly cost-free to encourage exploration, repeated reflections should be discouraged, and malformed reflections should not receive task credit. In our current setting, for example, `d_free=1` and `phi=4` implement this preference by allowing the first reflection while mildly discounting repeated reflection; the advantage floor is set as a moderate exploration safeguard rather than a permanent positive reward; and the wrong-no-reflect penalty is kept smaller than the reward for a correct answer. We agree that this is not a full sensitivity study, and we will present it as a limitation rather than claiming exhaustive hyperparameter robustness.

We sincerely appreciate the constructive feedback and believe the additional experiments and clarifications will strengthen the paper. If our responses have addressed the concerns, we would be grateful for a promotion of the overall score.
