# REFLECT+ Rebuttal TODO

## Priority 0: Generate Offline Evidence

- Run `python rebuttal/analyze_existing_results.py` from the repo root.
- Check `rebuttal/generated/rebuttal_analysis.md` for:
  - total decoded-token comparison on all six main benchmarks;
  - REFLECT+ reflection-depth breakdown;
  - paired bootstrap confidence intervals against GRPO and GRPO500;
  - existing checkpoint ablations on the common five-dataset subset;
  - training dynamics parsed from local DG run logs.

## Priority 1: Main Rebuttal Claims

- Training/inference mismatch:
  - State that both training rollout and evaluation use the same clean-slate rebuild rule: after a complete reflection block, continuation is generated from `(x, r)` only.
  - Distinguish this from teacher-forced cold-start SFT; the RL stage samples the actual inference trajectory.
  - Use training-dynamics table to show nonzero trigger rate, high format validity, and no observed collapse in the monitored DG run.

- Token/compute fairness:
  - Emphasize that total decoded tokens include all clean-slate rounds.
  - Note that single-turn GRPO/base receive 3072 max tokens, matching REFLECT+'s 3 rounds x 1024 per-round budget.
  - Use the per-benchmark token table to answer the high-trigger LogiQA/GPQA/MMLU-Pro concern directly.

- Statistical reliability:
  - Use paired bootstrap over questions for mean@N and pass@N deltas.
  - Acknowledge AMC23/AIME small sample variance and that AMC23 mean accuracy is lower, while AMC23 pass@32 is higher.

- Reward complexity and hyperparameters:
  - Explain each reward component by the failure mode it prevents: malformed reflection, no-reflection failure, over-reflection, reflection collapse on easy prompts.
  - Use the existing checkpoint ablation table as support, not as a full sensitivity sweep.
  - Avoid claiming exhaustive hyperparameter robustness.

- Prior reflection methods:
  - Map Table 2 to fixed/random forced reflection schedules.
  - Map Table 5 to context-preserving reflect-then-retry `(x, y-, r)` baselines.
  - Acknowledge that a full retraining comparison with every prior RL reflection method is outside rebuttal time.

## Priority 2: Paper Edits

- Fix typo: "showinging benefit" -> "showing the benefit".
- Rewrite Table 3 explanation:
  - Do not compare w/o-RL and w-RL rows directly.
  - Explain that RL changes the reflection-triggered population, making the w-RL subset harder.
  - Within w-RL, REFLECT+ internal reflections are competitive with the 3B teacher and better on MATH500/AMC23.
- Add an appendix subsection "Compute and Length Analysis" using the generated token table.
- Add an appendix subsection "Training Dynamics and Stability" using parsed log metrics.
- Add bootstrap CIs in the main results appendix or a short paragraph below Table 1.
- Expand limitations: one 1.5B model, math-centric training, OOD benchmarks are MCQ, no multi-seed retraining, multi-round serving complexity.

## Optional If GPU Time Exists

- Add a pure inference appended-retry baseline on a small subset:
  - same reflection points;
  - continue from `(x, y-, r)` instead of `(x, r)`;
  - report only as an extra qualitative/diagnostic comparison if completed.
- Do not start new RL training.
