# REFLECT training diagnosis and branch stability repair

Main experiment remains stopped. Branch experiment is restarted from SFT checkpoint268, with unchanged reward/advantage math and sample budget.

## Main findings

| Training steps | Tagged reflection rate | Reflection mean advantage | Reflection loss-token share | Mean generated tokens |
|---|---:|---:|---:|---:|
| 1-25 | 17.90% | -0.0053 | 19.57% | 636 |
| 26-50 | 15.38% | -0.0066 | 16.98% | 722 |
| 51-100 | 10.26% | -0.0435 | 11.05% | 971 |
| 101-150 | 5.12% | -0.0722 | 5.73% | 1326 |
| 151-200 | 2.56% | -0.0799 | 2.90% | 1653 |

1. Format gating was zero throughout this restarted run. First reflection already has zero efficiency cost; second reflection costs only 0.0625 on successful paths. Removing these costs alone is unlikely to restore the tagged behavior.
2. Whole-trajectory outcome advantage credits/blames the prefix, reflection action, and continuation together. The direct subgroup reward baseline is 1.25*p_direct - 0.25. One-reflection expected reward is p_reflect. Before normalization/flooring, their gap can be negative even without any reflection penalty.
3. The floor is a soft difficulty-weighted blend, not a hard global bound. floor0=-0.5 still leaves a negative update to reflection tokens. DG weight uses clipped direct mean reward; decay_steps is ignored while DG is enabled.
4. Grouped observations are not causal evidence that reflection harms the same prefix. Reflected trajectories are selected after an error and can be more difficult. True same-prefix controls are absent in the current main/branch algorithms.
5. Reflection continuation retains the problem and reflection summary, while direct inline self-correction retains the original reasoning context. This gives the two routes different information and is a potential utility disadvantage for weak summaries.
6. No KL/reference anchor, no SFT replay, and entropy coefficient zero make the SFT behavioral convention easy to lose. These are plausible contributors, not separately controlled causal findings.

## Untagged self-checking proxy

| Validation step | Tagged answers / all | Untagged answers with self-check keywords / untagged answers |
|---|---:|---:|
| 0 | 117/2160 | 89/2043 |
| 50 | 83/2160 | 141/2077 |
| 100 | 24/2160 | 342/2136 |
| 150 | 13/2160 | 630/2147 |
| 200 | 3/2160 | 891/2157 |

Keywords: wait, hold on, reconsider, recheck, mistake, let me check, let us check. This is a lexical proxy, not a human judgment of useful reflection. Six deterministic sampled excerpts show both correction and redundant/error-prone rechecking.

## Parameter proposals (not launched)

See main_parameter_proposal.json. First isolate a temporary reflection-negative-advantage warmup (floor0=0, DG off, decay100), retaining learning rate1e-6; evaluate every25steps for100steps. Separately compare learning rate3e-7. A reviewable main_floor_ablation.patch is prepared but NOT applied. The fourway launcher clears inherited TRAJ_GRPO_* variables and reinstalls REFLECT_ENV, so exporting shell variables alone does not apply these settings. A separate SFT-retention ablation can try KL0.01 to the frozen SFT reference, entropy0.001 or10% SFT replay. Do not combine all knobs and call the result strictly comparable with old baselines.

Do not reward tag presence alone. If utility remains low, use same-prefix K4 reflection versus K4 no-reflection continuations and event-level credit, rather than assuming root-weighted branches already provide that counterfactual.

## Branch crash and repair

Observed original failure: GPU1 Xid31 MMU page fault at05:22:37 on2026-09-14, during step12 initial generation after11completed optimizer updates. Asynchronous traceback points at vLLM0.8.5 bad-word logits bias; token IDs151667/151668 are valid for vocabulary151936. No exact out-of-memory or nonfinite-gradient error identified.

Concrete lifecycle defect found: the deployed FSDP-vLLM manager and vLLM0.8.5 allocator release/remap virtual memory without explicit device-wide synchronization boundaries. Added synchronization before/after sleep/wake and after parameter loading. This repairs a missing ordering invariant; the original asynchronous fault has not been uniquely proven to arise from this defect. CUDA graphs are also disabled as a conservative compatibility mitigation.

- Code: /root/reflect_branch_stable_20260915/code/reflection_grpo_clean
- Configuration: /root/reflect_branch_stable_20260915/config.json
- eager=True; actor/logprob token microbatch budget16384->8192; same64questions×8roots, K4children,3rounds×4096,lr1e-6,500updates,one optimizer update/step.
- User restored save frequency to50. Resumed from one transition checkpoint at step5, with save_freq=50, test_freq=50 and val_before_train=False; optimizer/dataloader state restored through native resume. Only evaluated checkpoints enter best selection. Keep best+latest.
- No shared Python environment modifications; frozen new code copy, previous artifacts preserved.
- Tests:44distinct CPU regression tests pass, one optional CUDA-specific test excluded. Full-batch GPU preflight and formal progress are recorded separately; launch alone is not a stability verdict.

Relevant upstream issue (supports failure class, not proof for this exact run): https://github.com/vllm-project/vllm/issues/48310
