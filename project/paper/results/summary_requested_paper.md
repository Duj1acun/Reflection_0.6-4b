## Table 1. Main Results (5 seeds).

| Model | GSM8K avg@32 / pass@32 | AMC23 avg@32 / pass@32 | Math500 avg@32 / pass@32 | LogicQA avg@32 / pass@32 | GPQA avg@32 / pass@32 | MMLU-Pro pass@1 | BBH avg@8 / pass@8 | DROP avg@8 / pass@8 | CLUTRR avg@8 / pass@8 | Trigger rate | Resp. len. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 16.10 / 91.96 | 14.84 / 80.00 | 18.06 / 87.40 | 9.38 / 90.01 | 4.32 / 71.72 | 5.39±0.00 | 5.27 / 32.61 | 3.23 / 21.25 | 0.20 / 1.48 | 0.00±0.00 | 953.8±186.1 |
| sft | 57.57 / 96.97 | 24.30 / 72.50 | 49.06 / 89.60 | 28.24 / **98.35** | **20.60** / **96.46** | 20.20±0.00 | 22.27 / 63.74 | 26.02 / 65.00 | 3.18 / 17.36 | 40.62±8.28 | 517.1±176.9 |
| grpo | 71.78 / **98.48** | **42.11** / 85.00 | 61.65 / **92.20** | 21.71 / 97.52 | 9.77 / 84.34 | 13.76±0.00 | 21.98 / 58.67 | 25.39 / 65.61 | 2.36 / 16.49 | 0.00±0.00 | 775.5±228.6 |
| refl_dg520 | **79.69** / 96.36 | 39.61 / **90.00** | **66.46** / 89.00 | **28.78** / 97.58 | 18.59 / 92.93 | 22.54±0.00 | **28.35** / **67.84** | **35.04** / **72.04** | **3.68** / **20.07** | 33.13±19.78 | 613.4±233.9 |

Notes: accuracy and trigger rate are mean ± sample std across seeds, reported as percentages. Response length is the average number of generated tokens. Bold indicates the best mean within each dataset column. `mmlupro` only has one seed under `test/`, so its std is `0.00`.
