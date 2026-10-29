# Context Token Accounting

Processed tokens are counted as first-round prompt + all decoded tokens + clean-slate continuation prompts.
For REFLECT+, each continuation prompt is reconstructed as `header + latest <reflection>...</reflection>`.

| Dataset | Seeds | N | GRPO acc | REFLECT+ acc | Acc delta | REF trig | GRPO decoded | REF decoded | Decoded delta | GRPO processed | REF processed | Processed delta | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| aime24 | 1 | @32 | 10.0 | 7.9 | -2.1 | 52.4 | 1240.7 | 1102.0 | -11.2 | 1375.8 | 1445.7 | 5.1 | results |
| aime25 | 1 | @32 | 4.8 | 5.2 | 0.4 | 51.8 | 1291.0 | 1083.4 | -16.1 | 1493.1 | 1534.3 | 2.8 | results |
| aime26 | 1 | @32 | 5.9 | 2.5 | -3.4 | 56.9 | 1278.7 | 1183.9 | -7.4 | 1431.2 | 1574.4 | 10.0 | results |
| amc23 | 5 | @32 | 43.4 | 39.5 | -3.8 | 26.8 | 990.0 | 868.9 | -12.2 | 1102.6 | 1097.4 | -0.5 | results_seeded |
| bbh_all | 5 | @8 | 22.4 | 28.8 | 6.4 | 52.7 | 699.0 | 607.8 | -13.0 | 846.9 | 941.8 | 11.2 | results_bbh_all_pass8_t1_boxed_5seed |
| clutrr | 5 | @8 | 2.6 | 4.0 | 1.4 | 78.0 | 569.7 | 734.2 | 28.9 | 750.2 | 1191.8 | 58.9 | results_clutrr_drop_pass8_t1_boxed_5seed |
| drop_dev | 5 | @8 | 27.3 | 35.7 | 8.4 | 43.2 | 514.3 | 373.9 | -27.3 | 861.5 | 968.8 | 12.5 | results_clutrr_drop_pass8_t1_boxed_5seed |
| gpqa | 5 | @32 | 10.7 | 18.8 | 8.0 | 59.7 | 995.5 | 960.6 | -3.5 | 1272.9 | 1544.9 | 21.4 | results_seeded |
| gsm8k | 5 | @32 | 71.7 | 80.1 | 8.4 | 11.8 | 466.7 | 345.1 | -26.1 | 556.0 | 493.9 | -11.2 | results_seeded |
| logiqa | 5 | @32 | 22.4 | 29.3 | 7.0 | 54.3 | 665.7 | 505.3 | -24.1 | 926.6 | 1009.4 | 8.9 | results_seeded |
| math500 | 5 | @32 | 62.1 | 66.4 | 4.4 | 18.1 | 721.9 | 628.8 | -12.9 | 820.8 | 817.3 | -0.4 | results_seeded |
| mmlu_pro | 1 | @1 | 13.8 | 22.5 | 8.8 | 53.6 | 611.2 | 485.0 | -20.7 | 878.4 | 1000.2 | 13.9 | results |
