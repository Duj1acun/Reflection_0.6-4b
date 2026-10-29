# REFLECT+ Rebuttal Offline Analyses

## Token and compute summary

Mean tokens count the complete decoded answer text. For REFLECT+, this is the sum over all clean-slate rounds, not a single round.

| Dataset | GRPO acc | GRPO tok | REFLECT+ acc | REFLECT+ tok | Tok delta | GRPO acc/1k tok | REFLECT+ acc/1k tok | REFLECT+ trig |
|---|---|---|---|---|---|---|---|---|
| math500 | 61.7 | 722 | 66.5 | 615 | -14.8 | 0.854 | 1.080 | 18.3 |
| amc23 | 42.1 | 1020 | 39.6 | 852 | -16.4 | 0.413 | 0.465 | 27.5 |
| gsm8k | 71.8 | 468 | 79.7 | 341 | -27.0 | 1.534 | 2.335 | 12.2 |
| logiqa | 21.7 | 684 | 28.8 | 420 | -38.6 | 0.317 | 0.685 | 51.0 |
| gpqa | 9.8 | 984 | 18.6 | 838 | -14.9 | 0.099 | 0.222 | 56.7 |
| mmlu_pro | 13.8 | 611 | 22.5 | 485 | -20.7 | 0.225 | 0.465 | 53.6 |

## REFLECT+ by reflection depth

| Dataset | # refl | Samples | Rate | Acc | Mean tok | Median tok | Budget-hit |
|---|---|---|---|---|---|---|---|
| math500 | 0 | 13075 | 81.7 | 74.8 | 495 | 439 | 4.3 |
| math500 | 1 | 2891 | 18.1 | 29.4 | 1144 | 1176 | 4.1 |
| math500 | 2 | 34 | 0.2 | 14.7 | 1926 | 1974 | 0.0 |
| amc23 | 0 | 928 | 72.5 | 49.5 | 664 | 678 | 11.5 |
| amc23 | 1 | 348 | 27.2 | 13.8 | 1342 | 1365 | 3.2 |
| amc23 | 2 | 4 | 0.3 | 0.0 | 1910 | 1959 | 0.0 |
| gsm8k | 0 | 37064 | 87.8 | 84.9 | 303 | 283 | 0.9 |
| gsm8k | 1 | 5095 | 12.1 | 42.6 | 612 | 513 | 0.2 |
| gsm8k | 2 | 49 | 0.1 | 20.4 | 1139 | 955 | 0.0 |
| logiqa | 0 | 24657 | 49.0 | 30.0 | 250 | 124 | 4.7 |
| logiqa | 1 | 25512 | 50.7 | 27.6 | 579 | 409 | 0.6 |
| logiqa | 2 | 135 | 0.3 | 25.9 | 1469 | 1482 | 0.0 |
| gpqa | 0 | 2743 | 43.3 | 17.8 | 476 | 430 | 12.2 |
| gpqa | 1 | 3517 | 55.5 | 19.2 | 1101 | 1064 | 4.2 |
| gpqa | 2 | 76 | 1.2 | 19.7 | 1686 | 1723 | 0.0 |
| mmlu_pro | 0 | 5579 | 46.4 | 23.8 | 371 | 298 | 8.9 |
| mmlu_pro | 1 | 6444 | 53.6 | 21.4 | 582 | 268 | 6.6 |
| mmlu_pro | 2 | 9 | 0.1 | 11.1 | 1910 | 1970 | 11.1 |

## Paired bootstrap confidence intervals

### REFLECT+ minus grpo

| Dataset | Questions | Mean@N delta | 95% CI |
|---|---|---|---|
| math500 | 500 | 4.8 | [3.5, 6.2] |
| amc23 | 40 | -2.5 | [-7.8, 2.7] |
| gsm8k | 1319 | 7.9 | [6.9, 8.9] |
| logiqa | 1572 | 7.1 | [6.0, 8.2] |
| gpqa | 198 | 8.8 | [6.9, 11.0] |
| mmlu_pro | 12032 | 8.8 | [8.0, 9.6] |
| overall avg | - | 5.8 | [4.8, 6.8] |

Overall pass@N delta: 2.9 [1.0, 4.8]

### REFLECT+ minus grpo500

| Dataset | Questions | Mean@N delta | 95% CI |
|---|---|---|---|
| math500 | 500 | 3.1 | [1.7, 4.5] |
| amc23 | 40 | -2.9 | [-8.0, 2.5] |
| gsm8k | 1319 | 5.9 | [5.0, 6.9] |
| logiqa | 1572 | 5.6 | [4.6, 6.6] |
| gpqa | 198 | 8.2 | [6.2, 10.4] |
| mmlu_pro | 12032 | 9.4 | [8.6, 10.2] |
| overall avg | - | 4.9 | [3.8, 5.9] |

Overall pass@N delta: 2.6 [0.6, 4.7]

## Existing checkpoint ablations

These are already-completed checkpoints on the common five-dataset subset; no new training is implied.

| Model | # datasets | Avg mean@N | Avg trigger | Per-dataset mean@N |
|---|---|---|---|---|
| grpo | 5 | 18.4 | 0.0 | math500:61.7 / aime24:10.0 / aime25:4.8 / aime26:5.9 / gpqa:9.8 |
| grpo500 | 5 | 19.1 | 0.0 | math500:63.4 / aime24:11.9 / aime25:4.6 / aime26:5.1 / gpqa:10.4 |
| refl_decoupled | 5 | 19.3 | 17.5 | math500:69.7 / aime24:6.4 / aime25:5.5 / aime26:3.0 / gpqa:11.8 |
| refl_dg519 | 5 | 16.5 | 75.0 | math500:60.5 / aime24:4.3 / aime25:2.6 / aime26:3.5 / gpqa:11.8 |
| refl_dg520 | 5 | 20.1 | 47.2 | math500:66.5 / aime24:7.9 / aime25:5.2 / aime26:2.5 / gpqa:18.6 |

## Training dynamics from existing logs

Parsed from local DG run logs. Blank validation cells indicate steps without validation.

| Step | Val acc | Val trig | Train trig | Format valid | Direct acc | Refl acc | Resp len |
|---|---|---|---|---|---|---|---|
| 0 | 62.4 | 11.6 | - | - | - | - | - |
| 1 | - | - | 32.2 | 98.4 | 68.0 | 39.4 | 619.0 |
| 10 | 62.0 | 12.8 | 21.3 | 97.5 | 71.2 | 33.9 | 587.9 |
| 20 | 61.8 | 12.2 | 27.1 | 96.9 | 66.0 | 30.9 | 694.3 |
| 50 | 65.4 | 15.2 | 27.7 | 96.5 | 71.9 | 33.8 | 687.8 |
| 100 | 68.0 | 12.6 | 25.2 | 97.5 | 73.6 | 36.4 | 633.5 |
| 150 | 69.2 | 10.0 | 18.2 | 97.5 | 75.2 | 32.3 | 605.2 |
| 200 | 73.2 | 11.6 | - | - | - | - | - |
| 240 | 71.0 | 12.6 | 18.2 | 98.0 | 73.7 | 28.0 | 598.2 |
| 300 | 71.6 | 14.0 | 18.9 | 98.6 | 79.5 | 25.8 | 562.9 |
| 350 | 72.6 | 12.0 | 13.1 | 97.5 | 77.1 | 25.4 | 496.1 |
| 400 | 73.0 | 13.0 | 9.2 | 99.0 | 79.4 | 29.8 | 498.8 |
| 450 | 71.2 | 9.0 | 11.1 | 98.8 | 82.6 | 15.8 | 522.4 |
