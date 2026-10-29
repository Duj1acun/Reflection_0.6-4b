# Reflect step450: full nine-benchmark evaluation configuration

## Run identity

- Model: `reflect_step450_hf` (the HF export of the Reflect training checkpoint at global step 450)
- Evaluation directory: `step450_eval9_full_n32_v2_20260923`
- Scope: full splits (`limit: null`), 31,898 questions and 1,020,736 generated samples
- Sampling protocol: 32 completions/question; temperature 0.7; top-p 0.8; top-k 20; seed 1234
- Runtime: vLLM, bfloat16; tensor parallelism 4; 4 GPUs; GPU-memory utilization 0.88; eager mode; maximum model context 16,384 tokens; question batch size/max sequences 512; maximum batched tokens 8,192.
- Global generation ceiling: 2,048 tokens. Per-dataset overrides are listed below.

## Benchmarks and data sources

| Group | Benchmark | Split / source | Questions | Generation limit | Scoring |
|---|---|---|---:|---:|---|
| reasoning3 | BBH | Official BBH local copy: all `bbh/*.json`, with matching official `cot-prompts/*.txt` | 6,511 | 2,048 | Case/punctuation/article-normalized exact match on terminal answer |
| reasoning3 | DROP | AllenAI DROP `drop_dataset_dev.json` | 9,536 | 32 | Official AllenAI DROP answer normalization, best exact match and F1 over validated answers |
| reasoning3 | CLUTRR | Official systematic-generalization `1.4_test.csv` | 190 | 32 | Relation-label exact match; constrained to the 18 labels in the file |
| standard6 | MATH500 | Local `math500.parquet` | 500 | 2,048 | Structured mathematical equivalence |
| standard6 | AMC23 | `math-ai/amc23`, `test` split | 40 | 1,024 | Exact integer-answer mathematical equivalence |
| standard6 | GSM8K | `openai/gsm8k`, `main`/`test` split | 1,319 | 1,024 | Structured mathematical equivalence |
| standard6 | LogiQA | `jeggers/logiqa2_formatted`, `test` split | 1,572 | 16 | Last valid `Answer: X` extraction, then option-label exact match |
| standard6 | GPQA | `Idavidrein/gpqa`, `gpqa_diamond`/`train` split | 198 | 16 | Final A–D extraction and exact match; choices deterministically shuffled per record ID |
| standard6 | MMLU-Pro | `TIGER-Lab/MMLU-Pro`, `test` split | 12,032 | 16 | Last valid `Answer: X` extraction, then option-label exact match |

For MATH500, AMC23 and GSM8K the instruction requires the final line to be exactly `\\boxed{ANSWER}`. For the three MCQ datasets it requires only `Answer: X`. BBH uses the official few-shot chain-of-thought prompt. DROP asks for only the answer span, number or date; CLUTRR asks for exactly one relation label.

## Aggregation

Each benchmark reports:

- `pass_at_n`: fraction of questions with at least one correct completion among 32;
- `mean_at_n`: fraction of all 32×questions completions that are correct;
- `exact_match` and `f1`: mean sample-level scores (for all non-DROP benchmarks these equal `mean_at_n`; DROP retains official F1).

The evaluator removes complete `<think>...</think>` blocks before scoring, treats malformed thinking tags as empty answers, and records this in the samples. It validates that every question has exactly 32 completions before publication.

## Legacy UID repair for this run

The original full-run samples contained one duplicate source UID in DROP and four in LogiQA. They were distinct questions, each with a complete 32-sample group. The saved samples were re-summarized using `dataset + original UID + SHA-256(problem, gold)` as their stable legacy question identity, restoring the intended 9,536 DROP and 1,572 LogiQA questions without regenerating model output. Future runs use disambiguated source IDs: DROP appends deterministic row position; LogiQA appends test-split position.

The repaired results are in `reasoning3/summary.json` and `standard6/summary.json`; the original per-group machine-readable settings remain in `reasoning3/config.json` and `standard6/config.json`.
