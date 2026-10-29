# Qwen3-0.6B SFT Teacher Semantic Quality Report

This report revalidates the existing MiniMax teacher outputs. No paid API call
was made for this revalidation.

## Scope

- Pilot trajectories: 100
- Correct Qwen3 trajectories retained as perfect examples: 8
- Incorrect Qwen3 trajectories requiring teacher transformation: 92
- Existing teacher outputs: 91
- Missing teacher output: 1

## Corrected acceptance rule

First-error localization is semantic. The reflection must identify the first
wrong reasoning direction or mathematical step, but it does **not** need to
quote, copy, or exactly match a span from the student response.

A teacher record is content-valid only when:

1. a complete structured payload can be recovered;
2. `verified_states`, `error_reason`, `new_direction`, and corrected
   `reasoning` are non-empty;
3. the reflection has exactly one complete `<reflection>...</reflection>`
   block;
4. the corrected reasoning passes the local answer-level math verifier.

Exact `faulty_quote` and exact-prefix matching are not quality gates.

## Results

- Content-valid reflective records: 81 / 91 (89.01%)
- Content quarantine: 10 / 91 (10.99%)
  - no complete valid teacher payload: 7
  - corrected reasoning fails the local answer verifier: 3
- Deterministic manual audit: 20 / 20 reflections identify a reasonable error
  direction and give a coherent correction direction.
- Duplicate error diagnoses: 0
- Required-field character ranges:
  - `verified_states`: 40--280 (median 117)
  - `error_reason`: 63--198 (median 132)
  - `new_direction`: 65--216 (median 120)
  - corrected `reasoning`: 117--1700 (median 449)

The manual audit used 20 evenly spaced records after sorting the 81 valid
records by `problem_id`. This is a deterministic, repeatable audit rather than
a hand-picked sample.

## Training-length status

Content validity is reported separately from sequence-length compatibility.
With the current full-student-completion Phase-1 context and a strict 2,048-token
limit:

- content- and length-valid reflective records: 74
- content-valid but overlength reflective records: 7

Together with the 8 perfect trajectories, 82 of the 100 pilot trajectories are
immediately usable under that strict context/length policy. Overlength records
must not be silently truncated.

## Artifacts

- `teacher_semantic_content_validated_v2.jsonl`: semantic/content validation
- `teacher_semantic_content_validated_v2.jsonl.report.json`: content report and
  hashes
- `teacher_semantic_validated_v2.jsonl`: semantic plus 2,048-token validation
- `teacher_semantic_validated_v2.jsonl.report.json`: formal-length report and
  hashes

Only records marked `validation_status="valid"` in the relevant validated file
may enter a compiled dataset. The 10 content-invalid records and the one missing
teacher record remain excluded; no regeneration is required to use the valid
subset.
