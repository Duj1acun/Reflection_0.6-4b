#!/usr/bin/env python3
"""Build backbone-specific cold-start SFT data with auditable stages.

Examples are documented in ``SFT_DATA_PIPELINE.md``.  Every long-running stage
uses JSONL append-only checkpoints keyed by a stable record identifier so it can
resume without resampling or paying for the same teacher request twice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rgrpo.chat_template import apply_chat_template
from rgrpo.sft_data import (
    HARD_THRESHOLD,
    SYSTEM_PROMPT,
    append_jsonl,
    compile_sft_records,
    extract_json_object,
    extract_json_objects,
    read_jsonl,
    select_hard_records,
    select_pilot_records,
    sha256_file,
    training_lengths,
    validate_teacher_payload,
    write_jsonl,
)


DEFAULT_DATASET = "xinpeng/Big-Math-RL-Verified-Combined-digit-hard"
DEFAULT_REVISION = "46b932ff794de16b6f96a7f3555659403e4eba2e"
DEFAULT_SOURCE_DIR = Path("/mnt/data/reflection_grpo/data/sft/source")
DEFAULT_MODEL = Path("/mnt/data/models/Qwen3-0.6B")
DEFAULT_MODEL_DATA = Path("/mnt/data/reflection_grpo/data/sft/qwen3-0.6b")
STUDENT_SYSTEM_PROMPT = (
    "You are a mathematical reasoning expert. Solve the problem step by step "
    "and put the final answer in \\boxed{}."
)


def dump_json(path: str | Path, value: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def completed_ids(path: str | Path) -> set[str]:
    output = Path(path)
    if not output.exists():
        return set()
    ids: list[str] = [str(row.get("problem_id")) for row in read_jsonl(output)]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate problem_id found in resumable output {output}")
    return set(ids)


def _trajectory_id(pid: str, sample_index: int) -> str:
    if sample_index < 0:
        raise ValueError("sample_index must be non-negative")
    return f"{pid}:sample:{sample_index}"


def _sampling_record_id(row: dict[str, Any]) -> str:
    saved = str(row.get("trajectory_id") or "").strip()
    if saved:
        return saved
    return _trajectory_id(str(row["problem_id"]), int(row.get("sample_index", 0)))


def _completed_sampling_ids(path: str | Path) -> set[str]:
    output = Path(path)
    if not output.exists():
        return set()
    ids = [_sampling_record_id(row) for row in read_jsonl(output)]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate trajectory_id found in resumable output {output}")
    return set(ids)


def _expand_sampling_records(
    records: Iterable[dict[str, Any]], samples_per_problem: int
) -> list[dict[str, Any]]:
    if samples_per_problem <= 0:
        raise ValueError("samples_per_problem must be positive")
    expanded: list[dict[str, Any]] = []
    for row in records:
        pid = str(row["problem_id"])
        for sample_index in range(samples_per_problem):
            expanded.append(
                {
                    **row,
                    "sample_index": sample_index,
                    "trajectory_id": _trajectory_id(pid, sample_index),
                }
            )
    return expanded


def _select_shard(
    records: Iterable[dict[str, Any]], num_shards: int, shard_index: int
) -> list[dict[str, Any]]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    return [row for index, row in enumerate(records) if index % num_shards == shard_index]


def _download(url: str, output: Path, verify_tls: bool) -> dict[str, Any]:
    import requests

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size > 0:
        return {
            "url": url,
            "path": str(output),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
            "reused": True,
        }
    partial = output.with_suffix(output.suffix + ".part")
    with requests.get(url, stream=True, timeout=120, verify=verify_tls) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)
    partial.replace(output)
    return {
        "url": url,
        "path": str(output),
        "bytes": output.stat().st_size,
        "sha256": sha256_file(output),
        "reused": False,
    }


def command_select(args: argparse.Namespace) -> None:
    import pyarrow.parquet as pq

    output_dir = Path(args.output_dir)
    download_dir = output_dir / "downloads" / args.revision
    downloads: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for split in ("train", "test"):
        filename = f"{split}-00000-of-00001.parquet"
        url = (
            f"https://huggingface.co/datasets/{args.dataset}/resolve/"
            f"{args.revision}/data/{filename}"
        )
        local = download_dir / filename
        info = _download(url, local, verify_tls=not args.allow_insecure_tls)
        downloads.append(info)
        split_rows = pq.read_table(local).to_pylist()
        for row in split_rows:
            row["source_split"] = split
        rows.extend(split_rows)

    selected, conflicts, stats = select_hard_records(rows, threshold=args.threshold)
    if args.expected_eligible and stats["eligible_rows"] != args.expected_eligible:
        raise RuntimeError(
            f"Expected {args.expected_eligible} eligible rows, got {stats['eligible_rows']}. "
            "The source or filter changed; refusing to silently continue."
        )

    selected_path = output_dir / "big_math_hard_le_1of64.jsonl"
    conflicts_path = output_dir / "duplicate_conflicts.jsonl"
    write_jsonl(selected_path, selected)
    write_jsonl(conflicts_path, conflicts)
    manifest = {
        "schema_version": 1,
        "created_unix": int(time.time()),
        "dataset": args.dataset,
        "revision": args.revision,
        "splits_merged": ["train", "test"],
        "normalization": "Unicode NFKC + whitespace collapse + strip",
        "conflict_policy": "drop entire normalized-question group",
        "downloads": downloads,
        "selection": stats,
        "outputs": {
            "questions": {
                "path": str(selected_path),
                "rows": len(selected),
                "sha256": sha256_file(selected_path),
            },
            "conflicts": {
                "path": str(conflicts_path),
                "rows": len(conflicts),
                "sha256": sha256_file(conflicts_path),
            },
        },
    }
    dump_json(output_dir / "source_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _sampling_manifest(args: argparse.Namespace, source: Path, count: int) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "selected_count": count,
        "selected_problem_count": count,
        "selected_trajectory_count": count * args.samples_per_problem,
        "model": str(Path(args.model).resolve()),
        "engine": args.engine,
        "system_prompt": STUDENT_SYSTEM_PROMPT,
        "enable_thinking": False,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "samples_per_problem": args.samples_per_problem,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "checkpoint_batch_size": args.batch_size,
        "shard": {
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
        },
    }


def _stable_seed(seed: int, pid: str, sample_index: int = 0) -> int:
    payload = f"{seed}:{pid}:{sample_index}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16) % (2**31 - 1)


def _create_vllm(args: argparse.Namespace):
    from vllm import LLM, SamplingParams

    del SamplingParams
    return LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        dtype="bfloat16",
        enable_prefix_caching=True,
        max_num_seqs=max(1, args.max_num_seqs),
    )


def _sample_vllm(
    args: argparse.Namespace,
    llm: Any,
    records: list[dict[str, Any]],
    prompts: list[str],
):
    from vllm import SamplingParams

    params = [
        SamplingParams(
            n=1,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
            seed=_stable_seed(args.seed, record["problem_id"], record["sample_index"]),
        )
        for record in records
    ]
    return llm.generate(prompts, params, use_tqdm=False)


def _sample_transformers(args: argparse.Namespace, records: list[dict[str, Any]], prompts: list[str]):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    outputs = []
    for record, prompt in zip(records, prompts):
        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
        torch.manual_seed(
            _stable_seed(args.seed, record["problem_id"], record["sample_index"])
        )
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        token_ids = generated[0, encoded.input_ids.shape[1] :].tolist()
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        finish_reason = "length" if len(token_ids) >= args.max_new_tokens else "stop"
        outputs.append(
            {
                "text": text,
                "token_ids": token_ids,
                "finish_reason": finish_reason,
            }
        )
    return outputs


def command_sample(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    os.environ["CHAT_TEMPLATE_ENABLE_THINKING"] = "false"
    source = Path(args.source)
    all_records = list(read_jsonl(source))
    selected_records = (
        select_pilot_records(all_records, args.limit, seed=args.seed)
        if args.pilot
        else list(all_records if args.limit <= 0 else all_records[: args.limit])
    )
    records = _select_shard(selected_records, args.num_shards, args.shard_index)
    trajectories = _expand_sampling_records(records, args.samples_per_problem)
    output = Path(args.output)
    done = _completed_sampling_ids(output) if args.resume else set()
    pending = [row for row in trajectories if row["trajectory_id"] not in done]
    print(
        f"[Sample] source_selected_problems={len(selected_records)} "
        f"shard={args.shard_index}/{args.num_shards} selected_problems={len(records)} "
        f"selected_trajectories={len(trajectories)} complete={len(done)} "
        f"pending={len(pending)}"
    )
    if not pending:
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = [
        apply_chat_template(
            tokenizer,
            [
                {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
                {"role": "user", "content": row["problem"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in pending
    ]
    if args.dry_run:
        print(prompts[0])
        print(json.dumps(_sampling_manifest(args, source, len(records)), indent=2))
        return

    started = time.monotonic()
    batch_size = max(1, args.batch_size)
    llm = _create_vllm(args) if args.engine == "vllm" else None
    for batch_start in range(0, len(pending), batch_size):
        batch_records = pending[batch_start : batch_start + batch_size]
        batch_prompts = prompts[batch_start : batch_start + batch_size]
        if args.engine == "vllm":
            generations = _sample_vllm(args, llm, batch_records, batch_prompts)
            values = []
            for generation in generations:
                choice = generation.outputs[0]
                values.append(
                    {
                        "text": choice.text,
                        "token_ids": list(choice.token_ids),
                        "finish_reason": choice.finish_reason,
                    }
                )
        else:
            values = _sample_transformers(args, batch_records, batch_prompts)

        if len(values) != len(batch_records):
            raise RuntimeError(
                f"Generation count mismatch: expected {len(batch_records)}, got {len(values)}"
            )
        for row, prompt, value in zip(batch_records, batch_prompts, values):
            completion = value["text"].strip()
            result = {
                **row,
                "model": Path(args.model).name,
                "model_path": str(Path(args.model).resolve()),
                "prompt": prompt,
                "completion": completion,
                "finish_reason": value["finish_reason"],
                "prompt_tokens": len(tokenizer(prompt, add_special_tokens=False)["input_ids"]),
                "completion_tokens": len(value["token_ids"]),
                "has_think_tag": "<think>" in completion or "</think>" in completion,
                "sampling": {
                    "enable_thinking": False,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "seed": _stable_seed(
                        args.seed, row["problem_id"], row["sample_index"]
                    ),
                    "max_new_tokens": args.max_new_tokens,
                },
            }
            append_jsonl(output, result)
        completed = min(batch_start + len(batch_records), len(pending))
        print(f"[Sample] checkpointed {completed}/{len(pending)} pending trajectories")

    manifest = _sampling_manifest(args, source, len(records))
    manifest.update(
        {
            "output": str(output),
            "output_rows": len(list(read_jsonl(output))),
            "output_sha256": sha256_file(output),
            "elapsed_seconds_this_run": round(time.monotonic() - started, 3),
        }
    )
    dump_json(output.with_suffix(output.suffix + ".manifest.json"), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def command_merge_samples(args: argparse.Namespace) -> None:
    source = Path(args.source)
    source_records = list(read_jsonl(source))
    expected_records = _expand_sampling_records(source_records, args.samples_per_problem)
    expected_ids = [str(row["trajectory_id"]) for row in expected_records]
    expected_id_set = set(expected_ids)

    shard_rows: dict[str, dict[str, Any]] = {}
    shard_metadata: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for input_name in args.input:
        input_path = Path(input_name)
        manifest_path = input_path.with_suffix(input_path.suffix + ".manifest.json")
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing shard manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifests.append(manifest)
        input_count = 0
        for row in read_jsonl(input_path):
            record_id = _sampling_record_id(row)
            if record_id not in expected_id_set:
                raise ValueError(f"Unexpected trajectory_id {record_id} in {input_path}")
            if record_id in shard_rows:
                raise ValueError(f"Duplicate trajectory_id across shards: {record_id}")
            shard_rows[record_id] = row
            input_count += 1
        shard_metadata.append(
            {
                "path": str(input_path),
                "rows": input_count,
                "sha256": sha256_file(input_path),
                "manifest_path": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
            }
        )

    if not manifests:
        raise ValueError("At least one --input shard is required")
    common_keys = (
        "source_sha256",
        "model",
        "engine",
        "system_prompt",
        "enable_thinking",
        "temperature",
        "top_p",
        "samples_per_problem",
        "seed",
        "max_new_tokens",
        "max_model_len",
        "max_num_seqs",
        "checkpoint_batch_size",
    )
    baseline = manifests[0]
    for manifest in manifests:
        for key in common_keys:
            if manifest.get(key) != baseline.get(key):
                raise ValueError(f"Shard manifest mismatch for {key}")
    if baseline.get("source_sha256") != sha256_file(source):
        raise ValueError("Shard source SHA256 does not match merge source")
    if int(baseline.get("samples_per_problem", 0)) != args.samples_per_problem:
        raise ValueError("Shard samples_per_problem does not match merge request")

    missing = [record_id for record_id in expected_ids if record_id not in shard_rows]
    if missing:
        raise ValueError(
            f"Cannot merge incomplete shards: missing {len(missing)} trajectories; "
            f"first={missing[0]}"
        )

    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output}; pass --overwrite to replace it")
    ordered_rows = [shard_rows[record_id] for record_id in expected_ids]
    write_jsonl(output, ordered_rows)
    manifest = {
        **{key: baseline.get(key) for key in common_keys},
        "schema_version": 2,
        "source_path": str(source),
        "selected_count": len(source_records),
        "selected_problem_count": len(source_records),
        "selected_trajectory_count": len(expected_records),
        "shards": shard_metadata,
        "output": str(output),
        "output_rows": len(ordered_rows),
        "output_sha256": sha256_file(output),
    }
    dump_json(output.with_suffix(output.suffix + ".manifest.json"), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def command_score(args: argparse.Namespace) -> None:
    from rgrpo.math_score_light import compute_math_score_payload, extract_answer

    input_path = Path(args.input)
    output_path = Path(args.output)
    done = _completed_sampling_ids(output_path) if args.resume else set()
    counts: Counter[str] = Counter()
    for row in read_jsonl(input_path):
        if _sampling_record_id(row) in done:
            continue
        payload = compute_math_score_payload(
            {"solution_str": row.get("completion", ""), "ground_truth": row.get("answer", "")}
        )
        diagnostics = {key: value for key, value in payload.items() if key != "score"}
        if payload.get("math_score_timeout") or payload.get("math_score_exception"):
            status = "quarantine"
            reason = "verifier_timeout_or_exception"
        else:
            status = "correct" if float(payload.get("score", 0)) == 1.0 else "incorrect"
            reason = ""
        result = {
            **row,
            "predicted_answer": extract_answer(str(row.get("completion", ""))),
            "math_score": float(payload.get("score", 0)),
            "score_status": status,
            "score_reason": reason,
            "score_diagnostics": diagnostics,
        }
        append_jsonl(output_path, result)
        counts[status] += 1
    all_rows = list(read_jsonl(output_path))
    total_counts = Counter(str(row.get("score_status")) for row in all_rows)
    report = {
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "rows": len(all_rows),
        "status_counts": dict(sorted(total_counts.items())),
        "new_this_run": dict(sorted(counts.items())),
    }
    dump_json(output_path.with_suffix(output_path.suffix + ".report.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _reasoning_text(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(value)


def _teacher_prompt(
    row: dict[str, Any],
    previous_errors: list[str] | None = None,
    previous_analysis: str = "",
) -> str:
    retry = ""
    if previous_errors:
        retry = "\nYour previous output failed validation: " + ", ".join(previous_errors) + "."
    continuation = ""
    if previous_analysis:
        continuation = f"""

A previous attempt exhausted its output budget before returning the JSON. Continue from its
scratch work below: do not restart the analysis, do not repeat it, and immediately finish the
missing derivation and required JSON. Correct the scratch work if necessary.

PRIOR_ANALYSIS:
{previous_analysis}
END_PRIOR_ANALYSIS
"""
    return f"""Verify the student's solution step by step, in its original order, and identify the first mathematical reasoning error.

Verification procedure:
1. Check each step against the problem conditions and the preceding valid steps: definitions,
   assumptions, domains, algebra, arithmetic, logical implications, case coverage, and theorem
   prerequisites. A correct final answer alone does not make an invalid derivation correct.
2. Locate the earliest false claim, unjustified inference, or omitted necessary case. Do not
   skip an earlier error to diagnose a later, more obvious error. If the student later corrects
   it, still identify its first occurrence. A clearly labeled tentative idea that is checked and
   rejected is not by itself an error; do not invent errors in valid alternative approaches.
3. Preserve only correct conclusions actually established before that first error. Do not move
   conclusions from later steps or from your corrected solution into verified_states. Semantic
   summaries are allowed; exact quotations and character offsets are not required. If no useful
   intermediate conclusion precedes the error, say so briefly instead of inventing one.
4. Give a corrected next direction, then verify the corrected solution step by step, including
   its conditions, calculations, completeness, and final answer. Keep this checking in your
   internal analysis; the final response must contain only the JSON below, not a checking trace.

Return exactly one JSON object, without Markdown, using this schema:
{{
  "verdict": "incorrect" or "actually_correct" or "unusable",
  "verified_states": "the correct intermediate conclusions established before the error",
  "error_reason": "one concise semantic diagnosis of the first wrong reasoning direction or step",
  "new_direction": "one concise sentence giving a corrected direction",
  "reasoning": "a concise, self-contained, step-by-step corrected solution ending in the verified actual answer"
}}

Verdict rules:
- incorrect: a mathematical error or an incomplete solution requires correction. If all existing
  steps are valid but the solution stops too early, diagnose incompleteness, not a fabricated error.
  All four text fields must be non-empty.
- actually_correct: the student's solution is mathematically valid and complete; do not fabricate
  an error merely because its wording or answer format differs. Set the four text fields to "".
- unusable: the problem has insufficient conditions or answer-changing ambiguity, or reliable
  reasoning contradicts the reference answer or shows it is incomplete. Explain the issue briefly
  in error_reason and set verified_states, new_direction, and reasoning to "".

GOLD_ANSWER is a reference for checking, not an unquestionable authority. Never add unstated
conditions, discard valid solutions, or invent reasoning to force agreement with it. Resolve
problem/reference defects as unusable before constructing a student correction.

For verdict=incorrect:
- Write verified_states, error_reason, and new_direction from the student's first-person
  perspective. Each must start with the following exact words:
  verified_states: "I have"; error_reason: "I incorrectly"; new_direction: "I will".
  Do not refer to "the student", "the solution", "the derivation", or address "you".
  For an incomplete but otherwise valid attempt, "I incorrectly stopped before ..." is allowed.
  If no useful conclusion precedes the error, write "I have not yet established a useful
  intermediate conclusion." Do not add reflection tags or the outer So far/However/Next template.
- verified_states is a checkpoint of the actual valid prefix, not a list of all correct facts
  found anywhere in the attempt. For each included conclusion, confirm it was established before
  the first error. A true equation first written after that error is still excluded. Do not
  repair an incorrect statement and then claim the repaired version was already established.
- error_reason must identify the first faulty claim or operation by its mathematical meaning
  and briefly explain why it is invalid; do not merely say the final answer is wrong. If the
  first error is evaluating the final requested quantity, identify the faulty computation and
  its failure without supplying the corrected final value in any reflection field.
- new_direction is one concise sentence. Necessary repair formulas and intermediate relations
  are allowed, but do not put the final answer or a complete solution in the reflection fields.
- reasoning must be understandable from PROBLEM and the three reflection fields, without
  referring back to STUDENT_SOLUTION. Give a clear, concise, step-by-step derivation within 1600
  tokens, ending with the actual verified answer in \\boxed{{...}}, not a placeholder.

Before emitting JSON, check the actual requested quantity, not a nearby easier quantity.
Do not silently assume distinctness, completeness of a listed set, an unstated sampling rule,
or that a stated subtotal equals the total. If alternative interpretations allowed by the text
change the answer and cannot be resolved from explicit context, use unusable; do not infer the
missing condition from GOLD_ANSWER. Do not reject a problem merely for wording differences that
do not affect its answer.

Final silent self-check: verdict consistent with problem/reference validity; earliest error
identified; every retained state precedes it; the three required first-person openings present
for incorrect; no final answer in those three fields; corrected reasoning independently valid.
If a check fails, fix the JSON before returning it. Do not add fields or print the checklist.

Return valid JSON with exactly these five fields. Escape every LaTeX backslash in JSON strings:
for example, the JSON source string "\\\\boxed{{2}}" must decode to the text \\boxed{{2}}.
Do not emit unescaped control characters, Markdown fences, or any additional text.{retry}

PROBLEM:
{row['problem']}

GOLD_ANSWER:
{row['answer']}

STUDENT_SOLUTION:
{row['completion']}{continuation}"""


def command_teacher(args: argparse.Namespace) -> None:
    from openai import OpenAI
    from transformers import AutoTokenizer

    from rgrpo.math_score_light import compute_math_score

    os.environ["CHAT_TEMPLATE_ENABLE_THINKING"] = "false"
    input_path = Path(args.input)
    output_path = Path(args.output)
    queue = [row for row in read_jsonl(input_path) if row.get("score_status") == "incorrect"]
    if args.offset > 0:
        queue = queue[args.offset :]
    if args.limit > 0:
        queue = queue[: args.limit]
    done = completed_ids(output_path) if args.resume else set()
    queue = [row for row in queue if row["problem_id"] not in done]
    print(f"[Teacher] pending incorrect trajectories: {len(queue)}")
    if args.dry_run:
        if queue:
            print(_teacher_prompt(queue[0]))
        return
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is not set; refusing to call the paid teacher API.")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    client = OpenAI(api_key=api_key, base_url=args.base_url, timeout=args.timeout)
    bootstrap: dict[str, tuple[str, list[str]]] = {}
    if args.bootstrap_teacher:
        for saved in read_jsonl(Path(args.bootstrap_teacher)):
            for saved_attempt in reversed(saved.get("attempts", [])):
                prior = _reasoning_text(saved_attempt.get("reasoning"))
                candidate = str(saved_attempt.get("content") or "")
                if candidate:
                    prior += "\n\nPREVIOUS_CANDIDATE_JSON:\n" + candidate
                if prior:
                    saved_errors = [
                        str(error) for error in saved_attempt.get("validation_errors", [])
                    ]
                    bootstrap[str(saved.get("problem_id"))] = (prior, saved_errors)
                    break

    def scorer(generated: str, answer: Any) -> float:
        return compute_math_score(generated=generated, ground_truth=answer)

    def process(row: dict[str, Any]) -> dict[str, Any]:
        previous_analysis, previous_errors = bootstrap.get(str(row["problem_id"]), ("", []))
        attempts: list[dict[str, Any]] = []
        for attempt in range(1, args.max_attempts + 1):
            try:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": "You are a rigorous mathematical data annotator."},
                        {
                            "role": "user",
                            "content": _teacher_prompt(row, previous_errors, previous_analysis),
                        },
                    ],
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_completion_tokens=args.max_tokens,
                    extra_body={"reasoning_split": True},
                )
                choice = response.choices[0]
                message = choice.message
                content = message.content or ""
                usage = response.usage.model_dump() if response.usage else {}
                attempt_record = {
                    "attempt": attempt,
                    "response_id": response.id,
                    "finish_reason": choice.finish_reason,
                    "content": content,
                    "reasoning": (
                        getattr(message, "reasoning_content", None)
                        or getattr(message, "reasoning_details", None)
                    ),
                    "usage": usage,
                }
                reasoning_value = attempt_record.get("reasoning")
                candidates = [("content", value) for value in extract_json_objects(content)]
                candidates.extend(
                    ("reasoning", value)
                    for value in reversed(extract_json_objects(_reasoning_text(reasoning_value)))
                )
                normalized = None
                errors = ["teacher_response_does_not_contain_json"]
                for payload_source, payload in candidates:
                    try:
                        candidate, candidate_errors = validate_teacher_payload(
                            payload,
                            row,
                            scorer=scorer,
                            tokenizer=tokenizer,
                            max_len=args.max_len,
                        )
                        if candidate and candidate.get("verdict") == "actually_correct":
                            if scorer(str(row.get("completion", "")), row.get("answer")) != 1.0:
                                candidate = None
                                candidate_errors = [
                                    "teacher_actually_correct_not_confirmed_by_local_verifier"
                                ]
                    except Exception as exc:
                        candidate = None
                        candidate_errors = [f"{type(exc).__name__}: {exc}"]
                    if candidate is not None:
                        normalized = candidate
                        errors = []
                        attempt_record["validated_payload_source"] = payload_source
                        break
                    errors = candidate_errors
                attempt_record["validation_errors"] = errors
                attempts.append(attempt_record)
                prior = _reasoning_text(attempt_record.get("reasoning"))
                if content:
                    prior += "\n\nPREVIOUS_CANDIDATE_JSON:\n" + content
                if prior:
                    previous_analysis = prior
                if normalized is not None:
                    validation_status = (
                        "quarantine" if normalized.get("verdict") == "unusable" else "valid"
                    )
                    return {
                        "problem_id": row["problem_id"],
                        "validation_status": validation_status,
                        "result": normalized,
                        "attempts": attempts,
                        "teacher_model": args.model,
                    }
                previous_errors = errors
            except Exception as exc:
                previous_errors = [f"{type(exc).__name__}: {exc}"]
                attempts.append(
                    {
                        "attempt": attempt,
                        "exception": previous_errors[0],
                    }
                )
            if attempt < args.max_attempts:
                time.sleep(min(2 ** attempt, 8))
        return {
            "problem_id": row["problem_id"],
            "validation_status": "quarantine",
            "validation_errors": previous_errors,
            "attempts": attempts,
            "teacher_model": args.model,
        }

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(process, row): row["problem_id"] for row in queue}
        for future in as_completed(futures):
            append_jsonl(output_path, future.result())

    rows = list(read_jsonl(output_path))
    counts = Counter(str(row.get("validation_status")) for row in rows)
    usage = Counter()
    for row in rows:
        for attempt in row.get("attempts", []):
            for key, value in (attempt.get("usage") or {}).items():
                if isinstance(value, (int, float)):
                    usage[key] += value
    report = {
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "teacher_model": args.model,
        "base_url": args.base_url,
        "rows": len(rows),
        "status_counts": dict(sorted(counts.items())),
        "usage": dict(sorted(usage.items())),
    }
    dump_json(output_path.with_suffix(output_path.suffix + ".report.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_revalidate_teacher(args: argparse.Namespace) -> None:
    """Re-run local gates over checkpointed teacher attempts without another API call."""
    from transformers import AutoTokenizer

    from rgrpo.math_score_light import compute_math_score

    scored = {str(row["problem_id"]): row for row in read_jsonl(Path(args.scored))}
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    def scorer(generated: str, answer: Any) -> float:
        return compute_math_score(generated=generated, ground_truth=answer)

    input_paths = [Path(args.input), *(Path(path) for path in args.extra_input)]
    saved_by_id: dict[str, dict[str, Any]] = {}
    for input_path in input_paths:
        for saved in read_jsonl(input_path):
            saved_by_id[str(saved.get("problem_id"))] = saved

    output_rows: list[dict[str, Any]] = []
    for saved in saved_by_id.values():
        pid = str(saved.get("problem_id"))
        source = scored.get(pid)
        validation_errors = ["missing_scored_source"]
        selected: dict[str, Any] | None = None
        if source is not None:
            attempts = saved.get("attempts", [])
            staged_solution = ""
            for attempt in reversed(attempts):
                if attempt.get("stage") != "solve":
                    continue
                candidates = extract_json_objects(str(attempt.get("content") or ""))
                candidates.extend(
                    reversed(extract_json_objects(_reasoning_text(attempt.get("reasoning"))))
                )
                for payload in candidates:
                    reasoning = str(payload.get("reasoning", "")).strip()
                    if not reasoning or scorer(reasoning, source.get("answer")) != 1.0:
                        continue
                    lengths = training_lengths(
                        tokenizer,
                        problem=str(source.get("problem", "")),
                        full_reasoning=reasoning,
                    )
                    if all(length <= args.max_len for length in lengths.values()):
                        staged_solution = reasoning
                        break
                if staged_solution:
                    break
            if staged_solution:
                for attempt in reversed(attempts):
                    if attempt.get("stage") != "diagnose":
                        continue
                    candidates = extract_json_objects(str(attempt.get("content") or ""))
                    candidates.extend(
                        reversed(extract_json_objects(_reasoning_text(attempt.get("reasoning"))))
                    )
                    for payload in candidates:
                        combined = {
                            **payload,
                            "verdict": "incorrect",
                            "reasoning": staged_solution,
                        }
                        selected, validation_errors = validate_teacher_payload(
                            combined,
                            source,
                            scorer=scorer,
                            tokenizer=tokenizer,
                            max_len=args.max_len,
                        )
                        if selected is not None:
                            break
                    if selected is not None:
                        break

            for attempt in reversed(attempts if selected is None else []):
                candidates = extract_json_objects(str(attempt.get("content") or ""))
                candidates.extend(
                    reversed(extract_json_objects(_reasoning_text(attempt.get("reasoning"))))
                )
                normalized = None
                for payload in candidates:
                    try:
                        normalized, validation_errors = validate_teacher_payload(
                            payload,
                            source,
                            scorer=scorer,
                            tokenizer=tokenizer,
                            max_len=args.max_len,
                        )
                    except Exception as exc:
                        normalized = None
                        validation_errors = [f"{type(exc).__name__}: {exc}"]
                    if normalized is not None:
                        break
                if normalized is None:
                    continue
                if normalized.get("verdict") == "actually_correct" and scorer(
                    str(source.get("completion", "")), source.get("answer")
                ) != 1.0:
                    validation_errors = ["teacher_actually_correct_not_confirmed_by_local_verifier"]
                    continue
                selected = normalized
                break
            if selected is None and validation_errors == ["missing_scored_source"]:
                validation_errors = ["no_valid_teacher_payload"]
        status = "valid" if selected and selected.get("verdict") == "incorrect" else "quarantine"
        output = {
            "problem_id": pid,
            "validation_status": status,
            "attempts": saved.get("attempts", []),
            "teacher_model": saved.get("teacher_model"),
            "revalidated_from": [str(path) for path in input_paths],
        }
        if selected is not None:
            output["result"] = selected
        if status != "valid":
            output["validation_errors"] = validation_errors
        output_rows.append(output)

    output_path = Path(args.output)
    write_jsonl(output_path, output_rows)
    counts = Counter(str(row.get("validation_status")) for row in output_rows)
    report = {
        "inputs": [
            {"path": str(path), "sha256": sha256_file(path)} for path in input_paths
        ],
        "scored": str(args.scored),
        "scored_sha256": sha256_file(args.scored),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "rows": len(output_rows),
        "status_counts": dict(sorted(counts.items())),
        "max_len": args.max_len,
    }
    dump_json(output_path.with_suffix(output_path.suffix + ".report.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_teacher_staged(args: argparse.Namespace) -> None:
    """Fallback for hard rows: solve first, then locate the student's first error."""
    from openai import OpenAI
    from transformers import AutoTokenizer

    from rgrpo.math_score_light import compute_math_score

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is not set; refusing to call the paid teacher API.")
    scored = list(read_jsonl(Path(args.scored)))
    prior = {str(row["problem_id"]): row for row in read_jsonl(Path(args.teacher_input))}
    queue = [
        row
        for row in scored
        if row.get("score_status") == "incorrect"
        and str(row["problem_id"]) in prior
        and prior.get(str(row["problem_id"]), {}).get("validation_status") != "valid"
    ]
    if args.offset > 0:
        queue = queue[args.offset :]
    if args.limit > 0:
        queue = queue[: args.limit]
    output_path = Path(args.output)
    done = completed_ids(output_path) if args.resume else set()
    queue = [row for row in queue if row["problem_id"] not in done]
    print(f"[Teacher staged] pending quarantined trajectories: {len(queue)}")
    if args.dry_run:
        if queue:
            print(_staged_solution_prompt(queue[0]))
        return

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    client = OpenAI(api_key=api_key, base_url=args.base_url, timeout=args.timeout)

    def scorer(generated: str, answer: Any) -> float:
        return compute_math_score(generated=generated, ground_truth=answer)

    def call_stage(
        stage: str,
        base_prompt: str,
        validate: Any,
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[str]]:
        attempts: list[dict[str, Any]] = []
        previous_errors: list[str] = []
        previous_analysis = ""
        for attempt in range(1, args.max_attempts_per_stage + 1):
            continuation = ""
            if previous_analysis:
                continuation = f"""

The previous attempt exhausted its budget or failed validation. Continue from it; do not restart.
Validation errors: {', '.join(previous_errors)}
PRIOR_ATTEMPT:
{previous_analysis}
END_PRIOR_ATTEMPT
"""
            try:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": "You are a rigorous mathematical data annotator."},
                        {"role": "user", "content": base_prompt + continuation},
                    ],
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_completion_tokens=args.max_tokens,
                    extra_body={"reasoning_split": True},
                )
                choice = response.choices[0]
                message = choice.message
                content = message.content or ""
                reasoning = _reasoning_text(
                    getattr(message, "reasoning_content", None)
                    or getattr(message, "reasoning_details", None)
                )
                record = {
                    "stage": stage,
                    "attempt": attempt,
                    "response_id": response.id,
                    "finish_reason": choice.finish_reason,
                    "content": content,
                    "reasoning": reasoning,
                    "usage": response.usage.model_dump() if response.usage else {},
                }
                candidates = [("content", value) for value in extract_json_objects(content)]
                candidates.extend(
                    ("reasoning", value)
                    for value in reversed(extract_json_objects(reasoning))
                )
                normalized = None
                errors = ["teacher_response_does_not_contain_json"]
                for payload_source, parsed in candidates:
                    try:
                        candidate, candidate_errors = validate(parsed)
                    except Exception as exc:
                        candidate = None
                        candidate_errors = [f"{type(exc).__name__}: {exc}"]
                    if candidate is not None:
                        normalized = candidate
                        errors = []
                        record["validated_payload_source"] = payload_source
                        break
                    errors = candidate_errors
                record["validation_errors"] = errors
                attempts.append(record)
                if normalized is not None:
                    return normalized, attempts, []
                previous_errors = errors
                previous_analysis = reasoning
                if content:
                    previous_analysis += "\n\nPREVIOUS_CANDIDATE_JSON:\n" + content
            except Exception as exc:
                previous_errors = [f"{type(exc).__name__}: {exc}"]
                attempts.append(
                    {"stage": stage, "attempt": attempt, "exception": previous_errors[0]}
                )
            if attempt < args.max_attempts_per_stage:
                time.sleep(min(2**attempt, 8))
        return None, attempts, previous_errors

    def process(row: dict[str, Any]) -> dict[str, Any]:
        def validate_solution(payload: dict[str, Any]):
            reasoning = str(payload.get("reasoning", "")).strip()
            errors = []
            if not reasoning:
                errors.append("empty_reasoning")
            elif scorer(reasoning, row.get("answer")) != 1.0:
                errors.append("corrected_reasoning_wrong_answer")
            lengths = training_lengths(
                tokenizer,
                problem=str(row.get("problem", "")),
                full_reasoning=reasoning,
            )
            if any(length > args.max_len for length in lengths.values()):
                errors.append("training_sequence_too_long")
            normalized = {"reasoning": reasoning, "training_token_lengths": lengths}
            return (None if errors else normalized), errors

        solution, solve_attempts, solve_errors = call_stage(
            "solve", _staged_solution_prompt(row), validate_solution
        )
        if solution is None:
            return {
                "problem_id": row["problem_id"],
                "validation_status": "quarantine",
                "validation_errors": ["staged_solve_failed", *solve_errors],
                "attempts": solve_attempts,
                "teacher_model": args.model,
            }

        def validate_diagnosis(payload: dict[str, Any]):
            combined = {
                **payload,
                "verdict": "incorrect",
                "reasoning": solution["reasoning"],
            }
            return validate_teacher_payload(
                combined,
                row,
                scorer=scorer,
                tokenizer=tokenizer,
                max_len=args.max_len,
            )

        diagnosis, diagnosis_attempts, diagnosis_errors = call_stage(
            "diagnose",
            _staged_diagnosis_prompt(row, solution["reasoning"]),
            validate_diagnosis,
        )
        if diagnosis is None:
            return {
                "problem_id": row["problem_id"],
                "validation_status": "quarantine",
                "validation_errors": ["staged_diagnosis_failed", *diagnosis_errors],
                "attempts": [*solve_attempts, *diagnosis_attempts],
                "teacher_model": args.model,
            }
        return {
            "problem_id": row["problem_id"],
            "validation_status": "valid",
            "result": diagnosis,
            "attempts": [*solve_attempts, *diagnosis_attempts],
            "teacher_model": args.model,
            "teacher_strategy": "solve_then_diagnose",
        }

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(process, row) for row in queue]
        for future in as_completed(futures):
            append_jsonl(output_path, future.result())

    rows = list(read_jsonl(output_path))
    counts = Counter(str(row.get("validation_status")) for row in rows)
    usage = Counter()
    for row in rows:
        for attempt in row.get("attempts", []):
            for key, value in (attempt.get("usage") or {}).items():
                if isinstance(value, (int, float)):
                    usage[key] += value
    report = {
        "teacher_input": str(args.teacher_input),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "rows": len(rows),
        "status_counts": dict(sorted(counts.items())),
        "usage": dict(sorted(usage.items())),
        "max_tokens_per_stage": args.max_tokens,
    }
    dump_json(output_path.with_suffix(output_path.suffix + ".report.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _staged_solution_prompt(row: dict[str, Any]) -> str:
    return f"""Solve the problem independently. Do not analyze any student response.
The GOLD_ANSWER is authoritative. Return exactly one JSON object without Markdown:
{{"reasoning":"a concise self-contained derivation, at most 1400 tokens, whose final text is \\\\boxed{{{row['answer']}}}"}}
Do not question the gold answer and do not include any field other than reasoning.

PROBLEM:
{row['problem']}

GOLD_ANSWER:
{row['answer']}"""


def _staged_diagnosis_prompt(row: dict[str, Any], reference_solution: str) -> str:
    return f"""Locate the first mathematical error in STUDENT_SOLUTION using REFERENCE_SOLUTION.
Return exactly one JSON object without Markdown:
{{
  "verified_states":"correct intermediate conclusions before that error",
  "error_reason":"one concise semantic diagnosis of the first wrong reasoning direction or step",
  "new_direction":"one concise sentence giving the corrected direction"
}}
Do not copy a student-solution span merely to locate the error; describe the mathematical mistake
semantically. Do not solve the problem again.

PROBLEM:
{row['problem']}

STUDENT_SOLUTION:
{row['completion']}

REFERENCE_SOLUTION:
{reference_solution}"""


def command_compile(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    os.environ["CHAT_TEMPLATE_ENABLE_THINKING"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    scored = list(read_jsonl(args.scored))
    teacher_path = Path(args.teacher)
    teacher = list(read_jsonl(teacher_path)) if teacher_path.exists() else []
    reflective, perfect, quarantine = compile_sft_records(
        scored,
        teacher,
        tokenizer=tokenizer,
        max_len=args.max_len,
        seed=args.seed,
    )
    output_dir = Path(args.output_dir)
    reflection_path = output_dir / "reasoning_outputs.jsonl"
    perfect_path = output_dir / "reflection_data_prepared_perfect.jsonl"
    quarantine_path = output_dir / "quarantine.jsonl"
    write_jsonl(reflection_path, reflective)
    write_jsonl(perfect_path, perfect)
    write_jsonl(quarantine_path, quarantine)

    total = len(scored)
    usable = len(reflective) + len(perfect)
    teacher_needed = sum(row.get("score_status") == "incorrect" for row in scored)
    valid_teacher = sum(
        row.get("validation_status") == "valid" and row.get("result", {}).get("verdict") == "incorrect"
        for row in teacher
    )
    teacher_yield = 1.0 if teacher_needed == 0 else valid_teacher / teacher_needed
    quarantine_rate = 0.0 if total == 0 else len(quarantine) / total
    report = {
        "scored_rows": total,
        "reflective_rows": len(reflective),
        "perfect_rows": len(perfect),
        "usable_rows": usable,
        "quarantine_rows": len(quarantine),
        "quarantine_rate": quarantine_rate,
        "teacher_needed": teacher_needed,
        "valid_reflective_teacher_rows": valid_teacher,
        "teacher_yield": teacher_yield,
        "no_train_eval_overlap": not (
            {row["problem_id"] for row in reflective if row["split"] == "train"}
            & {row["problem_id"] for row in reflective if row["split"] == "eval"}
        )
        and not (
            {row["problem_id"] for row in perfect if row["split"] == "train"}
            & {row["problem_id"] for row in perfect if row["split"] == "eval"}
        ),
        "outputs": {
            "reflective": {"path": str(reflection_path), "sha256": sha256_file(reflection_path)},
            "perfect": {"path": str(perfect_path), "sha256": sha256_file(perfect_path)},
            "quarantine": {"path": str(quarantine_path), "sha256": sha256_file(quarantine_path)},
        },
    }
    dump_json(output_dir / "manifest.json", report)
    markdown = f"""# SFT Data Quality Report

- Scored trajectories: {total}
- Reflective: {len(reflective)}
- Perfect: {len(perfect)}
- Usable: {usable}
- Quarantine: {len(quarantine)} ({quarantine_rate:.2%})
- Teacher yield: {teacher_yield:.2%}
- Train/eval overlap: {not report['no_train_eval_overlap']}
- Chat template thinking: disabled
- Maximum training length: {args.max_len}
"""
    (output_dir / "quality_report.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not args.allow_incomplete:
        if teacher_yield < args.min_teacher_yield:
            raise RuntimeError(
                f"Teacher yield {teacher_yield:.2%} is below {args.min_teacher_yield:.2%}."
            )
        if quarantine_rate > args.max_quarantine_rate:
            raise RuntimeError(
                f"Quarantine rate {quarantine_rate:.2%} exceeds {args.max_quarantine_rate:.2%}."
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    select = sub.add_parser("select", help="Freeze the <=1/64 Big-Math question set.")
    select.add_argument("--dataset", default=DEFAULT_DATASET)
    select.add_argument("--revision", default=DEFAULT_REVISION)
    select.add_argument("--output-dir", default=str(DEFAULT_SOURCE_DIR))
    select.add_argument("--threshold", type=float, default=HARD_THRESHOLD)
    select.add_argument("--expected-eligible", type=int, default=9778)
    select.add_argument("--allow-insecure-tls", action="store_true")
    select.set_defaults(func=command_select)

    sample = sub.add_parser("sample", help="Sample no-thinking trajectories per problem.")
    sample.add_argument("--source", default=str(DEFAULT_SOURCE_DIR / "big_math_hard_le_1of64.jsonl"))
    sample.add_argument("--model", default=str(DEFAULT_MODEL))
    sample.add_argument("--output", default=str(DEFAULT_MODEL_DATA / "raw_rollouts.jsonl"))
    sample.add_argument("--engine", choices=("vllm", "transformers"), default="vllm")
    sample.add_argument("--pilot", action="store_true")
    sample.add_argument("--limit", type=int, default=0)
    sample.add_argument("--temperature", type=float, default=0.5)
    sample.add_argument("--top-p", type=float, default=1.0)
    sample.add_argument("--seed", type=int, default=42)
    sample.add_argument("--samples-per-problem", type=int, default=1)
    sample.add_argument(
        "--max-new-tokens", type=int, default=int(os.environ.get("SFT_SAMPLE_MAX_NEW_TOKENS", "4096"))
    )
    sample.add_argument(
        "--max-model-len", type=int, default=int(os.environ.get("SFT_SAMPLE_MAX_MODEL_LEN", "8192"))
    )
    sample.add_argument("--tensor-parallel-size", type=int, default=1)
    sample.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    sample.add_argument("--max-num-seqs", type=int, default=64)
    sample.add_argument("--batch-size", type=int, default=256)
    sample.add_argument("--num-shards", type=int, default=1)
    sample.add_argument("--shard-index", type=int, default=0)
    sample.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    sample.add_argument("--dry-run", action="store_true")
    sample.set_defaults(func=command_sample)

    merge_samples = sub.add_parser(
        "merge-samples", help="Validate and merge completed sampling shards."
    )
    merge_samples.add_argument(
        "--source", default=str(DEFAULT_SOURCE_DIR / "big_math_hard_le_1of64.jsonl")
    )
    merge_samples.add_argument("--input", action="append", required=True)
    merge_samples.add_argument("--output", required=True)
    merge_samples.add_argument("--samples-per-problem", type=int, default=1)
    merge_samples.add_argument("--overwrite", action="store_true")
    merge_samples.set_defaults(func=command_merge_samples)

    score = sub.add_parser("score", help="Score and split correct/incorrect trajectories.")
    score.add_argument("--input", default=str(DEFAULT_MODEL_DATA / "raw_rollouts.jsonl"))
    score.add_argument("--output", default=str(DEFAULT_MODEL_DATA / "scored_rollouts.jsonl"))
    score.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    score.set_defaults(func=command_score)

    teacher = sub.add_parser("teacher", help="Transform wrong trajectories with MiniMax.")
    teacher.add_argument("--input", default=str(DEFAULT_MODEL_DATA / "scored_rollouts.jsonl"))
    teacher.add_argument("--output", default=str(DEFAULT_MODEL_DATA / "teacher_raw.jsonl"))
    teacher.add_argument("--tokenizer", default=str(DEFAULT_MODEL))
    teacher.add_argument(
        "--base-url", default=os.environ.get("MINIMAX_BASE_URL", "https://llmapi.paratera.com")
    )
    teacher.add_argument("--model", default=os.environ.get("MINIMAX_MODEL", "MiniMax-M2.7"))
    teacher.add_argument("--api-key-env", default="MINIMAX_API_KEY")
    teacher.add_argument("--temperature", type=float, default=1.0)
    teacher.add_argument("--top-p", type=float, default=0.95)
    teacher.add_argument(
        "--max-tokens", type=int, default=int(os.environ.get("MINIMAX_MAX_TOKENS", "8192"))
    )
    teacher.add_argument("--max-len", type=int, default=2048)
    teacher.add_argument("--max-attempts", type=int, default=3)
    teacher.add_argument("--timeout", type=float, default=180.0)
    teacher.add_argument("--concurrency", type=int, default=16)
    teacher.add_argument("--offset", type=int, default=0)
    teacher.add_argument(
        "--bootstrap-teacher",
        default="",
        help="Optional prior teacher JSONL whose saved reasoning seeds continuation retries.",
    )
    teacher.add_argument("--limit", type=int, default=0)
    teacher.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    teacher.add_argument("--dry-run", action="store_true")
    teacher.set_defaults(func=command_teacher)

    revalidate = sub.add_parser(
        "revalidate-teacher", help="Re-run local teacher gates without paid API calls."
    )
    revalidate.add_argument("--input", default=str(DEFAULT_MODEL_DATA / "teacher_raw.jsonl"))
    revalidate.add_argument("--extra-input", action="append", default=[])
    revalidate.add_argument("--scored", default=str(DEFAULT_MODEL_DATA / "scored_rollouts.jsonl"))
    revalidate.add_argument(
        "--output", default=str(DEFAULT_MODEL_DATA / "teacher_validated.jsonl")
    )
    revalidate.add_argument("--tokenizer", default=str(DEFAULT_MODEL))
    revalidate.add_argument("--max-len", type=int, default=2048)
    revalidate.set_defaults(func=command_revalidate_teacher)

    staged = sub.add_parser(
        "teacher-staged", help="Retry quarantined rows with solve-then-diagnose calls."
    )
    staged.add_argument("--scored", default=str(DEFAULT_MODEL_DATA / "scored_rollouts.jsonl"))
    staged.add_argument(
        "--teacher-input", default=str(DEFAULT_MODEL_DATA / "teacher_validated.jsonl")
    )
    staged.add_argument(
        "--output", default=str(DEFAULT_MODEL_DATA / "teacher_staged_fallback.jsonl")
    )
    staged.add_argument("--tokenizer", default=str(DEFAULT_MODEL))
    staged.add_argument(
        "--base-url", default=os.environ.get("MINIMAX_BASE_URL", "https://llmapi.paratera.com")
    )
    staged.add_argument("--model", default=os.environ.get("MINIMAX_MODEL", "MiniMax-M2.7"))
    staged.add_argument("--api-key-env", default="MINIMAX_API_KEY")
    staged.add_argument("--temperature", type=float, default=1.0)
    staged.add_argument("--top-p", type=float, default=0.95)
    staged.add_argument(
        "--max-tokens", type=int, default=int(os.environ.get("MINIMAX_MAX_TOKENS", "8192"))
    )
    staged.add_argument("--max-len", type=int, default=2048)
    staged.add_argument("--max-attempts-per-stage", type=int, default=2)
    staged.add_argument("--timeout", type=float, default=600.0)
    staged.add_argument("--concurrency", type=int, default=4)
    staged.add_argument("--offset", type=int, default=0)
    staged.add_argument("--limit", type=int, default=0)
    staged.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    staged.add_argument("--dry-run", action="store_true")
    staged.set_defaults(func=command_teacher_staged)

    compile_cmd = sub.add_parser("compile", help="Compile validated Phase-1/2 SFT files.")
    compile_cmd.add_argument("--scored", default=str(DEFAULT_MODEL_DATA / "scored_rollouts.jsonl"))
    compile_cmd.add_argument("--teacher", default=str(DEFAULT_MODEL_DATA / "teacher_raw.jsonl"))
    compile_cmd.add_argument("--tokenizer", default=str(DEFAULT_MODEL))
    compile_cmd.add_argument("--output-dir", default=str(DEFAULT_MODEL_DATA))
    compile_cmd.add_argument("--max-len", type=int, default=2048)
    compile_cmd.add_argument("--seed", type=int, default=42)
    compile_cmd.add_argument("--min-teacher-yield", type=float, default=0.99)
    compile_cmd.add_argument("--max-quarantine-rate", type=float, default=0.01)
    compile_cmd.add_argument("--allow-incomplete", action="store_true")
    compile_cmd.set_defaults(func=command_compile)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
