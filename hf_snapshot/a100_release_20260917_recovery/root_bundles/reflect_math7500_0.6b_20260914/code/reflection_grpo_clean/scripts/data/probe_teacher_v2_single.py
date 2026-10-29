#!/usr/bin/env python3
"""Bounded 4- or 20-record single-call M3 probes; no retries or training export."""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

from probe_teacher_v1 import DATA, digest, render, rows, save, segments, strict_json

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ASSET_ROOT = ROOT / "configs/sft/teacher_v2_single"
ASSETS = ("annotation.system.txt", "annotation.schema.json")
TRAINING_FIELDS = ("verified_states", "error_reason", "new_direction", "reasoning", "final_answer")
TAGS = re.compile(r"<\s*/?\s*(?:think|reflection)\b", re.I)
FENCE = re.compile(r"\A```(?:json)?[ \t]*\r?\n(?P<body>.*?)\r?\n```\Z", re.S)


def select_fresh_twenty(previous, seed):
    """Select before observing teacher outputs, one trajectory per question."""
    excluded = {r["problem_id"] for r in previous}
    used = set(excluded)
    selected, sources = [], {}
    quotas = {1: 4, 2: 4, 3: 2}
    for model in ("qwen3-0.6b", "qwen3-4b"):
        path = DATA / model / "scored_rollouts_n4.jsonl"
        source_hash = digest(path)
        groups = defaultdict(list)
        for row in rows(path):
            groups[row["problem_id"]].append((row["score_status"], row["sample_index"], row["trajectory_id"]))
        eligible = {}
        for pid, items in groups.items():
            if (pid not in used and len(items) == 4
                    and {x[1] for x in items} == {0, 1, 2, 3}
                    and len({x[2] for x in items}) == 4
                    and all(x[0] in ("correct", "incorrect") for x in items)):
                k = sum(x[0] == "correct" for x in items)
                if k in quotas:
                    eligible[pid] = k
        best = {}
        for row in rows(path):
            pid = row["problem_id"]
            if (pid not in eligible or row["score_status"] != "incorrect"
                    or row.get("finish_reason") != "stop" or not row.get("completion", "").strip()):
                continue
            rank = hashlib.sha256(f"{seed}:{model}:{row['trajectory_id']}".encode()).hexdigest()
            if pid not in best or rank < best[pid][0]:
                best[pid] = (rank, row)
        for k, quota in quotas.items():
            candidates = sorted(
                (pid for pid in best if eligible[pid] == k and pid not in used),
                key=lambda pid: hashlib.sha256(f"{seed}:{model}:{pid}".encode()).hexdigest(),
            )
            if len(candidates) < quota:
                raise ValueError("not_enough_fresh_questions")
            for index, pid in enumerate(candidates[:quota], 1):
                row = best[pid][1]
                used.add(pid)
                selected.append({
                    "probe_id": f"{model}_k{k}_{index:02d}", "student_model": model,
                    "problem_id": pid, "trajectory_id": row["trajectory_id"],
                    "sample_index": row["sample_index"], "correct_out_of_four": k,
                    "problem": row["problem"], "answer": row["answer"],
                    "completion": row["completion"], "predicted_answer": row.get("predicted_answer"),
                    "score_status": row["score_status"], "finish_reason": row["finish_reason"],
                    "student_steps": segments(row["completion"]),
                })
        if digest(path) != source_hash:
            raise ValueError("source_changed_during_selection")
        sources[model] = {"path": str(path), "sha256": source_hash}
    if len({r["problem_id"] for r in selected}) != 20:
        raise ValueError("fresh_probe_questions_must_be_unique")
    return selected, sources, {
        "seed": seed, "per_model_strata": quotas,
        "excluded_previous_problem_ids": sorted(excluded),
        "method": "Hash-ranked complete wrong trajectory per hash-ranked question, stratified before API calls.",
        "interpretation": "Balanced exploratory probe, not a frequency-weighted population estimate.",
    }


def prepare(args):
    if args.output.exists():
        raise ValueError("output_exists_refusing_overwrite")
    if not args.reference_probe:
        raise ValueError("reference_probe_required")
    baseline = args.reference_probe.resolve()
    prior = json.loads((baseline / "manifest.json").read_text())
    if digest(baseline / "samples.json") != prior["samples_sha256"]:
        raise ValueError("baseline_sample_hash_mismatch")
    selected = json.loads((baseline / "samples.json").read_text())
    fresh = args.action == "prepare-fresh"
    if fresh:
        if args.num_samples != 20:
            raise ValueError("fresh_probe_requires_explicit_num_samples_20")
        selected, sources, selection = select_fresh_twenty(selected, args.seed)
    else:
        sources, selection = prior["sources"], {"method": "Identical frozen baseline samples."}
    validate_samples(selected, args.num_samples)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": f"Bounded {args.num_samples}-record single-call prompt trial, not a full annotation run.",
        "baseline_directory": str(baseline),
        "sources": sources,
        "selection": selection,
        "assets": {
            name: {"text": (ASSET_ROOT / name).read_text(), "sha256": digest(ASSET_ROOT / name)}
            for name in ASSETS
        },
        "request": {
            "endpoint": "https://api.minimaxi.com/v1/chat/completions",
            "model": "MiniMax-M3", "temperature": 0.3, "top_p": 0.95,
            "max_completion_tokens": 16384, "timeout_seconds": 600,
            "max_requests": args.num_samples, "concurrency": 4, "automatic_retries": 0,
            "thinking": {"type": "adaptive"}, "reasoning_split": True,
        },
        "output_policy": {
            "schema": "Local validation only; not appended to the system prompt.",
            "fence": "May unwrap one complete JSON code fence; log this normalization, never repair JSON content.",
            "boxed": "Program appends boxed(final_answer) after reasoning; manual consistency review required.",
            "semantic_review": "Required before any training export; one call does not isolate continuation context.",
        },
    }
    if fresh and manifest["assets"] != prior["assets"]:
        raise ValueError("fresh_trial_must_use_unchanged_baseline_prompts")
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "samples.json", selected)
    manifest["samples_sha256"] = digest(args.output / "samples.json")
    if not fresh and manifest["samples_sha256"] != prior["samples_sha256"]:
        raise ValueError("comparison_sample_serialization_changed")
    save(args.output / "manifest.json", manifest)
    print(f"Prepared {len(selected)} {'new' if fresh else 'matched'} records; API requests: 0.", flush=True)


def validate_samples(selected, count=4):
    if count not in (4, 20) or len(selected) != count or len({r["trajectory_id"] for r in selected}) != count:
        raise ValueError("unexpected_number_of_unique_samples")
    if len({r["probe_id"] for r in selected}) != count:
        raise ValueError("duplicate_probe_id")
    if any(r["score_status"] != "incorrect" or r["finish_reason"] != "stop" for r in selected):
        raise ValueError("only_complete_wrong_candidates_allowed")


def make_request(row, manifest):
    cfg = manifest["request"]
    return {
        key: cfg[key] for key in (
            "model", "temperature", "top_p", "max_completion_tokens", "thinking", "reasoning_split"
        )
    } | {"messages": [
        {"role": "system", "content": manifest["assets"]["annotation.system.txt"]["text"]},
        {"role": "user", "content": json.dumps({
            "problem": row["problem"], "student_response": row["completion"],
            "reference_answer": row["answer"],
        }, ensure_ascii=False)},
    ]}


def parse_content(content):
    match = FENCE.fullmatch(content.strip())
    if match:
        return strict_json(match.group("body")), "removed_single_json_fence"
    return strict_json(content), None


def validate(payload, manifest):
    from jsonschema import Draft202012Validator
    schema = json.loads(manifest["assets"]["annotation.schema.json"]["text"])
    Draft202012Validator(schema).validate(payload)
    fields = TRAINING_FIELDS if payload["verdict"] == "incorrect" else ("review_note",)
    if any(not payload[key].strip() for key in fields):
        raise ValueError("whitespace_only_required_field")
    if payload["verdict"] == "incorrect":
        if any(TAGS.search(payload[key]) for key in TRAINING_FIELDS):
            raise ValueError("unexpected_thinking_or_reflection_markup")
        if any("\\boxed" in payload[key] for key in TRAINING_FIELDS):
            raise ValueError("boxed_format_is_program_owned")
        answer = payload["final_answer"].strip()
        if "\n" in answer or "\r" in answer or "$" in answer:
            raise ValueError("final_answer_must_be_bare_single_line")


def corrected_reasoning(payload):
    return payload["reasoning"].strip() + "\n\n\\boxed{" + payload["final_answer"].strip() + "}"


async def run_requests(selected, manifest, output, key):
    import httpx
    cfg = manifest["request"]
    gate = asyncio.Semaphore(cfg["concurrency"])
    count = 0
    timeout = httpx.Timeout(cfg["timeout_seconds"], connect=30)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
        async def one(row):
            nonlocal count
            result = {"probe_id": row["probe_id"], "trajectory_id": row["trajectory_id"]}
            prefix = output / "calls" / row["probe_id"]
            body = make_request(row, manifest)
            save(str(prefix) + ".request.json", body)
            async with gate:
                if count >= cfg["max_requests"]:
                    raise ValueError("request_budget_exhausted")
                count += 1
                print(f"REQUEST {count}/{cfg['max_requests']} {row['probe_id']} single-call", flush=True)
                started = time.monotonic()
                try:
                    response = await client.post(cfg["endpoint"], json=body, headers={"Authorization": f"Bearer {key}"})
                    result["http_status"] = response.status_code
                    raw = strict_json(response.text.replace(key, "[REDACTED]"))
                    save(str(prefix) + ".response.json", raw)
                    if response.status_code != 200:
                        result["error"] = f"http_{response.status_code}"
                    elif not raw.get("choices"):
                        result["error"] = "missing_choices"
                    else:
                        choice = raw["choices"][0]
                        message = choice["message"]
                        thought = message.get("reasoning_content") or message.get("reasoning_details")
                        result.update({
                            "response_model": raw.get("model"), "finish_reason": choice.get("finish_reason"),
                            "usage": raw.get("usage", {}), "thinking_returned": bool(thought),
                        })
                        if raw.get("model") != cfg["model"]:
                            result["error"] = "unexpected_response_model"
                        elif choice.get("finish_reason") != "stop":
                            result["error"] = "not_normally_finished"
                        else:
                            payload, normalized = parse_content(message.get("content") or "")
                            result["format_normalization"] = normalized
                            validate(payload, manifest)
                            result["payload"] = payload
                            if payload["verdict"] == "incorrect":
                                result["reflection_text"] = render(payload)
                                result["corrected_reasoning"] = corrected_reasoning(payload)
                except Exception as exc:
                    result["error"] = type(exc).__name__
                result["elapsed_seconds"] = round(time.monotonic() - started, 3)
            save(str(prefix) + ".result.json", result)
            print(json.dumps({"finished": row["probe_id"], "error": result.get("error"),
                              "verdict": result.get("payload", {}).get("verdict"),
                              "thinking": result.get("thinking_returned")}), flush=True)
            return result

        first = await one(selected[0])
        results = [first]
        if first.get("http_status") == 200 and first.get("response_model") == cfg["model"]:
            results.extend(await asyncio.gather(*(one(row) for row in selected[1:])))
    return results, count


def run(args):
    manifest = json.loads((args.output / "manifest.json").read_text())
    if digest(args.output / "samples.json") != manifest["samples_sha256"]:
        raise ValueError("samples_changed")
    for asset in manifest["assets"].values():
        if hashlib.sha256(asset["text"].encode()).hexdigest() != asset["sha256"]:
            raise ValueError("prompt_snapshot_changed")
    cfg = manifest["request"]
    if cfg["max_requests"] != args.num_samples or args.num_samples not in (4, 20) or cfg["automatic_retries"] != 0:
        raise ValueError("unexpected_request_budget")
    if cfg["endpoint"] != "https://api.minimaxi.com/v1/chat/completions":
        raise ValueError("unexpected_endpoint")
    selected = json.loads((args.output / "samples.json").read_text())
    validate_samples(selected, args.num_samples)
    if (args.output / "calls").exists():
        raise ValueError("calls_exist_refusing_duplicate_paid_run")
    key = os.environ.get("MINIMAX_API_KEY")
    if args.prompt_api_key:
        if not sys.stdin.isatty():
            raise ValueError("hidden_key_prompt_requires_tty")
        key = getpass.getpass("MiniMax API key (hidden, not saved): ").strip()
    if not key:
        raise ValueError("api_key_missing_no_requests_sent")
    (args.output / "calls").mkdir(exist_ok=False)
    save(args.output / "run_config.json", {
        "effective_request": cfg, "started_utc": datetime.now(timezone.utc).isoformat(),
        "runner_sha256": digest(Path(__file__)),
        "shared_helpers_sha256": digest(Path(__file__).with_name("probe_teacher_v1.py")),
    })
    results, count = asyncio.run(run_requests(selected, manifest, args.output, key))
    save(args.output / "results.json", results)
    summary = {
        "actual_requests": count, "records": len(results), "perfect_api_requests": 0,
        "incorrect_objects_passing_structure": sum(r.get("payload", {}).get("verdict") == "incorrect" for r in results),
        "review_only_objects": sum(r.get("payload", {}).get("verdict") not in (None, "incorrect") for r in results),
        "semantic_review": "Pending; no training data exported.",
    }
    save(args.output / "summary.json", summary)
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "prepare-fresh", "run"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-probe", type=Path)
    parser.add_argument("--num-samples", type=int, choices=[4, 20], default=4, help="Explicit per-run paid request cap; defaults to the original four-record probe.")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--prompt-api-key", action="store_true")
    args = parser.parse_args()
    run(args) if args.action == "run" else prepare(args)


if __name__ == "__main__":
    main()
