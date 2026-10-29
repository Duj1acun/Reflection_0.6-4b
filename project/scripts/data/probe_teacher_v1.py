#!/usr/bin/env python3
"""A bounded four-record teacher probe; never sends perfect rows to the API."""

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
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PROMPTS = ROOT / "configs/sft/teacher_v1"
DATA = Path("/mnt/data/reflection_grpo/data/sft")
ASSETS = (
    "diagnosis.system.txt", "diagnosis.schema.json",
    "continuation.system.txt", "continuation.schema.json",
)
STRATA = {"qwen3-0.6b": (1, 2), "qwen3-4b": (2, 3)}
MODEL_DIRS = {"qwen3-0.6b": "Qwen3-0.6B", "qwen3-4b": "Qwen3-4B"}
TAGS = re.compile(r"<\s*/?\s*(?:think|reflection)\b", re.I)


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate_json_key")
            result[key] = value
        return result

    def nonfinite(_):
        raise ValueError("nonfinite_json_number")

    result = json.loads(text, object_pairs_hook=pairs, parse_constant=nonfinite)
    if not isinstance(result, dict):
        raise ValueError("top_level_must_be_object")
    return result


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def segments(text):
    spans, start = [], 0
    for gap in re.finditer(r"\r?\n[ \t]*(?:\r?\n)+", text):
        if text[start:gap.start()].strip():
            spans.append((start, gap.start()))
        start = gap.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    return [
        {"step_id": f"S{index:04d}", "text": text[a:b], "start": a, "end": b}
        for index, (a, b) in enumerate(spans, 1)
    ]


def rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def select_probe(seed):
    selected, sources = [], {}
    for model, buckets in STRATA.items():
        path = DATA / model / "scored_rollouts_n4.jsonl"
        source_hash = digest(path)
        groups = defaultdict(list)
        for row in rows(path):
            groups[row["problem_id"]].append(
                (row["score_status"], row["sample_index"], row["trajectory_id"])
            )
        eligible = {}
        for pid, items in groups.items():
            if (
                len(items) == 4
                and {r[1] for r in items} == {0, 1, 2, 3}
                and len({r[2] for r in items}) == 4
                and all(r[0] in {"correct", "incorrect"} for r in items)
            ):
                k = sum(r[0] == "correct" for r in items)
                if k in buckets:
                    eligible[pid] = k
        best = {}
        for row in rows(path):
            pid = row["problem_id"]
            if (
                pid not in eligible or row["score_status"] != "incorrect"
                or row.get("finish_reason") != "stop"
            ):
                continue
            k = eligible[pid]
            rank = (
                hashlib.sha256(f"{seed}:{model}:{pid}".encode()).hexdigest(),
                hashlib.sha256(f"{seed}:{row['trajectory_id']}".encode()).hexdigest(),
            )
            if k not in best or rank < best[k][0]:
                best[k] = (rank, row)
        if set(best) != set(buckets):
            raise ValueError(f"missing_probe_stratum:{model}")
        if digest(path) != source_hash:
            raise ValueError(f"source_changed_during_selection:{model}")
        sources[model] = {"path": str(path), "sha256": source_hash}
        for k in buckets:
            row = best[k][1]
            selected.append({
                "probe_id": f"{model}_k{k}",
                "student_model": model,
                "problem_id": row["problem_id"],
                "trajectory_id": row["trajectory_id"],
                "sample_index": row["sample_index"],
                "correct_out_of_four": k,
                "problem": row["problem"],
                "answer": row["answer"],
                "completion": row["completion"],
                "predicted_answer": row.get("predicted_answer"),
                "score_status": row["score_status"],
                "finish_reason": row["finish_reason"],
                "student_steps": segments(row["completion"]),
            })
    return selected, sources


def make_request(stage, row, manifest, reflection=None):
    system = manifest["assets"][f"{stage}.system.txt"]["text"]
    system += "\n\nOUTPUT_SCHEMA:\n" + manifest["assets"][f"{stage}.schema.json"]["text"]
    if stage == "diagnosis":
        user = {
            "problem": row["problem"],
            "reference_answer": row["answer"],
            "student_steps": [
                {"step_id": s["step_id"], "text": s["text"]}
                for s in row["student_steps"]
            ],
        }
    elif stage == "continuation":
        if not reflection:
            raise ValueError("missing_reflection")
        user = {"problem": row["problem"], "reflection_text": reflection}
    else:
        raise ValueError("unknown_stage")
    cfg = manifest["request"]
    body = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        "temperature": cfg["temperature"],
        "top_p": cfg["top_p"],
        "max_completion_tokens": cfg["max_completion_tokens"],
        "reasoning_split": True,
    }
    if cfg["model"] == "MiniMax-M3":
        body["thinking"] = {"type": "adaptive"}
    return body


def validate(stage, payload, row, manifest):
    from jsonschema import Draft202012Validator
    schema = json.loads(manifest["assets"][f"{stage}.schema.json"]["text"])
    Draft202012Validator(schema).validate(payload)
    if stage == "diagnosis" and payload["verdict"] == "incorrect":
        lookup = {s["step_id"]: s["text"] for s in row["student_steps"]}
        source = lookup.get(payload["first_error_step"], "")
        if payload["first_error_quote"] not in source:
            raise ValueError("error_quote_not_in_specified_segment")
        fields = "\n".join(payload[f] for f in ("verified_states", "error_reason", "new_direction"))
        if TAGS.search(fields) or "\\boxed" in fields:
            raise ValueError("reflection_fields_contain_forbidden_markup")
    if stage == "continuation" and payload["status"] == "ok":
        if TAGS.search(payload["reasoning"]):
            raise ValueError("continuation_contains_reflection_or_think_tags")
        if "\\boxed{" not in payload["reasoning"]:
            raise ValueError("missing_boxed_answer")


def render(payload):
    return (
        "<reflection>\n"
        f"So far I have: {payload['verified_states'].strip()}\n"
        f"However, {payload['error_reason'].strip()}\n"
        f"Next, {payload['new_direction'].strip()}\n"
        "</reflection>"
    )


def prepare(args):
    if args.output.exists():
        raise ValueError("output_exists_refusing_overwrite")
    selected, sources = select_probe(args.seed)
    assets = {
        name: {"text": (PROMPTS / name).read_text(), "sha256": digest(PROMPTS / name)}
        for name in ASSETS
    }
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Four-record API probe, not a full annotation run.",
        "seed": args.seed,
        "selection": "Hash-ranked question and trajectory within each declared stratum.",
        "sources": sources,
        "assets": assets,
        "request": {
            "endpoint": "https://llmapi.paratera.com/v1/chat/completions",
            "model": "MiniMax-M2.7", "temperature": 0.3, "top_p": 0.95,
            "max_completion_tokens": 16384, "timeout_seconds": 600,
            "max_requests": 8, "concurrency": 4, "automatic_retries": 0,
        },
    }
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "samples.json", selected)
    manifest["samples_sha256"] = digest(args.output / "samples.json")
    save(args.output / "manifest.json", manifest)
    for row in selected:
        print(json.dumps({
            "probe_id": row["probe_id"], "trajectory_id": row["trajectory_id"],
            "correct_out_of_four": row["correct_out_of_four"],
            "problem": row["problem"], "student_answer": row["predicted_answer"],
            "reference_answer": row["answer"],
        }, ensure_ascii=False), flush=True)
    print(f"Prepared {len(selected)} wrong responses. API requests: 0.", flush=True)


def prepare_comparison(args):
    """Freeze the same inputs and prompts, changing only the teacher model."""
    if not args.reference_probe or not args.model:
        raise ValueError("comparison_requires_reference_probe_and_model")
    if args.output.exists():
        raise ValueError("output_exists_refusing_overwrite")
    baseline = args.reference_probe.resolve()
    manifest = json.loads((baseline / "manifest.json").read_text())
    if digest(baseline / "samples.json") != manifest["samples_sha256"]:
        raise ValueError("baseline_samples_hash_mismatch")
    for asset in manifest["assets"].values():
        if hashlib.sha256(asset["text"].encode()).hexdigest() != asset["sha256"]:
            raise ValueError("baseline_prompt_hash_mismatch")
    selected = json.loads((baseline / "samples.json").read_text())
    request = json.loads((baseline / "run_config.json").read_text())["effective_request"]
    previous_model = request["model"]
    request["model"] = args.model
    manifest.update({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Four-record matched teacher comparison; no full annotation run.",
        "request": request,
        "comparison": {
            "baseline_directory": str(baseline),
            "baseline_model": previous_model,
            "changed_request_parameters": ["model"],
            "thinking_note": "M3 uses thinking.type=adaptive; both models request reasoning_split=true.",
            "identical_samples_and_prompt_assets": True,
        },
    })
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "samples.json", selected)
    if digest(args.output / "samples.json") != manifest["samples_sha256"]:
        raise ValueError("comparison_sample_serialization_changed")
    save(args.output / "manifest.json", manifest)
    print(json.dumps({
        "prepared_records": len(selected), "model": args.model,
        "baseline": str(baseline), "api_requests": 0,
        "same_samples_sha256": manifest["samples_sha256"],
    }), flush=True)


async def call_stage(client, gate, stage, row, manifest, output, key, counter, reflection=None):
    body = make_request(stage, row, manifest, reflection)
    prefix = output / "calls" / f"{row['probe_id']}.{stage}"
    save(str(prefix) + ".request.json", body)
    started = time.monotonic()
    result = {"stage": stage, "probe_id": row["probe_id"]}
    async with gate:
        if counter["requests"] >= manifest["request"]["max_requests"]:
            raise ValueError("request_budget_exhausted")
        counter["requests"] += 1
        print(f"REQUEST {counter['requests']}/8 {row['probe_id']} {stage}", flush=True)
        try:
            response = await client.post(
                manifest["request"]["endpoint"],
                json=body,
                headers={"Authorization": f"Bearer {key}"},
            )
            result["http_status"] = response.status_code
            safe_text = response.text.replace(key, "[REDACTED]")
            try:
                raw = strict_json(safe_text)
            except (ValueError, TypeError):
                result["error"] = "response_not_json_object"
                save(str(prefix) + ".result.json", result)
                return result
            save(str(prefix) + ".response.json", raw)
            if response.status_code != 200:
                result["error"] = f"http_{response.status_code}"
            else:
                choice = raw["choices"][0]
                message = choice["message"]
                thought = message.get("reasoning_content") or message.get("reasoning_details")
                result.update({
                    "response_model": raw.get("model"),
                    "finish_reason": choice.get("finish_reason"),
                    "usage": raw.get("usage", {}),
                    "thinking_returned": bool(thought),
                    "thinking_characters": (
                        len(thought) if isinstance(thought, str)
                        else len(json.dumps(thought, ensure_ascii=False)) if thought else 0
                    ),
                })
                if choice.get("finish_reason") != "stop":
                    result["error"] = "response_not_normally_finished"
                else:
                    payload = strict_json(message.get("content") or "")
                    validate(stage, payload, row, manifest)
                    result["payload"] = payload
        except Exception as exc:
            # Exception messages may include request information. Keep the type only.
            result["error"] = type(exc).__name__
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    save(str(prefix) + ".result.json", result)
    print(json.dumps({
        "finished": row["probe_id"], "stage": stage,
        "error": result.get("error"),
        "verdict": result.get("payload", {}).get("verdict"),
        "status": result.get("payload", {}).get("status"),
        "thinking_returned": result.get("thinking_returned"),
    }), flush=True)
    return result


async def run_requests(selected, manifest, output, key):
    import httpx
    gate = asyncio.Semaphore(manifest["request"]["concurrency"])
    counter = {"requests": 0}
    timeout = httpx.Timeout(manifest["request"]["timeout_seconds"], connect=30)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
        async def one(row):
            result = {"probe_id": row["probe_id"], "student_model": row["student_model"]}
            first = await call_stage(client, gate, "diagnosis", row, manifest, output, key, counter)
            result["diagnosis"] = first
            payload = first.get("payload", {})
            if payload.get("verdict") == "incorrect":
                reflection = render(payload)
                result["reflection_text"] = reflection
                result["continuation"] = await call_stage(
                    client, gate, "continuation", row, manifest, output, key, counter, reflection
                )
            return result
        # Establish that the requested service accepts the first real sample
        # before spending requests on the remaining records.
        first = await one(selected[0])
        first_http_status = first["diagnosis"].get("http_status")
        results = [first]
        if first_http_status == 200 and first["diagnosis"].get("error") != "KeyError":
            results.extend(await asyncio.gather(*(one(row) for row in selected[1:])))
    return results, counter["requests"]


def run(args):
    key = os.environ.get("MINIMAX_API_KEY")
    if args.prompt_api_key:
        if not sys.stdin.isatty():
            raise ValueError("secure_key_prompt_requires_tty")
        key = getpass.getpass("MiniMax API key (input hidden, not saved): ").strip()
    if not key:
        raise ValueError("MINIMAX_API_KEY_missing_no_requests_sent")
    manifest = json.loads((args.output / "manifest.json").read_text())
    prepared_endpoint = manifest["request"]["endpoint"]
    if args.base_url:
        url = urlsplit(args.base_url)
        if url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("base_url_must_be_plain_https_without_credentials")
        manifest["request"]["endpoint"] = args.base_url.rstrip("/") + "/chat/completions"
    if digest(args.output / "samples.json") != manifest["samples_sha256"]:
        raise ValueError("samples_changed_after_prepare")
    for asset in manifest["assets"].values():
        if hashlib.sha256(asset["text"].encode()).hexdigest() != asset["sha256"]:
            raise ValueError("prompt_snapshot_hash_mismatch")
    selected = json.loads((args.output / "samples.json").read_text())
    if len(selected) != 4 or any(
        r["score_status"] != "incorrect" or r["finish_reason"] != "stop" for r in selected
    ):
        raise ValueError("probe_must_contain_exactly_four_complete_wrong_responses")
    if manifest["request"]["max_requests"] != 8:
        raise ValueError("unexpected_request_budget")
    # Refuse duplicate paid runs; preparing a new directory is an explicit new trial.
    (args.output / "calls").mkdir(exist_ok=False)
    save(args.output / "run_config.json", {
        "prepared_endpoint": prepared_endpoint,
        "effective_request": manifest["request"],
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "runner_sha256": digest(Path(__file__)),
    })
    results, requests = asyncio.run(run_requests(selected, manifest, args.output, key))
    save(args.output / "results.json", results)
    summary = {
        "actual_requests": requests,
        "records": len(results),
        "diagnosis_incorrect": sum(
            r["diagnosis"].get("payload", {}).get("verdict") == "incorrect" for r in results
        ),
        "continuation_ok": sum(
            r.get("continuation", {}).get("payload", {}).get("status") == "ok" for r in results
        ),
        "semantic_review": "Pending; API status is not evidence of mathematical correctness.",
        "perfect_api_requests": 0,
    }
    save(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "prepare-comparison", "run"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference-probe", type=Path, help="Completed baseline directory for a matched comparison.")
    parser.add_argument("--model", choices=["MiniMax-M2.7", "MiniMax-M3"], help="Teacher model for prepare-comparison.")
    parser.add_argument("--base-url", help="Explicit service override, e.g. https://api.minimaxi.com/v1")
    parser.add_argument("--prompt-api-key", action="store_true", help="Read the key from a hidden TTY prompt, never from command arguments or files.")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args)
    elif args.action == "prepare-comparison":
        prepare_comparison(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
