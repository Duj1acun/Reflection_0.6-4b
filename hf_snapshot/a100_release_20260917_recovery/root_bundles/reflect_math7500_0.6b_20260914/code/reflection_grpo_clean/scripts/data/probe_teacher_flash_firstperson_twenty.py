#!/usr/bin/env python3
"""Twenty bounded Paratera DeepSeek requests using the revised teacher prompt with thinking enabled at temperature 0.1."""
from __future__ import annotations

import argparse
import ast
import asyncio
from collections import Counter
from datetime import datetime, timezone
import getpass
import inspect
import json
import os
from pathlib import Path
import sys
import time

from prepare_sft_data import _teacher_prompt
from probe_teacher_v1 import digest, render, save
from probe_teacher_v2_single import parse_content


def validate_samples(samples, count):
    assert len(samples) == count == 20
    assert len({r["trajectory_id"] for r in samples}) == count
    assert len({r["probe_id"] for r in samples}) == count
    assert all(r["score_status"] == "incorrect" and r["finish_reason"] == "stop" for r in samples)


LEGACY = Path(__file__).with_name("prepare_sft_data.py")
ENDPOINT = "https://llmapi.paratera.com/v1/chat/completions"
SYSTEM = "You are a rigorous mathematical data annotator."
FIELDS = ("verified_states", "error_reason", "new_direction", "reasoning")
CONFIG = {
    "endpoint": ENDPOINT, "model": "DeepSeek-V4-Flash", "temperature": 0.1,
    "top_p": 0.95, "max_tokens": 16384,
    "thinking": {"type": "enabled"}, "reasoning_effort": "high",
    "max_requests": 20, "concurrency": 20, "automatic_retries": 0,
    "timeout_seconds": 600,
}


def request_body(row):
    return {k: CONFIG[k] for k in (
        "model", "temperature", "top_p", "max_tokens",
        "thinking", "reasoning_effort",
    )} | {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": _teacher_prompt(row)},
    ]}


def validate(payload):
    if set(payload) != {"verdict", *FIELDS}:
        raise ValueError("unexpected_fields")
    if payload["verdict"] not in ("incorrect", "actually_correct", "unusable"):
        raise ValueError("unexpected_verdict")
    if any(not isinstance(payload[k], str) for k in FIELDS):
        raise ValueError("text_fields_required")
    if payload["verdict"] == "incorrect" and any(not payload[k].strip() for k in FIELDS):
        raise ValueError("empty_incorrect_field")


def prepare(args):
    if args.output.exists():
        raise ValueError("output_exists_refusing_overwrite")
    if not args.reference_probe:
        raise ValueError("reference_probe_required")
    reference = args.reference_probe.resolve()
    prior = json.loads((reference / "manifest.json").read_text())
    if digest(reference / "samples.json") != prior["samples_sha256"]:
        raise ValueError("baseline_sample_hash_mismatch")
    samples = json.loads((reference / "samples.json").read_text())
    validate_samples(samples, 20)
    tree = ast.parse(LEGACY.read_text())
    teacher = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "command_teacher")
    assert any(isinstance(n, ast.Constant) and n.value == SYSTEM for n in ast.walk(teacher))
    requests = {r["probe_id"]: request_body(r) for r in samples}
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "samples.json", samples)
    assert digest(args.output / "samples.json") == prior["samples_sha256"]
    save(args.output / "requests.json", requests)
    save(args.output / "manifest.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Revised prompt with thinking enabled at temperature 0.1, twenty matched samples, one request per row, no training export.",
        "baseline_directory": str(reference), "sources": prior["sources"],
        "samples_sha256": digest(args.output / "samples.json"),
        "requests_sha256": digest(args.output / "requests.json"),
        "legacy_source": str(LEGACY), "legacy_source_sha256": digest(LEGACY),
        "prompt_function_snapshot": inspect.getsource(_teacher_prompt),
        "system_prompt": SYSTEM, "request": CONFIG,
        "review_policy": {
            "prompt": "Current revised first-attempt messages; no extra examples, schema, or instructions.",
            "normalization": "May remove one whole JSON fence, logged; no JSON repair or reasoning-channel extraction.",
            "answer": "Use the returned reasoning unchanged; do not append or repair boxed answers.",
            "semantics": "Manually assess diagnosis and solution; intermediate repair formulas alone are not rejection reasons.",
        },
    })
    print("Prepared twenty matched requests; API calls: 0.", flush=True)


async def call_all(samples, requests, output, key):
    import httpx
    gate = asyncio.Semaphore(CONFIG["concurrency"])
    count = 0
    async with httpx.AsyncClient(timeout=httpx.Timeout(600, connect=30), follow_redirects=False) as client:
        async def one(row):
            nonlocal count
            prefix = output / "calls" / row["probe_id"]
            body = requests[row["probe_id"]]
            save(str(prefix) + ".request.json", body)
            result = {"probe_id": row["probe_id"], "trajectory_id": row["trajectory_id"]}
            async with gate:
                if count >= CONFIG["max_requests"]:
                    raise ValueError("request_budget_exhausted")
                count += 1
                print(f"REQUEST {count}/20 {row['probe_id']}", flush=True)
                started = time.monotonic()
                try:
                    response = await client.post(ENDPOINT, json=body, headers={"Authorization": f"Bearer {key}"})
                    result["http_status"] = response.status_code
                    # The credential is never persisted, including in provider error bodies.
                    raw = json.loads(response.text.replace(key, "[REDACTED]"))
                    save(str(prefix) + ".response.json", raw)
                    if response.status_code != 200:
                        raise ValueError(f"http_{response.status_code}")
                    choice = raw["choices"][0]
                    message = choice["message"]
                    result.update({
                        "response_model": raw.get("model"), "finish_reason": choice.get("finish_reason"),
                        "usage": raw.get("usage", {}),
                        "thinking_returned": bool(message.get("reasoning_content") or message.get("reasoning_details")),
                    })
                    if str(raw.get("model", "")).lower() not in ("deepseek-v4-flash", "deepseek-v4-flash") or choice.get("finish_reason") != "stop":
                        raise ValueError("unexpected_model_or_finish")
                    payload, normalization = parse_content(message.get("content") or "")
                    result["format_normalization"] = normalization
                    result["parsed_payload"] = payload
                    validate(payload)
                    result["structure_valid"] = True
                    if payload["verdict"] == "incorrect":
                        result["reflection_text"] = render(payload)
                        result["corrected_reasoning"] = payload["reasoning"]
                except Exception as exc:
                    result["error"] = type(exc).__name__
                result["elapsed_seconds"] = round(time.monotonic() - started, 3)
            save(str(prefix) + ".result.json", result)
            print(json.dumps({"finished": row["probe_id"], "error": result.get("error"),
                              "verdict": result.get("parsed_payload", {}).get("verdict"),
                              "thinking": result.get("thinking_returned")}), flush=True)
            return result

        results = await asyncio.gather(*(one(row) for row in samples))
    return results, count


def run(args):
    manifest = json.loads((args.output / "manifest.json").read_text())
    if manifest["request"] != CONFIG:
        raise ValueError("unexpected_configuration")
    for name in ("samples", "requests"):
        if digest(args.output / f"{name}.json") != manifest[f"{name}_sha256"]:
            raise ValueError("frozen_input_changed")
    samples = json.loads((args.output / "samples.json").read_text())
    requests = json.loads((args.output / "requests.json").read_text())
    validate_samples(samples, 20)
    if requests != {r["probe_id"]: request_body(r) for r in samples}:
        raise ValueError("legacy_prompt_changed")
    if (args.output / "calls").exists():
        raise ValueError("refusing_duplicate_paid_run")
    key = os.environ.get("PARATERA_API_KEY")
    if args.prompt_api_key:
        if not sys.stdin.isatty():
            raise ValueError("hidden_key_prompt_requires_tty")
        key = getpass.getpass("Paratera API key (hidden, not saved): ").strip()
    if not key:
        raise ValueError("api_key_missing_no_requests_sent")
    (args.output / "calls").mkdir(exist_ok=False)
    save(args.output / "run_config.json", {
        "effective_request": CONFIG, "started_utc": datetime.now(timezone.utc).isoformat(),
        "runner_sha256": digest(Path(__file__)),
    })
    results, count = asyncio.run(call_all(samples, requests, args.output, key))
    save(args.output / "results.json", results)
    usage = Counter()
    for row in results:
        usage.update({k: v for k, v in row.get("usage", {}).items() if isinstance(v, (int, float))})
    summary = {
        "actual_requests": count, "records": len(results), "perfect_api_requests": 0,
        "structure_valid": sum(bool(r.get("structure_valid")) for r in results),
        "thinking_returned": sum(bool(r.get("thinking_returned")) for r in results),
        "verdict_counts": dict(Counter(r.get("parsed_payload", {}).get("verdict", "error") for r in results)),
        "usage": dict(usage), "semantic_review": "Pending; no training export.",
    }
    save(args.output / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-probe", type=Path)
    parser.add_argument("--prompt-api-key", action="store_true")
    args = parser.parse_args()
    prepare(args) if args.action == "prepare" else run(args)




