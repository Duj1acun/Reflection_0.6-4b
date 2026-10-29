#!/usr/bin/env python3
"""Exactly twenty single-call annotations with the user's minimal English prompt."""
import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import re
import sys
import time

from probe_teacher_v1 import digest, save
from probe_teacher_v2_single import parse_content, validate_samples

ROOT = Path(__file__).resolve().parents[2]
PROMPT = ROOT / "configs/sft/teacher_simple/annotation.user.txt"
BASE = "https://llmapi.paratera.com"
CONFIG = {"model": "DeepSeek-V4-Flash", "temperature": 0.1, "top_p": 0.95,
          "max_tokens": 16384, "thinking": {"type": "enabled"}, "reasoning_effort": "high"}
FIELDS = {"verified_states", "error_reason", "new_direction", "reasoning"}


def validate_selected(samples, count):
    if count not in (10, 20) or len(samples) != count:
        raise ValueError("unexpected_sample_count")
    for field in ("probe_id", "problem_id", "trajectory_id"):
        if len({r[field] for r in samples}) != count:
            raise ValueError("duplicate_sample_identifier")
    if any(r["score_status"] != "incorrect" or r["finish_reason"] != "stop" for r in samples):
        raise ValueError("unexpected_student_sample")


def request(row, template):
    content = re.sub(r"\{(problem|answer|completion)\}", lambda m: str(row[m[1]]), template)
    return dict(CONFIG, messages=[{"role": "user", "content": content}])


def validate(payload):
    if set(payload) != FIELDS or any(not isinstance(payload[k], str) for k in FIELDS):
        raise ValueError("expected_exactly_four_string_fields")


def prepare(args):
    if args.output.exists():
        raise ValueError("output_exists")
    ref = args.reference_probe
    prior = json.loads((ref / "manifest.json").read_text())
    assert digest(ref / "samples.json") == prior["samples_sha256"]
    samples = json.loads((ref / "samples.json").read_text())
    validate_samples(samples, 20)
    if args.num_samples == 10:
        ids = {f"{model}_k1_{i:02d}" for model in ("qwen3-0.6b", "qwen3-4b") for i in range(1, 5)}
        ids.update(("qwen3-0.6b_k2_03", "qwen3-4b_k2_01"))
        samples = [r for r in samples if r["probe_id"] in ids]
    validate_selected(samples, args.num_samples)
    prompt_path = args.prompt_template or PROMPT
    template = prompt_path.read_text()
    bodies = {r["probe_id"]: request(r, template) for r in samples}
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "samples.json", samples)
    save(args.output / "requests.json", bodies)
    if args.num_samples == 20:
        assert digest(args.output / "samples.json") == prior["samples_sha256"]
    save(args.output / "manifest.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_directory": str(ref), "sources": prior["sources"],
        "samples_sha256": digest(args.output / "samples.json"),
        "requests_sha256": digest(args.output / "requests.json"),
        "prompt_path": str(prompt_path), "prompt_text": template, "prompt_sha256": digest(prompt_path),
        "endpoint": BASE + "/v1/chat/completions", "request_parameters": CONFIG,
        "max_requests": args.num_samples, "concurrency": args.num_samples, "automatic_retries": 0, "timeout_seconds": 600,
        "selection": "Matched full twenty, or ten diagnostic samples: first four k1 per model plus 0.6B sum-of-squares and 4B ant; selected before API calls.",
        "policy": "Only the translated user prompt. No system message, verdict, extra constraints, JSON repair, answer repair, or production training export.",
    })
    print(f"Prepared {len(samples)} matched samples. No API calls yet.", flush=True)


async def call_all(samples, bodies, output, key):
    import httpx
    count = 0
    async with httpx.AsyncClient(timeout=httpx.Timeout(600, connect=30), follow_redirects=False) as client:
        models_response = await client.get(BASE + "/v1/models", headers={"Authorization": f"Bearer {key}"})
        if models_response.status_code != 200:
            raise ValueError("model_listing_failed")
        ids = [r["id"] for r in models_response.json().get("data", [])]
        save(output / "model_check.json", {"http_status": 200, "requested_model": CONFIG["model"],
             "available": CONFIG["model"] in ids, "deepseek_model_ids": [i for i in ids if "deepseek" in i.lower()]})
        if CONFIG["model"] not in ids:
            raise ValueError("requested_model_not_available_no_generation_sent")
        (output / "calls").mkdir(exist_ok=False)

        async def one(row):
            nonlocal count
            if count >= len(samples):
                raise ValueError("request_budget_exhausted")
            count += 1
            pid = row["probe_id"]
            prefix = output / "calls" / pid
            save(str(prefix) + ".request.json", bodies[pid])
            result = {"probe_id": pid, "trajectory_id": row["trajectory_id"]}
            print(f"REQUEST {count}/{len(samples)} {pid}", flush=True)
            started = time.monotonic()
            try:
                response = await client.post(BASE + "/v1/chat/completions", json=bodies[pid],
                                             headers={"Authorization": f"Bearer {key}"})
                result["http_status"] = response.status_code
                raw = json.loads(response.text.replace(key, "[REDACTED]"))
                save(str(prefix) + ".response.json", raw)
                if response.status_code != 200:
                    raise ValueError("non_200_response")
                choice = raw["choices"][0]
                message = choice["message"]
                result.update(response_model=raw.get("model"), finish_reason=choice.get("finish_reason"),
                              thinking_returned=bool(message.get("reasoning_content") or message.get("reasoning_details")),
                              usage=raw.get("usage", {}))
                if str(raw.get("model", "")).lower() != CONFIG["model"].lower():
                    raise ValueError("unexpected_response_model")
                if choice.get("finish_reason") != "stop":
                    raise ValueError("not_normally_finished")
                payload, normalization = parse_content(message.get("content") or "")
                result.update(parsed_payload=payload, format_normalization=normalization)
                validate(payload)
                result.update(structure_valid=True, all_fields_nonempty=all(payload[k].strip() for k in FIELDS))
            except Exception as exc:
                result["error_type"] = type(exc).__name__
            result["elapsed_seconds"] = round(time.monotonic() - started, 3)
            save(str(prefix) + ".result.json", result)
            print(json.dumps({"finished": pid, "http": result.get("http_status"),
                              "thinking": result.get("thinking_returned"), "valid": result.get("structure_valid", False)}), flush=True)
            return result

        first = await one(samples[0])
        results = [first]
        if first.get("http_status") == 200 and first.get("thinking_returned") and str(first.get("response_model", "")).lower() == CONFIG["model"].lower():
            results.extend(await asyncio.gather(*(one(r) for r in samples[1:])))
    return results, count


def run(args):
    out = args.output
    manifest = json.loads((out / "manifest.json").read_text())
    for name in ("samples", "requests"):
        assert digest(out / f"{name}.json") == manifest[f"{name}_sha256"]
    assert manifest["request_parameters"] == CONFIG and manifest["max_requests"] == args.num_samples
    samples = json.loads((out / "samples.json").read_text())
    validate_selected(samples, args.num_samples)
    bodies = json.loads((out / "requests.json").read_text())
    assert bodies == {r["probe_id"]: request(r, manifest["prompt_text"]) for r in samples}
    if (out / "run_config.json").exists() or (out / "calls").exists():
        raise ValueError("refusing_duplicate_paid_run")
    key = os.environ.get("PARATERA_API_KEY")
    if args.prompt_api_key:
        if not sys.stdin.isatty():
            raise ValueError("hidden_prompt_requires_tty")
        key = getpass.getpass("Paratera API key (hidden, not saved): ").strip()
    if not key:
        raise ValueError("missing_api_key")
    save(out / "run_config.json", {"started_utc": datetime.now(timezone.utc).isoformat(),
                                   "parameters": CONFIG, "runner_sha256": digest(Path(__file__))})
    results, count = asyncio.run(call_all(samples, bodies, out, key))
    save(out / "results.json", results)
    usage = Counter()
    for r in results:
        usage.update({k: v for k, v in r.get("usage", {}).items() if isinstance(v, (int, float))})
    summary = {"actual_requests": count, "records": len(results),
               "http_200": sum(r.get("http_status") == 200 for r in results),
               "thinking_returned": sum(bool(r.get("thinking_returned")) for r in results),
               "structure_valid": sum(bool(r.get("structure_valid")) for r in results),
               "usage": dict(usage), "perfect_api_requests": 0, "training_exported": False}
    save(out / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "run"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference-probe", type=Path)
    p.add_argument("--prompt-template", type=Path, help="Preparation only; run uses the frozen prompt snapshot.")
    p.add_argument("--prompt-api-key", action="store_true")
    p.add_argument("--model", choices=("DeepSeek-V4-Flash", "DeepSeek-V4-Pro"), default="DeepSeek-V4-Flash")
    p.add_argument("--num-samples", type=int, choices=(10, 20), default=20)
    args = p.parse_args()
    CONFIG["model"] = args.model
    prepare(args) if args.action == "prepare" else run(args)
