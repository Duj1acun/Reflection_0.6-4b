#!/usr/bin/env python3
"""Read-only scoring of saved model outputs; write separate inspection artifacts."""
import argparse
from collections import Counter
import json
from pathlib import Path

from probe_teacher_v1 import digest, save
from rgrpo.math_score_light import compute_math_score, extract_answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-reference-score", action="store_true", help="Do not apply the legacy answer extractor to unboxed prose.")
    args = parser.parse_args()
    root = args.output
    manifest = json.loads((root / "manifest.json").read_text())
    assert digest(root / "samples.json") == manifest["samples_sha256"]
    assert digest(root / "requests.json") == manifest["requests_sha256"]
    samples = json.loads((root / "samples.json").read_text())
    results = {r["probe_id"]: r for r in json.loads((root / "results.json").read_text())}
    records, stats = [], Counter()
    for s in samples:
        result = results.get(s["probe_id"], {})
        path = root / "calls" / (s["probe_id"] + ".response.json")
        raw = json.loads(path.read_text()) if path.exists() else {}
        content = ((raw.get("choices") or [{}])[0].get("message") or {}).get("content", "")
        payload = result.get("parsed_payload")
        reasoning = payload.get("reasoning", "") if isinstance(payload, dict) else ""
        score = compute_math_score(generated=reasoning, ground_truth=s["answer"]) if reasoning and not args.skip_reference_score else None
        stats["records"] += 1
        stats["http_200"] += result.get("http_status") == 200
        stats["normal_finish"] += result.get("finish_reason") == "stop"
        stats["thinking_returned"] += bool(result.get("thinking_returned"))
        stats["strict_four_fields"] += bool(result.get("structure_valid"))
        stats["parsed_json_objects"] += isinstance(payload, dict)
        stats["single_fence_removed"] += bool(result.get("format_normalization"))
        stats["gold_score_matches"] += score == 1.0
        records.append({
            "probe_id": s["probe_id"], "student_model": s["student_model"],
            "problem_id": s["problem_id"], "trajectory_id": s["trajectory_id"],
            "problem": s["problem"], "gold_answer": s["answer"],
            "student_solution": s["completion"], "annotation": payload,
            "raw_final_content": content, "checks": {
                "http_status": result.get("http_status"), "finish_reason": result.get("finish_reason"),
                "thinking_returned": result.get("thinking_returned"),
                "four_field_structure_valid": result.get("structure_valid", False),
                "format_normalization": result.get("format_normalization"),
                "extracted_answer": extract_answer(reasoning) if reasoning else None,
                "score_against_original_gold": score, "error_type": result.get("error_type"),
            }, "production_training_exported": False,
        })
    with (root / "annotations.jsonl").open("x", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    fence = "```"
    with (root / "ANNOTATIONS.md").open("x", encoding="utf-8") as f:
        f.write(f"# 英文 prompt：{len(records)} 条 API 最终输出\n\n仅展示最终 content，不展示内部 thinking；原输出没有修复、补字段或改写。答案匹配不是完整数学质量判定。\n\n")
        for index, row in enumerate(records, 1):
            f.write(f"## {index}. {row['probe_id']}\n\n题目：{row['problem']}\n\n参考答案：{row['gold_answer']}\n\n")
            f.write("检查：" + json.dumps(row["checks"], ensure_ascii=False) + "\n\n")
            f.write("### API 最终原文\n\n" + fence + "text\n" + row["raw_final_content"] + "\n" + fence + "\n\n")
            f.write("### 学生原始回答\n\n" + row["student_solution"] + "\n\n")
    report = dict(stats, scope="Automatic structural and reference-answer checks only; not a semantic acceptance rate.",
                  scorer_warning="The legacy extractor may select prose rather than a final number when no boxed answer is present. These raw scores must not be reported as mathematical accuracy; inspect the actual conclusions.")
    report["reference_scoring_performed"] = not args.skip_reference_score
    if args.skip_reference_score:
        report["gold_score_matches"] = None
    save(root / "audit_summary.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
