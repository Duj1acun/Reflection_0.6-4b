#!/usr/bin/env python3
"""Build local teacher audit artifacts; no API calls or production exports."""
import argparse
from collections import Counter
import json
from pathlib import Path

from probe_teacher_v2_single import corrected_reasoning, parse_content, render, validate
from probe_teacher_v1 import digest, save
from rgrpo.math_score_light import compute_math_score, extract_answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    manifest = json.loads((root / "manifest.json").read_text())
    assert digest(root / "samples.json") == manifest["samples_sha256"]
    samples = json.loads((root / "samples.json").read_text())
    reviews = {r["probe_id"]: r for r in json.loads((root / "manual_review.json").read_text())["records"]}
    assert set(reviews) == {r["probe_id"] for r in samples}
    records, usage, flags, models = [], Counter(), Counter(), {}
    for row in samples:
        pid = row["probe_id"]
        raw = json.loads((root / "calls" / (pid + ".response.json")).read_text())
        result = json.loads((root / "calls" / (pid + ".result.json")).read_text())
        choice = raw["choices"][0]
        payload, normalization = parse_content(choice["message"]["content"])
        validate(payload, manifest)
        assert payload["verdict"] == "incorrect"
        reasoning = corrected_reasoning(payload)
        score = compute_math_score(generated=reasoning, ground_truth=row["answer"])
        usage.update({k: raw.get("usage", {}).get(k, 0) for k in ("prompt_tokens", "completion_tokens", "total_tokens")})
        review = reviews[pid]
        flags.update(review["flags"])
        stats = models.setdefault(row["student_model"], Counter())
        stats.update(records=1, gold_matches=int(score == 1.0), accept_candidates=int(review["decision"] == "accept_candidate"))
        records.append({
            "probe_id": pid, "student_model": row["student_model"], "problem_id": row["problem_id"],
            "trajectory_id": row["trajectory_id"], "sample_index": row["sample_index"],
            "correct_out_of_four": row["correct_out_of_four"], "problem": row["problem"],
            "original_reference_answer": row["answer"], "student_response": row["completion"],
            "teacher_annotation": payload, "reflection_text": render(payload),
            "corrected_reasoning": reasoning,
            "automatic_checks": {
                "http_status": result["http_status"], "response_model": raw.get("model"),
                "finish_reason": choice.get("finish_reason"), "thinking_returned": result["thinking_returned"],
                "schema_valid": True, "format_normalization": normalization,
                "extracted_final_answer": extract_answer(reasoning), "score_against_original_reference": score,
            },
            "manual_review": review, "production_training_exported": False,
            "raw_response_file": str(root / "calls" / (pid + ".response.json")),
        })
    with (root / "annotations_reviewed.jsonl").open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    summary = {
        "records": len(records), "actual_api_requests": len(records),
        "all_http_200": all(r["automatic_checks"]["http_status"] == 200 for r in records),
        "all_finished_stop": all(r["automatic_checks"]["finish_reason"] == "stop" for r in records),
        "all_thinking_returned": all(r["automatic_checks"]["thinking_returned"] for r in records),
        "structural_valid": len(records),
        "removed_single_json_fences": sum(bool(r["automatic_checks"]["format_normalization"]) for r in records),
        "gold_matches": sum(r["automatic_checks"]["score_against_original_reference"] == 1.0 for r in records),
        "manual_accept_candidates": sum(r["manual_review"]["decision"] == "accept_candidate" for r in records),
        "manual_review_required": sum(r["manual_review"]["decision"] != "accept_candidate" for r in records),
        "per_student_model": models, "overlapping_issue_counts": flags, "usage": usage,
        "gold_mismatch_note": "Both mismatches were independently verified as reference-answer problems, not teacher final-answer errors.",
        "perfect_api_requests": 0, "production_training_exported": False,
        "scope": "20 stratified one-shot samples; conservative manual screen, not population accuracy.",
    }
    save(root / "audit_summary.json", summary)
    fence = chr(96) * 3
    with (root / "SAMPLES_REVIEWED.md").open("x", encoding="utf-8") as handle:
        handle.write("# 20 条标注与逐条复核\n\n这里只展示最终标注，不展示 teacher 内部 thinking。原文、字段和审核意见分开保留；未人工修正模型输出。\n\n")
        for index, r in enumerate(records, 1):
            handle.write(f"## {index}. {r['probe_id']}\n\n")
            handle.write(f"审核：{r['manual_review']['decision']}。{r['manual_review']['note']}\n\n")
            handle.write(f"题目：{r['problem']}\n\n原参考：{r['original_reference_answer']}；teacher：{r['teacher_annotation']['final_answer']}。\n\n")
            handle.write("### 模型最终 JSON（标准序列化展示）\n\n" + fence + "json\n" + json.dumps(r["teacher_annotation"], ensure_ascii=False, indent=2) + "\n" + fence + "\n\n")
            handle.write("### 学生原始错误回答\n\n" + r["student_response"] + "\n\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
