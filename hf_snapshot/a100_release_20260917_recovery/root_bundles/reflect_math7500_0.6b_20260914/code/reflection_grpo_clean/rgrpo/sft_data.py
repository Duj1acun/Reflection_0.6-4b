"""Utilities for reproducible, backbone-specific cold-start SFT data."""

from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

from rgrpo.chat_template import apply_chat_template


HARD_THRESHOLD = 1.0 / 64.0
SYSTEM_PROMPT = (
    "You are a math reasoning assistant. "
    "When you make an error during reasoning, "
    "use <reflection>...</reflection> to reflect on the mistake, "
    "then continue reasoning based on the reflection."
)


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip()


def problem_id(problem: str) -> str:
    return hashlib.sha256(normalize_text(problem).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            yield value


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _source_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        0 if row.get("source_split") == "train" else 1,
        normalize_text(row.get("source")),
        json.dumps(row.get("domain", []), ensure_ascii=False, sort_keys=True),
        normalize_text(row.get("answer")),
    )


def select_hard_records(
    rows: Iterable[dict[str, Any]],
    *,
    threshold: float = HARD_THRESHOLD,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Filter and conservatively deduplicate Big-Math rows.

    Duplicate normalized questions with different normalized answers are
    quarantined as a whole instead of choosing one label arbitrarily.
    """

    raw = 0
    empty = 0
    invalid_rate = 0
    above_threshold = 0
    eligible: list[dict[str, Any]] = []
    by_rate: dict[str, int] = defaultdict(int)

    for source_index, input_row in enumerate(rows):
        raw += 1
        row = dict(input_row)
        problem = normalize_text(row.get("problem"))
        answer = normalize_text(row.get("answer"))
        if not problem or not answer:
            empty += 1
            continue
        try:
            rate = float(row.get("llama8b_solve_rate"))
        except (TypeError, ValueError):
            invalid_rate += 1
            continue
        if rate < 0 or rate > threshold + 1e-12:
            above_threshold += 1
            continue
        row["problem"] = problem
        row["answer"] = answer
        row["llama8b_solve_rate"] = rate
        row["_source_index"] = source_index
        eligible.append(row)
        by_rate[f"{rate:.6f}"] += 1

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        groups[normalize_text(row["problem"])].append(row)

    selected: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    duplicate_groups = 0
    duplicate_extra_rows = 0

    for normalized_problem, group in groups.items():
        if len(group) > 1:
            duplicate_groups += 1
            duplicate_extra_rows += len(group) - 1
        answers = {normalize_text(row["answer"]) for row in group}
        pid = problem_id(normalized_problem)
        if len(answers) > 1:
            conflicts.append(
                {
                    "problem_id": pid,
                    "problem": normalized_problem,
                    "reason": "conflicting_duplicate_answers",
                    "answers": sorted(answers),
                    "rows": [
                        {k: v for k, v in row.items() if not k.startswith("_")}
                        for row in sorted(group, key=_source_key)
                    ],
                }
            )
            continue

        canonical = sorted(group, key=_source_key)[0]
        source_splits = sorted({str(row.get("source_split", "")) for row in group})
        selected.append(
            {
                "problem_id": pid,
                "problem": normalized_problem,
                "answer": canonical["answer"],
                "source": canonical.get("source"),
                "domain": canonical.get("domain", []),
                "llama8b_solve_rate": canonical["llama8b_solve_rate"],
                "source_splits": source_splits,
                "duplicate_count": len(group),
            }
        )

    selected.sort(key=lambda row: row["problem_id"])
    conflicts.sort(key=lambda row: row["problem_id"])
    stats = {
        "raw_rows": raw,
        "empty_problem_or_answer": empty,
        "invalid_solve_rate": invalid_rate,
        "above_threshold": above_threshold,
        "eligible_rows": len(eligible),
        "eligible_by_rate": dict(sorted(by_rate.items())),
        "normalized_unique_questions": len(groups),
        "duplicate_groups": duplicate_groups,
        "duplicate_extra_rows": duplicate_extra_rows,
        "conflict_groups": len(conflicts),
        "selected_questions": len(selected),
        "threshold": threshold,
    }
    return selected, conflicts, stats


def select_pilot_records(
    records: Sequence[dict[str, Any]], limit: int, seed: int = 42
) -> list[dict[str, Any]]:
    """Round-robin across difficulty/source/domain for a representative pilot."""

    if limit <= 0 or limit >= len(records):
        return list(records)
    buckets: dict[tuple[str, str, str], deque[dict[str, Any]]] = {}
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        domain = record.get("domain") or []
        domain0 = str(domain[0] if isinstance(domain, list) and domain else domain)
        key = (
            f"{float(record.get('llama8b_solve_rate', 0)):.6f}",
            str(record.get("source", "")),
            domain0,
        )
        grouped[key].append(record)
    for key, values in grouped.items():
        values.sort(
            key=lambda row: hashlib.sha256(
                f"{seed}:{row['problem_id']}".encode("utf-8")
            ).hexdigest()
        )
        buckets[key] = deque(values)
    keys = sorted(buckets)
    chosen: list[dict[str, Any]] = []
    while len(chosen) < limit:
        progressed = False
        for key in keys:
            if buckets[key]:
                chosen.append(buckets[key].popleft())
                progressed = True
                if len(chosen) == limit:
                    break
        if not progressed:
            break
    return chosen


def assign_exact_splits(
    records: Sequence[dict[str, Any]], *, seed: int = 42, eval_fraction: float = 0.1
) -> list[dict[str, Any]]:
    values = [dict(record) for record in records]
    order = list(range(len(values)))
    random.Random(seed).shuffle(order)
    eval_count = 0 if not values else max(1, round(len(values) * eval_fraction))
    eval_indices = set(order[:eval_count])
    for idx, value in enumerate(values):
        value["split"] = "eval" if idx in eval_indices else "train"
    return values


def deterministic_resample(
    records: Sequence[dict[str, Any]], size: int, *, seed: int, salt: str
) -> list[dict[str, Any]]:
    if size <= 0:
        return []
    if not records:
        raise ValueError(f"Cannot sample {size} records from an empty {salt} pool.")
    indices = list(range(len(records)))
    random.Random(f"{seed}:{salt}").shuffle(indices)
    result: list[dict[str, Any]] = []
    while len(result) < size:
        for idx in indices:
            result.append(dict(records[idx]))
            if len(result) == size:
                break
    return result


def build_paper_phase2_mix(
    reflection_records: Sequence[dict[str, Any]],
    perfect_records: Sequence[dict[str, Any]],
    *,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Build an exact-by-count approximation to the paper's 64:19:17 mix."""

    n_reflection = len(reflection_records)
    if n_reflection == 0:
        raise ValueError("Phase 2 paper mix requires reflection records.")
    n_replay = round(n_reflection * 19 / 64)
    n_perfect = round(n_reflection * 17 / 64)
    main = [dict(record, type="reflection") for record in reflection_records]
    replay_pool = [dict(record, type="phase1_replay") for record in reflection_records]
    perfect_pool = [dict(record, type="perfect") for record in perfect_records]
    replay = deterministic_resample(replay_pool, n_replay, seed=seed, salt="replay")
    perfect = deterministic_resample(perfect_pool, n_perfect, seed=seed, salt="perfect")
    result = main + replay + perfect
    random.Random(f"{seed}:phase2_mix").shuffle(result)
    return result


def extract_json_objects(text: str) -> list[dict[str, Any]]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    decoder = json.JSONDecoder()
    values: list[dict[str, Any]] = []
    seen: set[str] = set()
    for match in re.finditer(r"\{", cleaned):
        try:
            value, _ = decoder.raw_decode(cleaned[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            key = json.dumps(value, ensure_ascii=False, sort_keys=True)
            if key not in seen:
                seen.add(key)
                values.append(value)
    return values


def extract_json_object(text: str) -> dict[str, Any]:
    values = extract_json_objects(text)
    if values:
        return values[0]
    raise ValueError("Teacher response does not contain a JSON object.")


def build_reflection_text(payload: dict[str, Any]) -> str:
    verified = normalize_text(payload.get("verified_states"))
    reason = normalize_text(payload.get("error_reason"))
    direction = normalize_text(payload.get("new_direction"))
    return (
        f"<reflection> So far I have: {verified}.\n"
        f"However, {reason}. {direction}\n"
        "</reflection>"
    )


def _repair_unboxed_numeric_conclusion(reasoning: str, answer: Any) -> str | None:
    """Canonicalize a clear numeric conclusion without blessing contradictory scratch work."""
    reference = str(answer).strip()
    if not re.fullmatch(r"[+-]?\d+(?:,\d{3})*(?:\.\d+)?", reference):
        return None
    tail = reasoning[-500:]
    if re.search(
        r"\b(?:but|however|still\s+not|does\s+not|doesn't|wrong|incorrect|"
        r"mismatch|gold\s+answer|recheck|final\s+check)\b",
        tail,
        flags=re.IGNORECASE,
    ):
        return None
    if not re.search(
        r"\b(?:thus|hence|therefore|giving|remainder|maximum|answer|so)\b",
        tail,
        flags=re.IGNORECASE,
    ):
        return None
    candidates = re.findall(r"(?<![\w.])[+-]?\d+(?:,\d{3})*(?:\.\d+)?(?!\w)", tail)
    if not candidates:
        return None
    from rgrpo.math_score_light import math_equal

    if not math_equal(candidates[-1].replace(",", ""), reference.replace(",", "")):
        return None
    return reasoning.rstrip() + f"\n\nTherefore, the final answer is \\boxed{{{reference}}}."


def training_lengths(
    tokenizer: Any,
    *,
    problem: str,
    cut_point: str | None = None,
    reflection_text: str | None = None,
    reasoning: str | None = None,
    full_reasoning: str | None = None,
) -> dict[str, int]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]
    header = apply_chat_template(
        tokenizer, messages, tokenize=False, add_generation_prompt=True
    )
    eos = tokenizer.eos_token or ""
    output: dict[str, int] = {}
    if full_reasoning is not None:
        output["perfect"] = len(
            tokenizer(header + full_reasoning + eos, add_special_tokens=False)["input_ids"]
        )
    if cut_point is not None and reflection_text is not None:
        output["phase1"] = len(
            tokenizer(
                header + cut_point.rstrip() + "\n" + reflection_text + eos,
                add_special_tokens=False,
            )["input_ids"]
        )
    if reflection_text is not None and reasoning is not None:
        output["phase2"] = len(
            tokenizer(
                header + reflection_text.rstrip() + "\n" + reasoning + eos,
                add_special_tokens=False,
            )["input_ids"]
        )
    return output


def validate_teacher_payload(
    payload: dict[str, Any],
    source: dict[str, Any],
    *,
    scorer: Callable[[str, Any], float] | None = None,
    tokenizer: Any | None = None,
    max_len: int = 2048,
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    verdict = normalize_text(payload.get("verdict")).lower()
    if verdict not in {"incorrect", "actually_correct", "unusable"}:
        return None, ["invalid_verdict"]
    normalized = {"verdict": verdict}
    if verdict != "incorrect":
        return normalized, []

    fields = (
        "verified_states",
        "error_reason",
        "new_direction",
        "reasoning",
    )
    for field in fields:
        value = str(payload.get(field, "")).strip()
        if not value:
            errors.append(f"empty_{field}")
        normalized[field] = value
    if errors:
        return None, errors

    completion = str(source.get("completion", "")).rstrip()
    if not completion:
        errors.append("empty_student_completion")
    # Phase 1 now teaches post-hoc self-correction: the complete failed attempt is the
    # context and the reflection is the target.  MiniMax describes the faulty reasoning
    # semantically and is not responsible for copying or locating an exact source span.
    cut_point = completion
    normalized["cut_point"] = cut_point
    normalized["reflection_context_policy"] = "full_student_completion"
    reflection_text = build_reflection_text(normalized)
    if reflection_text.count("<reflection>") != 1 or reflection_text.count("</reflection>") != 1:
        errors.append("invalid_reflection_tags")
    normalized["reflection_text"] = reflection_text

    if scorer is not None:
        try:
            if float(scorer(normalized["reasoning"], source.get("answer"))) != 1.0:
                repaired = _repair_unboxed_numeric_conclusion(
                    normalized["reasoning"], source.get("answer")
                )
                if repaired is None or float(scorer(repaired, source.get("answer"))) != 1.0:
                    errors.append("corrected_reasoning_wrong_answer")
                else:
                    normalized["reasoning"] = repaired
                    normalized["reasoning_answer_format_repair"] = "boxed_numeric_conclusion"
        except Exception as exc:  # pragma: no cover - defensive around symbolic scorer
            errors.append(f"corrected_reasoning_score_error:{type(exc).__name__}")

    if tokenizer is not None:
        lengths = training_lengths(
            tokenizer,
            problem=str(source.get("problem", "")),
            cut_point=cut_point,
            reflection_text=reflection_text,
            reasoning=normalized["reasoning"],
        )
        normalized["training_token_lengths"] = lengths
        if any(length > max_len for length in lengths.values()):
            errors.append("training_sequence_too_long")
    return (None if errors else normalized), errors


def compile_sft_records(
    scored_records: Sequence[dict[str, Any]],
    teacher_records: Sequence[dict[str, Any]],
    *,
    tokenizer: Any | None = None,
    max_len: int = 2048,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    teacher_by_id = {str(row.get("problem_id")): row for row in teacher_records}
    reflective: list[dict[str, Any]] = []
    perfect: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []

    for row in scored_records:
        pid = str(row.get("problem_id"))
        status = row.get("score_status")
        common = {
            "problem_id": pid,
            "problem": row.get("problem"),
            "source": row.get("source"),
            "domain": row.get("domain", []),
            "llama8b_solve_rate": row.get("llama8b_solve_rate"),
            "backbone": row.get("model"),
        }
        if status == "correct":
            record = {**common, "full_reasoning": row.get("completion", "")}
            if tokenizer is not None:
                lengths = training_lengths(
                    tokenizer,
                    problem=str(record["problem"]),
                    full_reasoning=str(record["full_reasoning"]),
                )
                record["training_token_lengths"] = lengths
                if any(length > max_len for length in lengths.values()):
                    quarantine.append({**record, "reason": "training_sequence_too_long"})
                    continue
            perfect.append(record)
            continue

        if status != "incorrect":
            quarantine.append({**common, "reason": row.get("score_reason", "verifier_quarantine")})
            continue

        teacher = teacher_by_id.get(pid)
        if not teacher or teacher.get("validation_status") != "valid":
            quarantine.append({**common, "reason": "missing_or_invalid_teacher_record"})
            continue
        result = teacher.get("result", {})
        if result.get("verdict") != "incorrect":
            quarantine.append({**common, "reason": f"teacher_{result.get('verdict', 'unknown')}"})
            continue
        record = {
            **common,
            "cut_point": result["cut_point"],
            "reflection_context_policy": result.get(
                "reflection_context_policy", "full_student_completion"
            ),
            "reflection_text": result["reflection_text"],
            "reasoning": result["reasoning"],
        }
        if tokenizer is not None:
            lengths = training_lengths(
                tokenizer,
                problem=str(record["problem"]),
                cut_point=str(record["cut_point"]),
                reflection_text=str(record["reflection_text"]),
                reasoning=str(record["reasoning"]),
            )
            record["training_token_lengths"] = lengths
            if any(length > max_len for length in lengths.values()):
                quarantine.append({**record, "reason": "training_sequence_too_long"})
                continue
        reflective.append(record)

    reflective = assign_exact_splits(sorted(reflective, key=lambda x: x["problem_id"]), seed=seed)
    perfect = assign_exact_splits(sorted(perfect, key=lambda x: x["problem_id"]), seed=seed)
    quarantine.sort(key=lambda x: x["problem_id"])
    return reflective, perfect, quarantine
