"""CPU-only audit of all retained labels before a new MATH experiment."""
import json
import time
from collections import Counter
from pathlib import Path

from .contracts import atomic_json, file_sha
from .math_scoring import (SCORER_VERSION, literal_choice, literal_text, base_notation, clock_value, normalize_answer,
                           parse_answer, score_sample, simple_number)


def audit(c):
    from rgrpo.math_score_light import _has_base_notation, _score_alarm
    root = Path(c["output"])
    paths = [root / "data" / f"{split}.canonical.jsonl" for split in ("train", "validation")]
    fingerprints = {str(p): file_sha(p) for p in paths + [Path(__file__), Path(__file__).with_name("math_scoring.py")]}
    dest = root / "preparation/math_label_audit.json"
    if dest.exists():
        old = json.loads(dest.read_text())
        if old.get("fingerprints") == fingerprints and old.get("status") == "passed":
            return old
    started = time.monotonic()
    rows = [json.loads(line) for p in paths for line in p.read_text().splitlines() if line.strip()]
    labels = list(dict.fromkeys(row["gold"] for row in rows))
    errors, modes = [], Counter()
    for gold in labels:
        try:
            with _score_alarm(8):
                normalized = normalize_answer(gold)
                if not normalized:
                    raise ValueError("Empty label survived cleaning")
                if literal_choice(normalized):
                    mode = "choice"
                elif clock_value(normalized, reference=True) is not None:
                    mode = "clock_time"
                elif base_notation(normalized):
                    mode = "base_notation"
                elif literal_text(normalized) is not None:
                    mode = "literal_text"
                elif simple_number(normalized) is not None:
                    mode = "rational"
                else:
                    parsed = parse_answer(normalized)
                    if parsed is None:
                        raise ValueError("No parsed reference")
                    mode = type(parsed).__name__
                modes[mode] += 1
        except Exception as exc:
            errors.append(dict(gold=gold, stage="parse", error=f"{type(exc).__name__}: {exc}"))
    # This checks extraction on every supplied answer (including unusual braces),
    # separately from parsing above, which cannot pass via exact string matching.
    for gold in labels:
        result = score_sample((r"\boxed{" + gold + "}", gold, "math7500"))
        if result["correct"] != 1 or result["score_status"] != "ok":
            errors.append(dict(gold=gold, stage="gold_self_score", result=result))
    report = dict(status="failed" if errors else "passed", scorer_version=SCORER_VERSION,
                  rows=len(rows), unique_labels=len(labels), parser_modes=dict(modes),
                  errors=errors, fingerprints=fingerprints, elapsed_seconds=time.monotonic()-started,
                  limitation="Parser/extraction compatibility audit; does not establish that source labels are mathematically correct.")
    atomic_json(dest, report)
    if errors:
        raise ValueError(f"MATH label audit failed for {len(errors)} cases: {dest}")
    return report
