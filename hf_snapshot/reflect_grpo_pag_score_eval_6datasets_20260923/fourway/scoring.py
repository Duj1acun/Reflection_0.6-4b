"""One outcome judge for all methods, protected by killable CPU workers."""
import concurrent.futures
import multiprocessing
import re
import threading
import os
from functools import lru_cache

_pool = None
_lock = threading.Lock()
_WORKER_TIMEOUT_SECONDS = 15


class JudgeError(RuntimeError):
    """Unresolved scoring failure, distinguishable from model/trainer errors."""


def _worker_setup():
    # Bound additional address space in CPU-only scoring children. Account for
    # libraries already mapped by spawn instead of assuming a fixed base size.
    import resource
    from pathlib import Path
    mapped = int(Path("/proc/self/statm").read_text().split()[0]) * os.sysconf("SC_PAGE_SIZE")
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    limit = mapped + 512 * 1024**2
    if soft != resource.RLIM_INFINITY:
        limit = min(limit, soft)
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))

@lru_cache(maxsize=8)
def _scorer_version(path):
    from .contracts import config
    return config(path).get("scorer_version", "paired-v4") if path else "paired-v4"

def score_inline(text, gold, dataset=None):
    version = _scorer_version(os.environ.get("FOURWAY_CONFIG"))
    if version == "fourway-math-v5-20260914":
        from .math_scoring import score_sample
    elif version == "paired-v4":
        from evaluation.evaluators.paired_scoring import score_sample
    else:
        raise ValueError(f"Unknown scorer version: {version}")
    kind = dataset or ("amc23" if re.fullmatch(r"[+-]?\d+", str(gold).strip()) else "math500")
    return score_sample((text, str(gold), kind))

def _reset():
    global _pool
    if _pool is not None:
        processes = list((_pool._processes or {}).values())
        for proc in processes:
            proc.terminate()
        for proc in processes:
            proc.join(timeout=0.5)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=0.5)
        _pool.shutdown(wait=False, cancel_futures=True)
    _pool = None

def judge(text, gold, dataset=None):
    from math_verify.errors import TimeoutException
    global _pool
    with _lock:
        for attempt in range(2):
            try:
                if _pool is None:
                    _pool = concurrent.futures.ProcessPoolExecutor(
                        max_workers=1, mp_context=multiprocessing.get_context("spawn"), initializer=_worker_setup)
                result = _pool.submit(score_inline, text, gold, dataset).result(timeout=_WORKER_TIMEOUT_SECONDS)
                if result["score_status"] not in {"exception", "timeout"}:
                    return result
                error = result
            except (Exception, TimeoutException) as exc:
                error = f"{type(exc).__name__}: {exc}"
            _reset()
        # Central logging also covers PAG/SCoRe, whose native managers propagate
        # failures without the GRPO manager's additional diagnostic record.
        destination = None
        if os.environ.get("FOURWAY_CONFIG"):
            import time
            from pathlib import Path
            from .contracts import config, atomic_json
            destination = Path(config()["output"]) / "diagnostics/reward_errors" / f"judge_{os.getpid()}_{time.time_ns()}.json"
            atomic_json(destination, dict(response_text=text, ground_truth=str(gold), dataset=dataset,
                                         method=os.environ.get("FOURWAY_METHOD"), error=str(error)))
        raise JudgeError(f"Unresolved math judge error; refusing to count it as wrong: {error}; diagnostic={destination}")

def policy_score(text, gold):
    r = judge(text, gold)
    return {"score": float(r["correct"]), "acc": float(r["correct"]), "pred": r["extracted"],
            "score_status": r["score_status"], "scorer_version": r["scorer_version"]}
