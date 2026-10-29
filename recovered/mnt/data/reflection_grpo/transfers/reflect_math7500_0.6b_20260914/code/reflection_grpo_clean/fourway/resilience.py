"""Skip unresolved judge batches before updates, with an auditable stop limit."""
import os
import time
import traceback
from pathlib import Path

from .contracts import atomic_json, config
from .scoring import JudgeError


def is_judge_error(error):
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, JudgeError):
            return True
        # Ray preserves the remote exception as .cause; local managers use
        # explicit exception chaining. Do not match error-message substrings.
        pending.extend((getattr(current, "__cause__", None), getattr(current, "cause", None)))
    return False


class BatchJudgeGuard:
    def __init__(self, trainer, logger, batch_dict, epoch):
        self.trainer, self.logger = trainer, logger
        self.batch_dict, self.epoch = batch_dict, epoch
        self.skipped = False
        self.updating = False
        self.settings = config()

    def __enter__(self):
        return self

    def mark_updating(self):
        # Once either optimizer may have run, skipping would hide partial updates.
        self.updating = True
        # Checkpoints are saved inside the guard, before __exit__. Persist the
        # reset streak once all reward computations for this batch succeeded.
        self.trainer._fourway_consecutive_judge_failures = 0

    def __exit__(self, kind, error, tb):
        if error is None:
            self.trainer._fourway_consecutive_judge_failures = 0
            return False
        if (self.updating or self.settings.get("judge_failure_policy", "raise") != "skip_batch"
                or not is_judge_error(error)):
            return False
        trainer = self.trainer
        consecutive = getattr(trainer, "_fourway_consecutive_judge_failures", 0) + 1
        total = getattr(trainer, "_fourway_skipped_batches", 0) + 1
        trainer._fourway_consecutive_judge_failures = consecutive
        trainer._fourway_skipped_batches = total
        limit = int(self.settings.get("judge_max_consecutive_failures", 3))
        if limit < 1:
            raise ValueError("judge_max_consecutive_failures must be positive")
        destination = (Path(trainer.config.trainer.default_local_dir) / "diagnostics" /
                       f"skipped_batch_{trainer.global_steps}_{time.time_ns()}.json")
        ids = [str(info.get("index", "")) for info in self.batch_dict.get("extra_info", [])]
        atomic_json(destination, dict(
            method=os.environ.get("FOURWAY_METHOD"), step=int(trainer.global_steps), epoch=self.epoch,
            question_ids=ids, consecutive_failures=consecutive, skipped_batches=total,
            action="stop" if consecutive >= limit else "skip_batch",
            error=f"{type(error).__name__}: {error}", traceback="".join(traceback.format_exception(kind, error, tb))))
        print(f"FOURWAY_JUDGE_BATCH_SKIPPED step={trainer.global_steps} consecutive={consecutive}/{limit} "
              f"total={total} diagnostic={destination}", flush=True)
        self.logger.log(data={"judge/skipped_batches": total, "judge/consecutive_failures": consecutive},
                        step=trainer.global_steps)
        if consecutive >= limit:
            raise JudgeError(f"Stopping after {consecutive} consecutive judge-failed batches; diagnostic={destination}") from error
        self.skipped = True
        return True


def batch_guard(trainer, logger, batch_dict, epoch):
    from contextlib import nullcontext
    return BatchJudgeGuard(trainer, logger, batch_dict, epoch) if os.environ.get("FOURWAY_CONFIG") else nullcontext()
