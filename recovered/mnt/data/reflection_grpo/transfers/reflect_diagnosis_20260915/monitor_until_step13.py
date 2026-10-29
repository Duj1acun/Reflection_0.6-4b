"""Read-only bounded observation; never changes or restarts a training job."""
from pathlib import Path
import datetime
import json
import subprocess
import time

root = Path(__file__).resolve().parent
deadline = time.monotonic() + 3 * 3600
previous = None
while time.monotonic() < deadline:
    result = subprocess.run(
        ["python3", str(root / "check_remote.py"), "--brief"],
        text=True, capture_output=True, timeout=40,
    )
    if result.returncode:
        print("MONITOR_CONNECTION_FAILURE " + result.stderr[-1200:], flush=True)
        raise SystemExit(3)
    report = json.loads((root / "latest_status.json").read_text())
    job = report["jobs"][0]
    train = job.get("train", {})
    step = train.get("step")
    current = (step, tuple(train.get("validation_progress", [])), job.get("stage"))
    if current != previous:
        print(result.stdout.rstrip(), flush=True)
        previous = current
        (root / "observations").mkdir(exist_ok=True)
        label = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        (root / "observations" / (label + ".json")).write_text(json.dumps(report, indent=2))
    else:
        print(json.dumps({"time": report["time"], "step": step,
                          "validation": train.get("validation_progress"),
                          "alive": job.get("process_stat"),
                          "traceback": train.get("traceback")}), flush=True)
    if job.get("stage") == "failed" or job.get("process_stat") in {"missing", "Z"} or train.get("traceback"):
        print("FORMAL_RUN_NEEDS_DIAGNOSIS", flush=True)
        raise SystemExit(2)
    if step is not None and step >= 13:
        (root / "step13_observation.json").write_text(json.dumps(report, indent=2))
        print("PASSED_PREVIOUS_FAILURE_STEP; verify recovery checkpoint files before closing.", flush=True)
        break
    time.sleep(45)
else:
    print("OBSERVATION_TIMEOUT; training left untouched.", flush=True)
    raise SystemExit(4)
