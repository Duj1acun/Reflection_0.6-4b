import json
from pathlib import Path
from probe_teacher_v1 import save, digest
from probe_teacher_paratera_ten import SYSTEM, request_body
from prepare_sft_data import _teacher_prompt

root = Path("/mnt/data/reflection_grpo/data/sft")
old = root / "teacher_legacy_probe4_m3_t03_20260910"
for row in json.loads((old / "samples.json").read_text()):
    assert request_body(row)["messages"] == json.loads((old / "requests.json").read_text())[row["probe_id"]]["messages"]
prior = root / "teacher_v2_single_probe20_m3_t03_20260910"
manifest = json.loads((prior / "manifest.json").read_text())
assert digest(prior / "samples.json") == manifest["samples_sha256"]
samples = [r for r in json.loads((prior / "samples.json").read_text()) if int(r["probe_id"].rsplit("_", 1)[1]) <= (1 if r["correct_out_of_four"] == 3 else 2)]
assert len(samples) == 10
out = root / "teacher_legacy_probe10_reference_20260910"
out.mkdir(exist_ok=False)
save(out / "samples.json", samples)
save(out / "manifest.json", {"samples_sha256": digest(out / "samples.json"), "sources": manifest["sources"], "selection": "First two k1, first two k2, first one k3 per model from frozen 20-record probe; selected without reading teacher outputs."})
print("Exact prompt verified against user-designated saved requests; ten complete wrong samples prepared.")
