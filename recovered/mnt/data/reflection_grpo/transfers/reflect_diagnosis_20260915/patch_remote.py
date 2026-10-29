from pathlib import Path
import ast
b=Path('/root/reflect_branch_stable_20260915');repo=b/'code/reflection_grpo_clean'
p=repo/'rgrpo/vllm_lifecycle.py';p.write_text('"""Serialize CUDA work around vLLM virtual-memory lifecycle transitions."""\nimport torch\n\n\ndef synchronized_transition(operation, *args, **kwargs):\n    # empty_cache() alone is not a device-wide quiescence guarantee.\n    torch.cuda.synchronize()\n    result = operation(*args, **kwargs)\n    torch.cuda.synchronize()\n    return result\n')
for rel in ['verl/workers/sharding_manager/fsdp_vllm.py','verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py']:
 p=repo/rel;t=p.read_text()
 replacements={
  'self.inference_engine.sleep(level=1)':'synchronized_transition(self.inference_engine.sleep, level=1)',
  'self.inference_engine.wake_up(tags=["weights"])':'synchronized_transition(self.inference_engine.wake_up, tags=["weights"])',
  'self.inference_engine.wake_up(tags=["kv_cache"])':'synchronized_transition(self.inference_engine.wake_up, tags=["kv_cache"])',
  'self.inference_engine.wake_up()':'synchronized_transition(self.inference_engine.wake_up)',
 }
 changed=0
 for a,z in replacements.items():changed+=t.count(a);t=t.replace(a,z)
 assert changed>0
 # Import after the module's imports (before the first class declaration).
 idx=t.index('\nclass ');t=t[:idx]+'\nfrom rgrpo.vllm_lifecycle import synchronized_transition\n'+t[idx:]
 if rel.endswith('fsdp_vllm.py'):
  a='            self.update_params(params, peft_config=peft_config)';assert a in t;t=t.replace(a,a+'\n            get_torch_device().synchronize()',1)
 p.write_text(t);ast.parse(t);print(rel,changed)
p=repo/'fourway/configuration.py';t=p.read_text();a='"trainer.save_freq": -1 if probe else c["validation"]["frequency"]';assert a in t;p.write_text(t.replace(a,'"trainer.save_freq": -1 if probe else c.get("save_frequency", c["validation"]["frequency"])',1))
p=repo/'fourway/runtime.py';t=p.read_text();a='        if not summary.exists():\n            raise RuntimeError("Checkpoint has no complete validation; refusing best selection")';z='        if not summary.exists():\n            if step % config()["validation"]["frequency"] == 0:\n                raise RuntimeError("Checkpoint has no complete validation; refusing best selection")\n            # Frequent recoverable saves do not participate in best selection.\n            atomic_json(run / "latest.json", dict(step=step, validation_status="not_scheduled",\n                        checkpoint=str(run / f"global_step_{step}")))\n            best_path = run / "best.json"\n            keep = {step}\n            if best_path.exists():\n                keep.add(int(json.loads(best_path.read_text())["step"]))\n            if (run / "fourway_run.json").exists():\n                import shutil\n                for p in run.glob("global_step_*"):\n                    if p.is_dir() and p.name.removeprefix("global_step_").isdigit() and int(p.name.split("_")[-1]) not in keep:\n                        shutil.rmtree(p)\n            return';assert a in t;p.write_text(t.replace(a,z,1))
# Make no-update preflight robust to Ray shutting down before stdout forwarding.
p=repo/'fourway/cli.py';t=p.read_text();a='                    log = run_child(c, args, method, stage, probe=True)\n                    text = log.read_text()';z='                    import time\n                    probe_started = time.time()\n                    log = run_child(c, args, method, stage, probe=True)\n                    text = log.read_text()\n                    if "FOURWAY_NO_UPDATE_PROBE" not in text:\n                        proofs = []\n                        for worker_log in (Path(c["ray_tmp_dir"]) / "session_latest/logs").glob("worker-*.out"):\n                            if worker_log.stat().st_mtime >= probe_started:\n                                proofs += [line for line in worker_log.read_text(errors="replace").splitlines()\n                                           if line.startswith("FOURWAY_NO_UPDATE_PROBE ")]\n                        if proofs:\n                            evidence = "\\n".join(proofs) + "\\n"\n                            with log.open("a") as handle:\n                                handle.write(evidence)\n                            text += evidence';assert a in t;p.write_text(t.replace(a,z,1))
for p in [repo/'fourway/runtime.py',repo/'fourway/configuration.py',repo/'fourway/cli.py']:ast.parse(p.read_text())
print('patched')
