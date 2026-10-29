from pathlib import Path
import json,subprocess
b=Path('/root/reflect_main_sftprompt_20260914');c=json.loads((b/'config.json').read_text());o=Path(c['output']);assert json.loads((b/'SERVER_STATUS.json').read_text())['stage']=='failed'
p=b/'code/reflection_grpo_clean/fourway/cli.py';t=p.read_text()
a='                    log = run_child(c, args, method, stage, probe=True)\n                    text = log.read_text()'
bb='                    import time\n                    probe_started = time.time()\n                    log = run_child(c, args, method, stage, probe=True)\n                    text = log.read_text()\n                    # Ray can shut down before the final worker stdout is forwarded.\n                    # Accept only direct worker logs created/updated in this probe.\n                    if "FOURWAY_NO_UPDATE_PROBE" not in text:\n                        worker_dir = Path(c["ray_tmp_dir"]) / "session_latest" / "logs"\n                        proofs = []\n                        for worker_log in worker_dir.glob("worker-*.out"):\n                            if worker_log.stat().st_mtime < probe_started:\n                                continue\n                            for line in worker_log.read_text(errors="replace").splitlines():\n                                if line.startswith("FOURWAY_NO_UPDATE_PROBE "):\n                                    proofs.append(f"{line} [source: {worker_log.resolve()}]")\n                        if proofs:\n                            evidence = "\\n".join(proofs) + "\\n"\n                            with log.open("a") as handle:\n                                handle.write(evidence)\n                            text += evidence'
assert a in t;p.write_text(t.replace(a,bb,1))
for f in ['restart.log','gpu_preflight.log']:(o/f).rename(o/(f+'.attempt2'))
with (o/'restart.log').open('w') as f:
 p=subprocess.Popen(['/usr/bin/python3','-u',str(b/'restart_and_train.py')],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
(b/'restart.pid').write_text(str(p.pid)+'\n');(o/'launcher.pid').write_text(str(p.pid)+'\n');print('main restart',p.pid)
