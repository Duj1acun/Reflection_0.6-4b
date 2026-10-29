from pathlib import Path
import ast,json,shutil,hashlib
prompt = 'You are a math reasoning assistant. When you make an error during reasoning, use <reflection>...</reflection> to reflect on the mistake, then continue reasoning based on the reflection.'
regression_test = '"""Regression tests for the SFT prompt and disabled RL format reward gate."""\nimport ast\nfrom pathlib import Path\n\nimport pytest\n\n\ndef test_prompt_matches_both_sft_stages():\n    from fourway.contracts import REFLECT_PROMPT\n    root = Path(__file__).resolve().parents[1]\n    for stage in (1, 2):\n        tree = ast.parse((root / f"training/sft/train_phase{stage}.py").read_text())\n        node = next(n for n in tree.body if isinstance(n, ast.Assign)\n                    and any(isinstance(t, ast.Name) and t.id == "SYSTEM_PROMPT" for t in n.targets))\n        assert REFLECT_PROMPT == ast.literal_eval(node.value)\n\n\n@pytest.mark.parametrize("body", [\n    "", "<reflection>The sign is wrong. Recompute it.</reflection>",\n    "<reflection>So far I have a result. However it is wrong.</reflection>",\n    "<reflection> </reflection>", "<reflection>unclosed",\n])\n@pytest.mark.parametrize("correct", [False, True])\ndef test_disabled_gate_does_not_change_outcome_credit(monkeypatch, body, correct):\n    from fourway.configuration import REFLECT_ENV\n    from rgrpo import core_trajectory_grpo as reward\n    for name in list(reward.os.environ):\n        if name.startswith("TRAJ_GRPO_"):\n            monkeypatch.delenv(name)\n    for name, value in REFLECT_ENV.items():\n        monkeypatch.setenv(name, value)\n    assert REFLECT_ENV["TRAJ_GRPO_REQUIRE_REFLECTION_FORMAT"] == "0"\n    monkeypatch.setattr(reward, "_reflection_format_valid",\n                        lambda text: pytest.fail("Disabled format gate was called"))\n    monkeypatch.setattr(reward, "compute_math_score", lambda **kwargs: float(correct))\n    result = reward.compute_trajectory_grpo_score(\n        generated=body + "\\\\boxed{2}", ground_truth="2",\n        extra_info={"num_reflections": int(bool(body)), "n_iters": 2 if body else 1})\n    assert result["format_reward"] == 1.0\n    assert result["format_gate_zero"] == 0.0\n    assert result["score"] == (1.0 if correct else 0.0 if body else -0.25)\n'
pairs=[('reflect_math7500_0.6b_20260914','reflect_main_sftprompt_20260914'),('reflect_branch_gpu1_20260914','reflect_branch_sftprompt_20260914')]
records=[]
for oldname,newname in pairs:
 old=Path('/root')/oldname;new=Path('/root')/newname
 assert not new.exists(),str(new)
 new.mkdir();c=json.loads((old/'config.json').read_text());repo=new/'code/reflection_grpo_clean'
 shutil.copytree(c['reflection_repo'],repo,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache','.git'))
 shutil.copy2(old/'run_reflect.py',new/'run_reflect.py')
 (repo/'tests/test_reflect_sft_contract.py').write_text(regression_test)
 p=repo/'fourway/contracts.py';text=p.read_text();lines=text.splitlines(keepends=True)
 node=next(n for n in ast.parse(text).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REFLECT_PROMPT' for t in n.targets))
 lines[node.lineno-1:node.end_lineno]=['REFLECT_PROMPT = '+repr(prompt)+'\n'];p.write_text(''.join(lines))
 p=repo/'fourway/configuration.py';text=p.read_text();a='"TRAJ_GRPO_REQUIRE_REFLECTION_FORMAT": "1"';assert a in text;p.write_text(text.replace(a,'"TRAJ_GRPO_REQUIRE_REFLECTION_FORMAT": "0"',1))
 p=repo/'training/rl/train_reflect_plus.sh';text=p.read_text();a='${TRAJ_GRPO_REQUIRE_REFLECTION_FORMAT:-1}';assert a in text;p.write_text(text.replace(a,'${TRAJ_GRPO_REQUIRE_REFLECTION_FORMAT:-0}',1))
 oldoutput=Path(c['output']);c['experiment']+='_sftprompt_nogate';c['output']+='_sftprompt_nogate';c['reflection_repo']=str(repo)
 c['ray_tmp_dir']='/tmp/rf-'+hashlib.sha256(c['output'].encode()).hexdigest()[:10]
 c['reflection_format_version']='disabled-sft-prompt-20260914'
 out=Path(c['output']);assert not out.exists();(out/'initialization').mkdir(parents=True)
 (out/'initialization/reflect_merged').symlink_to((oldoutput/'initialization/reflect_merged').resolve())
 (new/'config.json').write_text(json.dumps(c,indent=2)+'\n');(new/'results').symlink_to(out)
 report=dict(old_bundle=str(old),old_output=str(oldoutput),new_bundle=str(new),output=str(out),gpu=c['gpus'],format_gate=False,prompt=prompt,initialization=str((out/'initialization/reflect_merged').resolve()),restart_from_sft=True)
 (new/'restart_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
 (new/'CODE_SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(new))+'\n' for p in sorted(repo.rglob('*')) if p.is_file()))
 records.append(report)
print(json.dumps(records,indent=2))
