from pathlib import Path
import json, shutil, hashlib, datetime
files = {'rgrpo/reflection_format.py': '"""SFT-compatible reflection markup validation, without a fixed English phrase."""\nimport re\n\n_MARKUP = re.compile(r"<\\s*/?\\s*reflection\\b[^>]*(?:>|$)", re.IGNORECASE)\n\n\ndef reflection_format_valid(text):\n    """Require nonempty, balanced, nonnested <reflection>...</reflection> blocks."""\n    if not isinstance(text, str):\n        return False\n    opened = None\n    count = 0\n    for match in _MARKUP.finditer(text):\n        tag = match.group().lower()\n        if tag == "<reflection>":\n            if opened is not None:\n                return False\n            opened = match.end()\n        elif tag == "</reflection>":\n            if opened is None or not text[opened:match.start()].strip():\n                return False\n            opened = None\n            count += 1\n        else:\n            return False\n    return count > 0 and opened is None\n', 'tests/test_sft_reflection_format.py': 'import pytest\n\nfrom rgrpo.reflection_format import reflection_format_valid\n\n\n@pytest.mark.parametrize("text", [\n    "<reflection>The arithmetic is incorrect. Recompute the product.</reflection>",\n    "<reflection>\\n已有结论。错误原因。新的方向。\\n</reflection>",\n    "<reflection>So far I have a result. However it is wrong.</reflection>",\n    "prefix <reflection>First correction.</reflection> next <reflection>Second correction.</reflection> answer",\n])\ndef test_accepts_content_without_fixed_phrase(text):\n    assert reflection_format_valid(text)\n\n\n@pytest.mark.parametrize("text", [\n    None, "", "direct answer", "<reflection></reflection>",\n    "<reflection> \\n\\t </reflection>", "<reflection>unclosed",\n    "orphan</reflection>", "</reflection><reflection>reversed",\n    "<reflection>outer<reflection>inner</reflection></reflection>",\n    "<reflection>valid</reflection><reflection>unclosed",\n    "<reflection>valid</reflection></reflection>",\n    "<reflection>valid</reflection><reflection> </reflection>",\n    "<reflection extra=\'x\'>bad</reflection>",\n    "<reflection>valid</reflection><reflection",\n])\ndef test_rejects_empty_or_malformed_blocks(text):\n    assert not reflection_format_valid(text)\n\n\ndef test_reward_preserves_correct_reflection_credit(monkeypatch):\n    from rgrpo import core_trajectory_grpo as reward\n    for name in list(reward.os.environ):\n        if name.startswith("TRAJ_GRPO_"):\n            monkeypatch.delenv(name)\n    monkeypatch.setenv("TRAJ_GRPO_REQUIRE_REFLECTION_FORMAT", "1")\n    monkeypatch.setenv("TRAJ_GRPO_REWARD_PHI", "4")\n    monkeypatch.setenv("TRAJ_GRPO_EFF_FREE_REFLECTIONS", "1")\n    monkeypatch.setattr(reward, "compute_math_score", lambda **kwargs: 1.0)\n    valid = reward.compute_trajectory_grpo_score(\n        generated="<reflection>The sign is wrong. Correct it.</reflection>\\\\boxed{2}",\n        ground_truth="2", extra_info={"num_reflections": 1, "n_iters": 2})\n    assert valid["reflection_format_valid"] == 1.0\n    assert valid["score"] == 1.0\n    malformed = reward.compute_trajectory_grpo_score(\n        generated="<reflection> </reflection>\\\\boxed{2}", ground_truth="2")\n    assert malformed["format_reward"] == 0.0\n    assert malformed["score"] == 0.0\n    direct = reward.compute_trajectory_grpo_score(generated="\\\\boxed{2}", ground_truth="2")\n    assert direct["score"] == 1.0\n'}
pairs=[('reflect_math7500_0.6b_20260914','reflect_main_tagfmt_20260914'),('reflect_branch_gpu1_20260914','reflect_branch_tagfmt_20260914')]
records=[]
for oldname,newname in pairs:
 old=Path('/root')/oldname; new=Path('/root')/newname
 assert not new.exists(),str(new)
 new.mkdir()
 c=json.loads((old/'config.json').read_text());oldrepo=Path(c['reflection_repo']);repo=new/'code/reflection_grpo_clean'
 shutil.copytree(oldrepo,repo,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache','.git'))
 shutil.copy2(old/'run_reflect.py',new/'run_reflect.py')
 for name,text in files.items():(repo/name).write_text(text)
 p=repo/'rgrpo/core_trajectory_grpo.py';text=p.read_text();assert '    _reflection_format_valid,\n' in text
 text=text.replace('    _reflection_format_valid,\n','',1)
 anchor='    compute_math_score,\n)'
 assert anchor in text;text=text.replace(anchor,anchor+'\nfrom .reflection_format import reflection_format_valid as _reflection_format_valid',1)
 text=text.replace("reflection markup to use the Hybrid-v2 format:\n      '<reflection>So far I have ... However ...</reflection>'.","reflection markup to contain balanced, nonnested, nonempty\n      '<reflection>...</reflection>' blocks, matching the SFT targets.")
 p.write_text(text)
 p=repo/'fourway/contracts.py';text=p.read_text()
 a='    "use <reflection>So far I have ... However ...</reflection> to summarize the "\n    "reasoning so far and identify the mistake, then continue reasoning based on "\n    "the reflection. Put your final answer within \\boxed{}."'
 b='    "use <reflection>...</reflection> to summarize the valid reasoning so far, "\n    "identify the mistake, and describe how to correct it. Then continue reasoning "\n    "based on the reflection. Put your final answer within \\boxed{}."'
 assert a in text; p.write_text(text.replace(a,b,1))
 oldoutput=Path(c['output']);c['experiment']+='_tagfmt_v2';c['output']+='_tagfmt_v2';c['reflection_repo']=str(repo)
 c['ray_tmp_dir']='/tmp/rf-'+hashlib.sha256(c['output'].encode()).hexdigest()[:10]
 c['reflection_format_version']='sft-tags-v1-20260914'
 out=Path(c['output']);assert not out.exists();(out/'initialization').mkdir(parents=True)
 (out/'initialization/reflect_merged').symlink_to((oldoutput/'initialization/reflect_merged').resolve())
 (new/'config.json').write_text(json.dumps(c,indent=2)+'\n');(new/'results').symlink_to(out)
 report=dict(old_bundle=str(old),old_output=str(oldoutput),new_bundle=str(new),output=str(out),gpu=c['gpus'],format_version=c['reflection_format_version'],initialization=str((out/'initialization/reflect_merged').resolve()),restart_from_sft=True)
 (new/'restart_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
 (new/'CODE_SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(new))+'\n' for p in sorted(repo.rglob('*')) if p.is_file()))
 records.append(report)
print(json.dumps(records,indent=2))
