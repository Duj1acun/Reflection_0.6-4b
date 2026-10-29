from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import sys
import traceback

b=Path(__file__).resolve().parent
c=json.loads((b/'config.json').read_text())
out=Path(c['output']);repo=Path(c['reflection_repo'])
env=os.environ.copy()
env.update(PYTHONPATH=str(repo), PATH=str(Path(c['python']).parent)+':'+env.get('PATH',''),
           HF_HOME=c['hf_home'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',
           TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled')
status=dict(launcher_pid=os.getpid(),detached=True,gpu='0',method='score',steps=500,
            stage1_steps=250,save_frequency=50,val_before_train=False,output=str(out),
            config_sha256=hashlib.sha256((b/'config.json').read_bytes()).hexdigest())
def state(stage,**extra):
    status.update(stage=stage,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra)
    tmp=b/'SERVER_STATUS.json.tmp';tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(b/'SERVER_STATUS.json')
    print(stage,flush=True)
def run(args,log,child_env=None):
    with (out/log).open('w') as f:
        subprocess.run(args,cwd=repo,env=child_env or env,stdout=f,stderr=subprocess.STDOUT,check=True)
entry=repo/'experiments/four_methods/run_math7500.sh'
try:
    state('preparing_data')
    run(['bash',str(entry),'prepare'],'prepare.log')
    state('native_regression_tests')
    test_env=env.copy();test_env['PYTHONPATH']=c['pag_repo']+':'+str(repo)
    test_env.pop('FOURWAY_CONFIG',None)
    script="import sys; sys.path.insert(0, "+repr(c['pag_repo'])+"); import verl; assert 'Policy-As-GenVerifier' in verl.__file__; import pytest; raise SystemExit(pytest.main("+repr([
        str(repo/'tests/test_fourway_native_rollouts.py'),str(repo/'tests/test_vllm_lifecycle.py'),
        '-q','-k','not native_dataloader_uses_shared_batches'])+"))"
    run([c['python'],'-c',script],'regression_tests.log',test_env)
    state('gpu_no_update_preflight',regression_tests='passed')
    run(['bash',str(entry),'preflight','--gpu','--method','score'],'gpu_preflight.log')
    report=json.loads((out/'preparation/gpu_preflight.json').read_text())
    assert report['config_sha256']==status['config_sha256']
    assert all(report['methods'][f'score.stage{s}']['status']=='passed' for s in [1,2])
    for stage in [1,2]:
        log=(out/f'probes/score/stage{stage}/stage{stage}.log').read_text()
        assert log.count('FOURWAY_NO_UPDATE_PROBE')>=2, 'Actor and critic backward proofs both required'
    state('formal_training',gpu_preflight='both_stages_passed')
    run(['bash',str(entry),'train','--method','score'],'launcher.log')
    state('completed')
except BaseException as error:
    state('failed',error=repr(error));traceback.print_exc();sys.exit(1)
