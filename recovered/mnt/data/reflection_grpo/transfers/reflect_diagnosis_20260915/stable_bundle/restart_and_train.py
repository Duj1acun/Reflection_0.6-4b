from pathlib import Path
import json,os,subprocess,sys,datetime,hashlib,traceback
b=Path(__file__).resolve().parent;c=json.loads((b/'config.json').read_text());out=Path(c['output']);repo=c['reflection_repo']
env=os.environ.copy();env.update(PYTHONPATH=repo,PATH=str(Path(c['python']).parent)+':'+env.get('PATH',''),HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',FOURWAY_CONFIG=str(b/'config.json'))
status=dict(launcher_pid=os.getpid(),detached=True,gpu=c['gpus'],output=str(out),config_sha256=hashlib.sha256((b/'config.json').read_bytes()).hexdigest(),format_gate=False,restart_from_sft=True)
def state(stage,**extra):
 status.update(stage=stage,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra)
 (b/'SERVER_STATUS.json').write_text(json.dumps(status,indent=2)+'\n');print(stage,flush=True)
def run(args,log):
 with (out/log).open('w') as f:
  child_env=env.copy()
  if 'pytest' in args: child_env.pop('FOURWAY_CONFIG',None)
  subprocess.run(args,cwd=repo,env=child_env,stdout=f,stderr=subprocess.STDOUT,check=True)
try:
 state('preparing_data')
 run([c['python'],'-u','-m','fourway.cli','prepare','--config',str(b/'config.json')],'prepare.log')
 state('regression_tests')
 tests=['tests/test_reflect_sft_contract.py','tests/test_trajectory_context.py','tests/test_vllm_lifecycle.py','tests/test_recovery_checkpoints.py']
 if c.get('reflection_branching',{}).get('enable'):tests+=['tests/test_branch_grpo.py']
 run([c['python'],'-m','pytest','-q',*tests,'-k','not cuda'],'regression_tests.log')
 state('gpu_no_update_preflight',regression_tests='passed')
 run(['python3',str(b/'run_reflect.py'),'preflight','--gpu'],'gpu_preflight.log')
 state('formal_training',gpu_preflight='passed',formal_training_started=True)
 if c.get('reflection_branching',{}).get('enable'):
  cmd=['bash',str(Path(repo)/'training/rl/train_reflect_branch_grpo.sh')]
 else:cmd=['python3',str(b/'run_reflect.py'),'train']
 run(cmd,'launcher.log')
 state('completed',formal_training_completed=True)
except BaseException as error:
 state('failed',error=repr(error));traceback.print_exc();sys.exit(1)
