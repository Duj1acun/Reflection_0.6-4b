import ast
import json
from pathlib import Path
from types import SimpleNamespace
import pytest


def save_hook(root):
    from fourway.contracts import atomic_json, rank_key
    from fourway import runtime
    tree=ast.parse(Path(runtime.__file__).read_text())
    install=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="install")
    save=next(n for n in install.body if isinstance(n,ast.FunctionDef) and n.name=="save")
    def native(self):
        p=root/f"global_step_{self.global_steps}"
        (p/"actor").mkdir(parents=True)
        (p/"data.pt").write_text("test native state")
    ns=dict(Path=Path,json=json,atomic_json=atomic_json,rank_key=rank_key,old_save=native,config=lambda:{"validation":{"frequency":50}})
    exec(compile(ast.fix_missing_locations(ast.Module(body=[save],type_ignores=[])),"actual_save_hook","exec"),ns)
    return ns["save"]


def obj(root,step):
    return SimpleNamespace(global_steps=step,config=SimpleNamespace(trainer=SimpleNamespace(default_local_dir=str(root)),fourway=SimpleNamespace(method="reflect")))


def test_recovery_saves_do_not_require_or_invent_validation(tmp_path):
    (tmp_path/"fourway_run.json").write_text("{}")
    save=save_hook(tmp_path)
    save(obj(tmp_path,5));assert not (tmp_path/"best.json").exists()
    latest=json.loads((tmp_path/"latest.json").read_text())
    assert latest['validation_status']=='not_scheduled'
    save(obj(tmp_path,10));assert not (tmp_path/"global_step_5").exists()
    assert (tmp_path/"global_step_10/data.pt").exists()


def test_recovery_preserves_best_evaluated_checkpoint(tmp_path):
    (tmp_path/"fourway_run.json").write_text("{}")
    (tmp_path/"global_step_50").mkdir()
    (tmp_path/"best.json").write_text('{"step":50}')
    save=save_hook(tmp_path);save(obj(tmp_path,55));save(obj(tmp_path,60))
    assert (tmp_path/"global_step_50").exists()
    assert not (tmp_path/"global_step_55").exists()
    assert json.loads((tmp_path/"best.json").read_text())['step']==50


def test_scheduled_validation_cannot_be_silently_missing(tmp_path):
    with pytest.raises(RuntimeError,match="no complete validation"):
        save_hook(tmp_path)(obj(tmp_path,50))
