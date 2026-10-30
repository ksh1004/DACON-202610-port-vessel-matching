"""Wait for the primary decision; never search past a qualified primary."""
import json
import time

from src.common import ROOT
from tools.run_byte_names import stage


def main():
    from tools import byte_names_length_experiment as v
    v.check()
    lock={p:v.base.sha(ROOT/p) for p in ["tools/run_conditional_byte_length.py","tools/continue_byte_length.py","tools/fresh_byte_names_length.py"]}
    path=ROOT/"experiments/byte_names_length_v1_execution_lock.json"
    if path.exists():assert json.loads(path.read_text())==lock
    else:path.write_text(json.dumps(lock,indent=2),encoding="utf-8")
    primary=ROOT/"experiments/byte_names_v1_selection.json"
    while not primary.exists():time.sleep(10)
    d=json.loads(primary.read_text())
    if d["stage"]!="screen" or d["selected"] is not None:
        (ROOT/"experiments/byte_names_length_v1_not_run.json").write_text(json.dumps({"reason":"Primary qualified its screen; no later recipe after fresh/full", "primary_stage":d["stage"]},indent=2),encoding="utf-8")
        return
    stage("length_screen","tools.byte_names_length_experiment")
    stage("length_continue","tools.continue_byte_length")
    print("Conditional length trial completed",flush=True)


if __name__=="__main__":main()
