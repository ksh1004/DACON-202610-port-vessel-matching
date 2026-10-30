"""Fixed prospective family order; freeze first qualifying two-fold candidate."""
import json
import os
import subprocess
import sys

from src.common import ROOT


def main():
    protocol=json.loads((ROOT/"experiments/nested_family_protocol_20261029.json").read_text())
    state_path=ROOT/"experiments/nested_family_selection_20261029.json"
    assert not state_path.exists(), "Selection already recorded; inspect it instead of restarting search"
    state={"selected_family":None,"selected_contender":None,"families":{},"actual_competition_submission":False}
    modules={"nested_raw78_fallback_v1":"tools.audit_nested_raw_fallback",
             "nested_combiner_reverse_v1":"tools.nested_reverse_extension",
             "nested_combiner_token_prior_v1":"tools.nested_token_prior_extension"}
    for tag in protocol["family_order"]:
        report_path=ROOT/"experiments"/f"{tag}_audit.json"
        if not report_path.exists():
            assert tag in modules
            with (ROOT/"experiments"/f"{tag}_run.log").open("w",encoding="utf-8") as log:
                subprocess.run([sys.executable,"-m",modules[tag]],stdout=log,stderr=subprocess.STDOUT,
                               env=dict(os.environ,PYTHONUTF8="1",PYTHONIOENCODING="utf-8"),check=True)
        report=json.loads(report_path.read_text())
        assert report["complete_folds"]==[0,2]
        snapshot=ROOT/"experiments"/f"{tag}_screen_locked.json"
        if not snapshot.exists():snapshot.write_text(json.dumps(report,indent=2),encoding="utf-8")
        state["families"][tag]={"selected":report["selected"],"report":str(report_path.relative_to(ROOT)),
                                 "delta_by_contender":{m:r["delta_vs_v12"] for m,r in report["contenders"].items()}}
        if report["selected"] is not None:
            state["selected_family"],state["selected_contender"]=tag,report["selected"]
            state["next_stage"]="strict fresh fold3 guard, then full five-fold; no later family after guard/full rejection"
            break
    if state["selected_family"] is None:state["next_stage"]="All screen families rejected; no new final model or ZIP"
    state_path.write_text(json.dumps(state,indent=2),encoding="utf-8")
    print(json.dumps(state,indent=2),flush=True)


if __name__=="__main__":main()
