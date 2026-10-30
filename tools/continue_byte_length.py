"""Conditional length correction with the unchanged fresh/full gates."""
import json

from src.common import ROOT
from tools import byte_names_length_experiment as v
from tools.run_byte_names import stage


def main():
    p=v.check();path=ROOT/"experiments"/f"{v.TAG}_selection.json"
    decision=json.loads(path.read_text())
    if not decision["screen_qualified"]:return
    assert decision["stage"]=="screen"
    fold=p["fresh_guard"]["fold"]
    stage("length_guard_training","tools.byte_names_experiment","--folds",fold,"--train-only")
    stage("length_fresh","tools.fresh_byte_names_length")
    fresh=json.loads((ROOT/"experiments"/f"{v.TAG}_fresh.json").read_text())
    decision.update(stage="fresh",fresh=fresh)
    if fresh["delta_vs_v12"]>=p["fresh_guard"]["minimum_mean_accuracy_delta"]:
        for k in sorted(set(range(5))-set(p["screen_folds"])):v.run(k)
        from tools.audit_reverse_ranker import audit
        full=audit(v.FOLDER,v.PROTOCOL);r=full["contenders"]["residual30"]
        a,b,g=full["reference_v12"],r["candidate"],p["full_gate"]
        checks={"five_folds":full["complete_folds"]==list(range(5)),"testlike":b["testlike"]-a["testlike"]>=g["testlike_gain_min"],
                "hard_hangul":b["hard_hangul"]-a["hard_hangul"]>=g["hard_hangul_delta_min"],
                "drop_word":b["scenarios"]["drop_word"]-a["scenarios"]["drop_word"]>=g["drop_word_delta_min"],
                "oof":b["oof"]-a["oof"]>=g["oof_delta_min"],
                "positive_folds":sum(x>0 for x in r["delta_by_fold_vs_v12"].values())>=g["positive_folds_min"],
                "matched_control":r["delta_vs_matched_control"]>=g["gain_vs_matched_control_min"]}
        decision.update(stage="full",full=full,full_checks=checks)
        if all(checks.values()):decision["selected"]="residual30"
    path.write_text(json.dumps(decision,indent=2),encoding="utf-8")
    if decision["selected"]:stage("length_release","tools.release_byte_names_length")


if __name__=="__main__":main()
