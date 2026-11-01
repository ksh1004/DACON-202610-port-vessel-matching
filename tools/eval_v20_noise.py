"""v20 패키지(script.py)를 주최 측 방식 노이즈 세트에 돌려 설정별 정확도를 비교한다.

평가 선박(노이즈 세트의 train 한글 선박)은 제외 목록에서 빼고 돌린다. 패키지 파일은 바꾸지 않는다.
KO_W=0, KO_K=0 은 v18 과 같은 동작이다. 순위 모델이 평가 선박을 학습했으므로 절대값은 낙관적이다.

    python -m tools.eval_v20_noise ../submit_v20 E:/dacon_tmp/noise 0:0 6:50 8:50
"""
import json
import sys
from pathlib import Path

import pandas as pd

pkg, work = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
sys.path.insert(0, str(pkg / "model"))
sys.path.insert(0, str(pkg))
import script  # noqa: E402

truth = pd.read_csv(work / "truth.csv", encoding="utf-8-sig")
vessels = json.loads((pkg / "model" / "train_vessels.json").read_text(encoding="utf-8"))
keep = sorted(set(vessels) - set(truth.target_vessel_id))
(work / "train_vessels_eval.json").write_text(json.dumps(keep), encoding="utf-8")
script.EXCLUDE_PATH = work / "train_vessels_eval.json"

_beams = script.LLMBeams()
script.LLMBeams = lambda: _beams  # 설정 사이에 음역 결과 재사용 (이름별 결과는 같음)

res = {}
for cfg in sys.argv[3:]:
    w, k = cfg.split(":")
    script.KO_W, script.KO_K = float(w), int(k)
    out = work / f"pred_{w}_{k}.csv"
    script.run(work / "data/test.csv", work / "data/vessel_catalog.csv", out)
    pred = pd.read_csv(out).rename(columns={"target_vessel_id": "pred"}).merge(truth, on="sample_id")
    ok = pred.pred == pred.target_vessel_id
    res[cfg] = {"acc": round(ok.mean(), 4), "nogt": round(ok[pred.nogt].mean(), 4), "gt": round(ok[~pred.nogt].mean(), 4),
                "correct": int(ok.sum())}
    print(cfg, res[cfg], flush=True)
base = None
for cfg in sys.argv[3:]:
    w, k = cfg.split(":")
    p = pd.read_csv(work / f"pred_{w}_{k}.csv").set_index("sample_id").target_vessel_id
    if base is None:
        base, base_cfg = p, cfg
        continue
    t = truth.set_index("sample_id").target_vessel_id
    print(f"{cfg} vs {base_cfg}: 회복 {int(((p == t) & (base != t)).sum())} / 손실 {int(((p != t) & (base == t)).sum())}")
