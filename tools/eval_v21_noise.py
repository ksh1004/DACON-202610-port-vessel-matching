"""v21 패키지(script.py)를 노이즈 세트에 돌려 설정별 정확도를 비교한다 (eval_v20_noise 의 일반화).

설정은 script 모듈 상수를 `이름=값` 으로 바꾼다 (쉼표로 여러 개). 평가 선박은 제외 목록에서 뺀다.

    python -m tools.eval_v21_noise E:/dacon_tmp/v21pkg E:/dacon_tmp/noise_en EN_K=0 EN_K=50 EN_K=200
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

if truth.vessel_name.map(script.is_hangul).any():
    _beams = script.LLMBeams()
    script.LLMBeams = lambda: _beams  # 설정 사이에 음역 결과 재사용
else:
    script.LLM_DIR = work / "no_llm"  # 영문 세트: LLM 음역을 쓰지 않는다

t = truth.set_index("sample_id").target_vessel_id
preds = {}
for cfg in sys.argv[3:]:
    for kv in cfg.split(","):
        k, v = kv.split("=")
        setattr(script, k, type(getattr(script, k))(v))
    out = work / f"pred_{cfg.replace(',', '_').replace('=', '')}.csv"
    script.run(work / "data/test.csv", work / "data/vessel_catalog.csv", out)
    p = pd.read_csv(out).set_index("sample_id").target_vessel_id
    ok = (p == t.reindex(p.index)).to_numpy()
    ng = truth.set_index("sample_id").nogt.reindex(p.index).to_numpy()
    preds[cfg] = p
    print(cfg, {"acc": round(ok.mean(), 4), "nogt": round(ok[ng].mean(), 4), "gt": round(ok[~ng].mean(), 4),
                "correct": int(ok.sum())}, flush=True)
base_cfg = sys.argv[3]
for cfg in sys.argv[4:]:
    a, b = preds[base_cfg], preds[cfg]
    tt = t.reindex(a.index)
    print(f"{cfg} vs {base_cfg}: 회복 {int(((b == tt) & (a != tt)).sum())} / 손실 {int(((b != tt) & (a == tt)).sum())}")
