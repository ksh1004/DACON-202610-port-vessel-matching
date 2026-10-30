"""학습 산출물과 src 코드를 모아 제출 패키지 폴더(submit/<name>/)를 조립한다.

    python tools/build_package.py v1_hybrid --hangul-mode baseline
    python tools/build_package.py v2_rank --hangul-mode ranker
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_MODULES = ["__init__.py", "common.py", "hangul.py", "retrieval.py", "features.py", "llm_translit.py", "llm_cache.py",
               "ensemble.py", "match_features.py", "matching_ranker.py", "neural_pair.py", "pair_evidence.py", "pair_views.py",
               "encoder_pair.py", "encoder_phonetics.py", "encoder_views.py"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--hangul-mode", choices=["ranker", "baseline"], required=True)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "experiments" / "artifacts" / "ranker_v1",
                        help="ranker.txt, mappers.joblib 이 있는 폴더")
    parser.add_argument("--llm-dir", type=Path, help="함께 넣을 LLM 가중치 폴더 (model/llm 으로 복사)")
    parser.add_argument("--callsign-lookup", action="store_true",
                        help="train.csv 의 call_sign → vessel_id 대응표를 넣어, test 행의 호출부호가 train 에 있으면 그 선박으로 예측")
    args = parser.parse_args()

    submit_root = (ROOT / "submit").resolve()
    package = (submit_root / args.name).resolve()
    if package.parent != submit_root:
        raise ValueError("Package name must be one directory directly under submit/")
    if package.exists():
        shutil.rmtree(package)
    model = package / "model"
    (model / "src").mkdir(parents=True)

    for name in SRC_MODULES:
        shutil.copy(ROOT / "src" / name, model / "src" / name)
    shutil.copy(args.artifacts / "mappers.joblib", model / "mappers.joblib")
    if (args.artifacts / "encoder_views.json").exists():
        spec = json.loads((args.artifacts / "encoder_views.json").read_text(encoding="utf-8"))
        base = json.loads((args.artifacts / "ensemble.json").read_text(encoding="utf-8"))
        for name in ["encoder_views.json", "ensemble.json"]+[m["file"] for m in base["members"]]:
            shutil.copy(args.artifacts / name, model / name)
        for view in spec["views"]:
            shutil.copytree(args.artifacts / view["dir"], model / view["dir"])
    elif (args.artifacts / "pair_views.json").exists():
        spec = json.loads((args.artifacts / "pair_views.json").read_text(encoding="utf-8"))
        base = json.loads((args.artifacts / "ensemble.json").read_text(encoding="utf-8"))
        for name in ["pair_views.json", "ensemble.json"] + [m["file"] for m in base["members"]]:
            shutil.copy(args.artifacts / name, model / name)
        for view in spec["views"]:
            shutil.copytree(args.artifacts / view["dir"], model / view["dir"])
    elif (args.artifacts / "neural.json").exists():
        spec = json.loads((args.artifacts / "neural.json").read_text(encoding="utf-8"))
        base = json.loads((args.artifacts / "ensemble.json").read_text(encoding="utf-8"))
        for name in ["neural.json", "ensemble.json"] + [m["file"] for m in base["members"]]:
            shutil.copy(args.artifacts / name, model / name)
        shutil.copytree(args.artifacts / spec["model_dir"], model / spec["model_dir"])
    elif (args.artifacts / "matching.json").exists():
        spec = json.loads((args.artifacts / "matching.json").read_text(encoding="utf-8"))
        for name in ("matching.json", spec["primary"], spec["second"]):
            shutil.copy(args.artifacts / name, model / name)
        if spec.get("base_ensemble"):
            base = json.loads((args.artifacts / "ensemble.json").read_text(encoding="utf-8"))
            for name in ["ensemble.json"] + [m["file"] for m in base["members"]]:
                shutil.copy(args.artifacts / name, model / name)
    elif (args.artifacts / "ensemble.json").exists():  # 앙상블: 설정 파일과 학습기 파일들
        spec = json.loads((args.artifacts / "ensemble.json").read_text(encoding="utf-8"))
        for name in ["ensemble.json"] + [m["file"] for m in spec["members"]]:
            shutil.copy(args.artifacts / name, model / name)
    else:
        shutil.copy(args.artifacts / "ranker.txt", model / "ranker.txt")
    if args.hangul_mode == "baseline":
        for name in ("rf.pkl", "baseline_lib.py"):
            shutil.copy(ROOT / "submit" / "baseline" / "model" / name, model / name)

    if args.callsign_lookup:
        import pandas as pd
        train = pd.read_csv(ROOT / "train.csv", encoding="utf-8-sig")
        pairs = train.drop_duplicates("call_sign")[["call_sign", "target_vessel_id"]]
        if train.groupby("call_sign")["target_vessel_id"].nunique().max() != 1:
            raise ValueError("call_sign 이 선박과 1:1 이 아님")
        (model / "callsign_map.json").write_text(json.dumps(dict(zip(pairs["call_sign"], pairs["target_vessel_id"]))),
                                                 encoding="utf-8")
    if args.llm_dir:  # 추론 서버는 인터넷이 차단되므로 가중치·토크나이저를 그대로 넣는다 (라이선스 파일 포함)
        shutil.copytree(args.llm_dir, model / "llm", ignore=shutil.ignore_patterns(".cache", "*.md"))

    script = (ROOT / "tools" / "script_template.py").read_text(encoding="utf-8")
    if not args.callsign_lookup:
        start_marker = "    # __CALLSIGN_LOOKUP_BEGIN__\n"
        end_marker = "    # __CALLSIGN_LOOKUP_END__\n"
        start = script.index(start_marker)
        end = script.index(end_marker, start)+len(end_marker)
        script = script[:start]+script[end:]
    (package / "script.py").write_text(script.replace("__HANGUL_MODE__", args.hangul_mode), encoding="utf-8")
    (package / "requirements.txt").write_text(
        "# 평가 서버에 기본 설치된 numpy, pandas, scikit-learn, joblib, lightgbm, catboost, rapidfuzz, torch, transformers 만 사용\n",
        encoding="utf-8",
    )
    print(f"패키지 조립: {package}")


if __name__ == "__main__":
    main()
