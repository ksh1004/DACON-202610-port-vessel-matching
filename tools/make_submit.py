"""제출 패키지(submit/<name>/)를 submit.zip으로 묶고, 평가 서버와 같은 구조에서 검증한다.

사용법:
    python tools/make_submit.py submit/baseline
    python tools/make_submit.py submit/baseline --test-csv experiments/holdout.csv --n 0

검증 항목:
    1. 패키지 구조 (최상위에 script.py, requirements.txt, model/)
    2. requirements.txt 에 서버 기본 설치 패키지가 들어있는지 경고
    3. zip 을 임시 폴더에 풀고 data/test.csv, data/vessel_catalog.csv 를 넣어 script.py 실행
    4. output/submission.csv 형식 (컬럼, 행 수, sample_id 일치, 카탈로그 ID 여부)
    5. 행 독립성 (셔플한 부분집합, 단독 행으로 다시 실행해도 같은 예측인지)
    6. 실행 시간 (서버 제한 30분), zip 용량
    7. 정답이 있으면 로컬 가중 정확도 (0.7 * 한글 + 0.3 * 비한글)
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TIME_LIMIT_SEC = 30 * 60
SERVER_PACKAGES = {
    "torch", "pandas", "numpy", "scipy", "scikit-learn", "sklearn", "joblib",
    "threadpoolctl", "lightgbm", "catboost", "rapidfuzz", "unidecode",
    "transformers", "accelerate", "safetensors", "sentencepiece", "tqdm",
}
EXCLUDE_PARTS = {"__pycache__", ".ipynb_checkpoints", ".DS_Store"}
TEST_COLUMNS = [
    "sample_id", "vessel_name", "call_sign", "gross_tonnage",
    "international_gross_tonnage", "flag_name", "vessel_type", "arrival_year",
]


class CheckFailed(Exception):
    pass


def log(status: str, message: str) -> None:
    print(f"[{status:4s}] {message}")


def check_package(package: Path) -> list[Path]:
    if not package.is_dir():
        raise CheckFailed(f"패키지 폴더가 없음: {package}")
    for required in ("script.py", "requirements.txt"):
        if not (package / required).is_file():
            raise CheckFailed(f"최상위에 {required} 가 없음")
    if not (package / "model").is_dir():
        raise CheckFailed("최상위에 model/ 폴더가 없음")
    for reserved in ("data", "output"):
        if (package / reserved).exists():
            raise CheckFailed(f"{reserved}/ 는 평가 서버가 만드는 경로이므로 패키지에서 제외해야 함")

    files = [
        p for p in sorted(package.rglob("*"))
        if p.is_file() and not EXCLUDE_PARTS.intersection(p.parts) and p.suffix != ".pyc"
    ]
    log("OK", f"구조 확인: 파일 {len(files)}개")

    for line in (package / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        name = re.split(r"[\s<>=!~\[;]", line, maxsplit=1)[0].lower().replace("_", "-")
        if name in SERVER_PACKAGES:
            log("WARN", f"requirements.txt 의 '{line}' 은 서버 기본 설치 패키지 (버전 충돌 위험)")
    return files


def build_zip(package: Path, files: list[Path], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            # 신경망 가중치는 압축이 거의 안 되므로 저장만 한다 (zip 생성 시간 단축)
            method = zipfile.ZIP_STORED if path.suffix in (".safetensors", ".bin") else zipfile.ZIP_DEFLATED
            zf.write(path, path.relative_to(package).as_posix(), compress_type=method)
    with zipfile.ZipFile(out) as zf:
        top = {name.split("/", 1)[0] for name in zf.namelist()}
    if not {"script.py", "requirements.txt", "model"}.issubset(top):
        raise CheckFailed(f"zip 최상위 구성이 잘못됨: {sorted(top)}")
    size_mb = out.stat().st_size / 2**20
    shown = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    log("OK", f"zip 생성: {shown} ({size_mb:.2f} MB)")
    if size_mb > 10 * 1024:
        raise CheckFailed("zip 이 서버 제한 10GB 를 넘음")


def make_test(args: argparse.Namespace) -> pd.DataFrame:
    """정답 컬럼(target_vessel_id)이 남아 있는 가짜 test 를 만든다. 실행 전에 정답은 떼어낸다."""
    if args.test_csv:
        test = pd.read_csv(args.test_csv, encoding="utf-8-sig")
        source = args.test_csv
    else:
        test = pd.read_csv(ROOT / "train.csv", encoding="utf-8-sig")
        source = "train.csv"
    if args.n and args.n < len(test):
        test = test.sample(args.n, random_state=args.seed)
    test = test.reset_index(drop=True)
    log("INFO", f"가짜 test: {source} 에서 {len(test)}행")
    return test


def run_script(workdir: Path, test: pd.DataFrame, catalog_path: Path, timeout: int) -> tuple[pd.DataFrame, float]:
    """서버 구조(data/ 읽기용, output/ 결과)로 script.py 를 실행한다."""
    data = workdir / "data"
    output = workdir / "output"
    resolved_workdir = workdir.resolve()
    if data.resolve().parent != resolved_workdir or output.resolve().parent != resolved_workdir:
        raise CheckFailed("Temporary data/output directories must stay inside the execution workspace")
    shutil.rmtree(data, ignore_errors=True)
    shutil.rmtree(output, ignore_errors=True)
    data.mkdir()
    test[TEST_COLUMNS].to_csv(data / "test.csv", index=False, encoding="utf-8")
    shutil.copy(catalog_path, data / "vessel_catalog.csv")

    start = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "script.py"], cwd=workdir, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout,
    )
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:])
        raise CheckFailed(f"script.py 실행 실패 (exit {proc.returncode}) - 서버에서는 제출 횟수가 차감됨")
    result_path = output / "submission.csv"
    if not result_path.is_file():
        raise CheckFailed("output/submission.csv 가 생성되지 않음")
    return pd.read_csv(result_path, encoding="utf-8-sig", dtype=str), elapsed


def check_output(result: pd.DataFrame, test: pd.DataFrame, catalog_ids: set[str]) -> None:
    if list(result.columns) != ["sample_id", "target_vessel_id"]:
        raise CheckFailed(f"컬럼이 다름: {list(result.columns)}")
    if len(result) != len(test):
        raise CheckFailed(f"행 수가 다름: {len(result)} != {len(test)}")
    if result.isna().any().any():
        raise CheckFailed("결측값이 있음")
    if not result["sample_id"].is_unique:
        raise CheckFailed("sample_id 중복")
    if set(result["sample_id"]) != set(test["sample_id"].astype(str)):
        raise CheckFailed("sample_id 집합이 test 와 다름")
    unknown = set(result["target_vessel_id"]) - catalog_ids
    if unknown:
        raise CheckFailed(f"카탈로그에 없는 ID {len(unknown)}개: {sorted(unknown)[:5]}")


def is_hangul(value: object) -> bool:
    return isinstance(value, str) and any("가" <= ch <= "힣" for ch in value)


def score(test: pd.DataFrame, result: pd.DataFrame) -> None:
    merged = test.assign(sample_id=test["sample_id"].astype(str)).merge(
        result, on="sample_id", suffixes=("", "_pred")
    )
    correct = (merged["target_vessel_id"] == merged["target_vessel_id_pred"]).to_numpy()
    hangul = merged["vessel_name"].map(is_hangul).to_numpy(dtype=bool)
    if not hangul.any() or hangul.all():
        log("INFO", f"정확도 {correct.mean():.4f} (한글/비한글 중 한쪽만 있어 가중 점수 생략)")
        return
    h, n = correct[hangul].mean(), correct[~hangul].mean()
    log("INFO", f"로컬 점수 {0.7 * h + 0.3 * n:.4f}  (한글 {h:.4f} / {hangul.sum()}행, "
                f"비한글 {n:.4f} / {(~hangul).sum()}행)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("package", type=Path, help="제출 패키지 폴더 (예: submit/baseline)")
    parser.add_argument("--out", type=Path, help="zip 경로 (기본: dist/<패키지명>.zip)")
    parser.add_argument("--test-csv", type=Path, help="가짜 test 로 쓸 CSV (target_vessel_id 가 있으면 채점)")
    parser.add_argument("--n", type=int, default=3000, help="가짜 test 행 수 (0 = 전체)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset", type=int, default=300, help="행 독립성 검사용 부분집합 크기")
    parser.add_argument("--singles", type=int, default=3, help="단독 행 실행 검사 횟수")
    parser.add_argument("--skip-run", action="store_true", help="zip 만 만들고 실행 검증은 생략")
    args = parser.parse_args()

    package = args.package.resolve()
    out = (args.out or ROOT / "dist" / f"{package.name}.zip").resolve()
    print(f"== 패키지: {package}")
    try:
        files = check_package(package)
        build_zip(package, files, out)
        if args.skip_run:
            return 0

        catalog_path = ROOT / "vessel_catalog.csv"
        catalog_ids = set(pd.read_csv(catalog_path, encoding="utf-8-sig", usecols=["vessel_id"])["vessel_id"].astype(str))
        test = make_test(args)
        rng = np.random.default_rng(args.seed)

        with tempfile.TemporaryDirectory(prefix="dacon_sim_") as tmp:
            workdir = Path(tmp)
            with zipfile.ZipFile(out) as zf:
                zf.extractall(workdir)

            result, elapsed = run_script(workdir, test, catalog_path, TIME_LIMIT_SEC)
            check_output(result, test, catalog_ids)
            per_row = elapsed / len(test)
            log("OK", f"형식 확인, 실행 {elapsed:.1f}초 ({per_row * 1000:.2f} ms/행)")
            if elapsed > TIME_LIMIT_SEC * 0.5:
                log("WARN", "실행 시간이 서버 제한(30분)의 절반을 넘음")
            if "target_vessel_id" in test.columns:
                score(test, result)

            full = dict(zip(result["sample_id"], result["target_vessel_id"]))
            checks: list[pd.DataFrame] = []
            if args.subset:
                pick = rng.choice(len(test), size=min(args.subset, len(test)), replace=False)
                checks.append(test.iloc[pick])  # 순서도 섞인 부분집합
            for i in rng.choice(len(test), size=min(args.singles, len(test)), replace=False):
                checks.append(test.iloc[[i]])
            for part in checks:
                sub_result, _ = run_script(workdir, part, catalog_path, TIME_LIMIT_SEC)
                check_output(sub_result, part, catalog_ids)
                diff = [s for s, p in zip(sub_result["sample_id"], sub_result["target_vessel_id"]) if full[s] != p]
                if diff:
                    raise CheckFailed(f"행 독립성 위반: {len(part)}행 실행에서 {len(diff)}개 예측이 달라짐 {diff[:5]}")
            log("OK", f"행 독립성 확인: 셔플 부분집합 {args.subset}행 + 단독 행 {args.singles}회")
    except CheckFailed as error:
        log("FAIL", str(error))
        return 1
    print(f"== 제출 준비 완료: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
