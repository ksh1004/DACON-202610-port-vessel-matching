"""LLM 음역 결과 캐시: 한글 선박명 → 영문 후보 N-best.

같은 이름은 항상 같은 결과(배치 1, 빔 서치, 결정적)이므로 이름 단위 캐시는 행 독립성에 영향이 없다.
학습·검증 단계에서 생성 비용(약 0.46초/이름)을 한 번만 치르기 위해 파일로 저장한다.

    python -m src.llm_cache          # train 한글 행 + 증강 한글 행의 음역을 미리 생성
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from src.common import ROOT, is_hangul
from src.llm_translit import FEW_SHOT, SYSTEM_PROMPT

MODEL_DIR = Path("D:/KSH/hf_models/kakaocorp__kanana-1.5-2.1b-instruct-2505")
N_BEST = 4
_TAG = hashlib.md5((SYSTEM_PROMPT + json.dumps(FEW_SHOT, ensure_ascii=False) + str(N_BEST)).encode()).hexdigest()[:8]
CACHE_PATH = ROOT / "experiments" / "cache" / f"llm_kanana_{_TAG}.json"


class BeamCache:
    """한글 이름이면 음역 N-best, 아니면 빈 목록을 돌려준다."""

    def __init__(self, path: Path = CACHE_PATH, model_dir: Path = MODEL_DIR, allow_generate: bool = True) -> None:
        self.path = path
        self.model_dir = model_dir
        self.allow_generate = allow_generate
        self.data: dict[str, list[str]] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        self._model = None
        self._dirty = 0

    def _generate_one(self, name: str) -> list[str]:
        if self._model is None:
            from src.llm_translit import Transliterator
            self._model = Transliterator(self.model_dir)
        return self._model.generate([name], n_best=N_BEST, batch_size=1)[0]

    def __call__(self, names) -> list[list[str]]:
        out = []
        for name in names:
            if not is_hangul(name):
                out.append([])
                continue
            if name not in self.data:
                if not self.allow_generate:
                    raise KeyError(f"LLM 캐시에 없는 이름: {name}")
                self.data[name] = self._generate_one(name)
                self._dirty += 1
                if self._dirty >= 200:
                    self.save()
            out.append(self.data[name])
        return out

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)
        self._dirty = 0


def main() -> None:
    from src.augment import augment
    from src.common import load_train

    train = load_train()
    # train_ranker 의 기본 증강 설정과 같아야 캐시가 맞는다 (약칭 혼합 전후 모두)
    augs = [augment(train, 4, 1, 7), augment(train, 4, 1, 7, abbrev_mix=True)]
    names = list(dict.fromkeys(n for n in list(train["vessel_name"]) + [x for a in augs for x in a["vessel_name"]]
                               if is_hangul(n)))
    cache = BeamCache()
    todo = [n for n in names if n not in cache.data]
    print(f"한글 이름 {len(names)}개 중 생성 필요 {len(todo)}개 → {cache.path}", flush=True)
    t0 = time.perf_counter()
    for i, name in enumerate(todo, 1):
        cache([name])
        if i % 250 == 0:
            rate = (time.perf_counter() - t0) / i
            print(f"{i}/{len(todo)}  {rate * 1000:.0f} ms/이름, 남은 시간 {rate * (len(todo) - i) / 60:.1f}분", flush=True)
    cache.save()
    print(f"완료: {len(cache.data)}개 저장", flush=True)


if __name__ == "__main__":
    main()
