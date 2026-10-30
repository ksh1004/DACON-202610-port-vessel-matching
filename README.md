# 입항 기록 ↔ 선박 카탈로그 매칭 (DACON 236771)

DACON **「항만 문제해결을 위한 AI 기술 공모전」** 참가 기록입니다. 항만 입항 기록 한 행(선박명·호출부호·톤수·국적·선종·입항 연도)을 공식 선박 카탈로그 78,595척 가운데 하나의 `vessel_id`에 연결하는 개체 매칭 문제를 풀었습니다.

- **평가:** `0.7 × 한글 선박명 행 정확도 + 0.3 × 비한글 행 정확도`
- **제약:** 코드 제출(L4 GPU, 30분), 행 독립 예측(다른 test 행 사용 금지), 외부 데이터 금지, 공개 사전학습 가중치만 허용
- **결과:** Public **0.3616(공식 베이스라인) → 0.9452(v17)**, 제출 14회

## 점수 변화

| # | 버전 | 핵심 변경 | Public |
|---|---|---|---:|
| 1 | baseline | 공식 베이스라인 (TF-IDF 후보 + RandomForest) | 0.3616 |
| 2 | v2 | 여러 검색기의 후보 + LightGBM LambdaRank | 0.4981 |
| 3 | v3 | 한글·영문 공통 **발음 키**(자음 골격) 검색·특징 | 0.7408 |
| 4 | v4 | 톤수 결측·오타·약칭 **증강 학습** | 0.8289 |
| 5 | v6 | 소형 LLM(Kanana-1.5-2.1B)의 **영문 음역 후보** 4개 | 0.8979 |
| 6 | v7 | 약칭 검색·사전 통계 특징, CatBoost + LightGBM 앙상블 | 0.8998 |
| 7 | v8 | 증강 시드가 다른 CatBoost 추가 | 0.8987 |
| 8 | v11 | 상위 20개 후보를 **Qwen3-Reranker-0.6B**(지도학습)로 재판별 | 0.9040 |
| 9 | v12 | 발음 키 입력 **BGE 재판별기** 결합 | 0.9066 |
| 10 | v13 | train 호출부호 조회 → 결과 동일 (test에 같은 호출부호 없음) | 0.9066 |
| 11 | v14 | **train 선박을 후보에서 제외** (test 선박은 train과 겹치지 않음) | 0.9096 |
| 12 | v15 | 한글 질의에 **편집 거리 퍼지 검색** 후보 추가 | 0.9228 |
| 13 | v16 | 퍼지 후보 20 → 50개 | 0.9333 |
| 14 | v17 | 퍼지 후보 200개 + **오타 되돌림** 검색(인접 음절 순서·중복 음절) | **0.9452** |

다음 후보 v18(퍼지 후보 500개)은 준비를 마쳤고 10/31에 제출할 예정입니다.

전체 기록과 해석: [experiments/submissions.md](experiments/submissions.md) · v13~v18 과정 요약: [docs/fuzzy_retrieval_20261030.md](docs/fuzzy_retrieval_20261030.md)

## 접근 방법

```
입항 기록 1행
  │
  ├─ 1단계 후보 검색 (행마다 독립)
  │    문자 n-gram TF-IDF · 정확 일치 · 톤수 근접 · 단어 포함(약칭)
  │    한글이면: 발음 키 TF-IDF · LLM 음역 4개로 영문 검색
  │             · 음역/발음 키 ↔ 카탈로그의 Indel 편집 거리 퍼지 검색 (v15~)
  │             · 오타 되돌림(인접 음절 순서·중복 음절) 발음 키 일치 (v17~)
  │    train에 나온 선박은 후보에서 제외 (v14~)
  │
  ├─ 2단계 순위 학습 (특징 ~106개: 문자열·발음 키·음역 유사도, 톤수 차, 국적·선종 대응 확률 …)
  │    CatBoost YetiRank ×3 + LightGBM LambdaRank ×2, 그룹 내 softmax 가중 평균
  │
  └─ 3단계 한글 행 재판별 (상위 20개)
       Qwen3-Reranker-0.6B 12.5% + 발음 키 입력 BGE-reranker-v2-m3 12.5% + 2단계 75%
```

- 한글 선박명(예: `글로비스 슈프림`)은 영문 카탈로그(`Glovis Supreme`)와 직접 비교할 수 없습니다. 그래서 **규칙 기반 발음 키**(`모닝`·`Morning` → `MNN`)와 **LLM 음역**이라는 두 경로로 같은 공간에 옮겨 비교했습니다.
- 모든 검색·특징·분기는 그 행의 입력과 고정된 학습 산출물만 씁니다. 그래서 행 순서를 섞거나 한 행만 실행해도 같은 결과가 나옵니다(`tools/make_submit.py`가 매번 검사).

## 성능을 끌어올린 판단들

1. **검증은 선박 단위로 나눈다.** 같은 선박의 기록이 학습·검증에 함께 들어가면 점수가 부풀려집니다. 처음부터 선박(호출부호) 단위 GroupKFold를 썼습니다.
2. **Public 점수의 "크기"를 가설 검증에 쓴다.**
   - v13(호출부호 조회)이 v12와 소수점 10자리까지 같았습니다. 이것으로 test 선박이 train에 없다는 것을 확인했고, 바로 train 선박을 후보에서 빼는 v14로 이어졌습니다.
   - v14가 로컬 예측(+0.019)의 1/6만 오른 것은 test 한글의 어려움이 "단어 생략"이 아니라는 증거였습니다.
3. **train에 남은 노이즈 흔적에서 test 노이즈를 추정한다.**
   - 주최 측은 train 영문 이름에 오타를 넣었습니다. 선박의 38%이고, 종류는 글자 삭제·중복·인접 글자 순서 바뀜이며, 단어 생략은 없습니다.
   - 같은 노이즈가 test 한글에 음절 단위로 들어갔다고 보고 측정해 보니, 톤수가 없을 때 정확도가 0.82~0.92로 떨어졌습니다. 실패의 67%는 **정답이 후보에 아예 없었습니다.**
   - 편집 거리 퍼지 검색으로 후보를 보강하자 Public이 세 번 연속 1%p 이상 올랐습니다(v15, v16, v17).
4. **로컬 지표에 갇히지 않는다.**
   - 한동안 합성 지표(testlike)를 0.001 단위로 올리는 실험을 10가지 이상 반복했지만 Public은 움직이지 않았습니다.
   - 지표가 가정한 노이즈(단어 생략)가 실제와 달랐기 때문입니다. 자세한 기록은 [docs/performance_summary_20261029.md](docs/performance_summary_20261029.md)에 있습니다.

## 저장소 구조

```
src/            검색기·특징·순위 학습·증강·LLM 음역·신경망 재판별기 (실험과 제출이 공유)
tools/          패키지 빌드·제출 검증(make_submit.py)·각종 진단/프로브 스크립트
submit/<버전>/  제출 패키지의 script.py와 설정(JSON). 가중치는 저장소에 없음
experiments/    실험 결과 요약(JSON·로그), fold 배정(folds.csv), 제출 기록(submissions.md)
docs/           단계별 실험 보고서, 작업 로그(portfolio_log.md), 프로젝트 개요
env/            서버 고정 버전(requirements-server.txt)·개발용 의존성·환경 점검
PLAN.md, MODEL_PLAN.md   초기 계획서
```

처음 읽는다면 [docs/project_overview.md](docs/project_overview.md)(구조·코드 지도)와 [docs/portfolio_log.md](docs/portfolio_log.md)(작업 순서별 기록, 어려웠던 점·배운 점)를 권합니다.

## 재현 방법

저장소에는 **대회 데이터와 모델 가중치가 없습니다.**

1. DACON 대회 페이지에서 `train.csv`, `vessel_catalog.csv`를 내려받아 프로젝트 루트에 둡니다.
2. Python 3.11 가상환경을 만들고 `env/requirements-server.txt`와 `env/requirements-dev.txt`를 설치합니다.
3. 공개 사전학습 가중치를 받습니다(`tools/download_models.py`).
   - kakaocorp/kanana-1.5-2.1b-instruct-2505
   - Qwen/Qwen3-Reranker-0.6B
   - BAAI/bge-reranker-v2-m3
4. 학습·평가·패키징 예시는 다음과 같습니다(자세한 명령은 [docs/project_overview.md](docs/project_overview.md) 7절).

```bash
python -m src.validate                                   # 선박 단위 fold 생성
python -m src.experiment --learners lgb_base cat_yeti400 # 5-fold OOF + 합성 평가
python tools/build_package.py ...                        # 제출 패키지 구성
python tools/make_submit.py submit/v17_fuzzy200          # ZIP 생성 + 형식·행 독립성·시간 검사
```

> Windows에서 `make_submit.py`는 ZIP(약 6.6GB)을 임시 폴더에 풉니다. 시스템 드라이브 공간이 부족하면 `TMP`/`TEMP`를 다른 드라이브로 지정하세요.

## 규칙 준수

- 학습 데이터는 공식 `train.csv`와 `vessel_catalog.csv`만 사용했고, 외부 선박 데이터는 쓰지 않았습니다.
- 예측은 행마다 독립입니다. 셔플, 부분집합, 단독 행 실행 결과가 같은지 제출 전마다 검사했습니다.
- v13의 호출부호 조회는 규칙 해석 확인 전에 시험한 것입니다. 효과가 없었기 때문에(test에 같은 호출부호 없음) 이후 버전에는 포함하지 않았습니다.
