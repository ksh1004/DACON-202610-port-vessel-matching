# 프로젝트 전체 개요 (인수인계용)

> 이 문서 하나로 "무엇을 왜 했고, 지금 어디에 있고, 다음에 무엇을 하면 되는지"를 이해하도록 쓴 문서입니다.
> 사람과 AI 모두를 독자로 가정합니다. 기준 시점: **2026-10-30**
>
> 최신 실행: **[explicit_input_20261030.md](explicit_input_20261030.md)**. 기존입력15,756쌍에서Qwen/BGE잘림0을확인했다. 짧은필드·톤수관계와모음철자표현의BGE4모델·2,400업데이트도완료했으나두형식모두같은fold1·4의v12 0.908779보다낮은0.907212로기각했다. 별도[Qwen3-Reranker-4B 고정진단](reranker4b_probe_20261030.md)도17/48→15/48정답으로하락해확대하지않았다. 새제출용최종모델·ZIP·실제제출없음. [결산](../experiments/representation_iteration_20261030.json), 작업로그49~51장.
>
> 이전 실행: **[listwise_training_20261030.md](listwise_training_20261030.md)**. 같은1,200질의의BGE학습후보8개/20개모델4개·총2,400업데이트를완료했다. 20후보testlike0.906713은같은fold1·4의v12 0.908779와8후보대조군0.907088보다낮아기각했다. 직전[저차원보정](calibration_20261030.md)도같은fold0·2의v12 0.917877보다낮은0.913327/0.912963으로기각했다. 저장모델·체크포인트·기존v12 소스/ZIP보존확인, 새최종모델·ZIP·실제제출없음. [결산](../experiments/improvement_iteration_20261030.json), 작업로그47·48장.
>
> 성능 개선 과정 요약: **[performance_summary_20261029.md](performance_summary_20261029.md)**. 베이스라인부터v12까지의실제제출9회, 이후기각한실험, 검증한계와남은과제를정리했다. 문서 정리 작업은로그46장.
>
> 이전 실행: **[context_density_20261029.md](context_density_20261029.md)**. 공식 기록의 국적·선종·연도·넓은 크기 구간으로 조건부 제원 모델과 대조군6개를 실제 학습했다. 같은fold1·4의testlike0.906311은v12 0.908779보다 낮아 기각했다. 제원 특징이 같아 분리할 수 없는 오답과 후보 누락도 별도 집계했다. 새최종모델·ZIP·실제제출 없음. [결산](../experiments/context_iteration_20261029.json), 작업 로그45장.
>
> 이전 실행: **[byte_names_20261029.md](byte_names_20261029.md)**. ByT5를 두fold에서 실제 학습·재학습했지만 원래확률0.904412·길이정규화0.898605로 같은fold v12의0.908779보다 낮아 모두 기각했다. 별도[기존Kanana 전방확률48척 진단](forward_name_probe_20261029.md)도17→15척으로 하락했다. 저장모델·행/후보순서·기존v12 소스/ZIP 보존을 확인했다. 새최종모델·ZIP·실제제출 없음. [결산](../experiments/name_iteration_20261029.json), 작업 로그43·44장.
>
> 이전 실행: **[ambiguity_loss_20261029.md](ambiguity_loss_20261029.md)**. 모호한 음성 가중치0.25 후보와 두 대조군을fold1·4에서 실제 학습해6개 모델·7,200업데이트를 완료했다. 후보testlike0.907739는 같은두fold v12의0.908779보다 낮고, 원래손실 대조군 대비+0.000125에 그쳐 기각했다. 초기 파라미터·입력과 실제 저장 모델의 행/후보 순서 검사를 확인했으며 재학습과 기존v12의 차이도 기록했다. 새 최종 모델·ZIP·실제 제출 없이 v12 유지. 작업 로그42장.
>
> 이전 설계: **[observation_aware_redesign_20261029.md](observation_aware_redesign_20261029.md)**. 약칭·결측의 관측 모호성을 집계하고 손실 완화→ByT5 이름 전용 모델→공식 기록의 조건부 입항 사전확률을 제안했다. 세 가설은 이후 각각 실제 실행해 기각했다. 해당 문서는 당시 제안으로 보존하며 최신 실행과 구분한다. 작업 로그41~45장.
>
> 이전 실행: **[nested_combiner_20261029.md](nested_combiner_20261029.md)**. 내부 교차 예측용 Qwen/BGE8개를 새로 학습하고4계열·8선택 구성을 평가했다. 최고 토큰 통계 후보0.916929도 같은 두 fold의 v12 기준0.917877보다 낮아 모두 기각했다. 신경망 결합0.913911·원시78개0.909866·역방향 추가0.913643. 새 최종 모델·ZIP·실제 제출 없이 v12를 유지하며 목표0.97은 미달이다. 작업 로그40장.
>
> 직전 실행: **[reverse_name_ranking_20261028.md](reverse_name_ranking_20261028.md)**. 후보 조건부 음역186,300쌍과 직접 LambdaRank 네 구성을 실제 평가했지만 v12보다 낮아 기각했다. 4B 후보 공동 비교도 어려운60척에서21→17척으로 탈락했다. 새 ZIP·실제 제출 없음.
>
> 이전 조사: **[research_097_redesign_20261028.md](research_097_redesign_20261028.md)**. 현재 결합에서 BGE12.5%만 완벽하게 교체해도 로컬 오라클은0.947200. 제안했던 역방향·교사·내부 결합은 이후 실제 실행해 기각했다. 이 문서의 제안 상태와 최신 실행 결과를 구분한다.
>
> 이전 실행: **[learned_alignment_20261028.md](learned_alignment_20261028.md)** (실제 개선 실험 네 안 모두 기각, v12 유지) · [catalog_warm_training_20261028.md](catalog_warm_training_20261028.md) (새 이름 사전 적응) · [focused_training_20261028.md](focused_training_20261028.md) (약칭·결측 구성 A1/A2) · 개선 방향 [performance_rethink_20261028.md](performance_rethink_20261028.md) · 이전 후속 실험 [encoder_extension_20261028.md](encoder_extension_20261028.md) (4개안 기각) · v12 기록 [encoder_iteration_20261027.md](encoder_iteration_20261027.md) · v11 기록 [pair_iteration_20261027.md](pair_iteration_20261027.md) · v10 기록 [implementation_run_20261027.md](implementation_run_20261027.md) · 목표 전략 [target_097_plan.md](target_097_plan.md) · 작업 로그 [portfolio_log.md](portfolio_log.md) (51장까지) · 제출 기록 [../experiments/submissions.md](../experiments/submissions.md)

---

## 1. 문제 정의

**DACON 236771 "항만 문제해결을 위한 AI 기술 공모전"**

- **입력**: 항만 입항 기록 1행 (`vessel_name, call_sign, gross_tonnage, international_gross_tonnage, flag_name, vessel_type, arrival_year`)
- **출력**: 공식 선박 카탈로그(78,595척)에서 그 선박의 `vessel_id`
- **지표**: `0.7 × 한글 기록 정확도 + 0.3 × 비한글 기록 정확도`
  - 한글 기록 = 선박명에 한글 음절이 하나라도 있는 행
  - 카탈로그 선박명은 100% 영문 → **한글 → 영문 음역 매칭이 핵심**
- **데이터**: train 12,222행 (한글 1,191행 / 382척, 9.7%), 카탈로그 78,595척. **실제 test 행 수는 비공개**이며 sample_submission의 3,000행은 실제 규모를 뜻하지 않음 ([공식 데이터 설명](https://dacon.io/competitions/official/236771/data))
- **공식 평가 범위**: Public은 test의 50%, Private은 100%. 온라인 예선은 종료 시점 Public 사용 ([공식 평가 안내](https://dacon.io/competitions/official/236771/overview/evaluation))

### 꼭 지켜야 하는 규칙

| 규칙 | 의미 | 이 프로젝트의 대응 |
|---|---|---|
| **행 독립** | test의 다른 행·분포를 쓰면 안 됨. 셔플·부분집합으로 실행해도 예측이 같아야 함 | 모든 계산을 행 단위로. 정규화도 한 질의의 후보 그룹 안에서만. `make_submit.py`가 자동 검사 |
| **결정성** | 같은 입력 → 같은 출력 | LLM은 **배치 1**로 생성 (배치 생성은 패딩 때문에 결과가 바뀜) |
| 외부 데이터 금지 | 공개 범용 사전학습 가중치만 허용 | LLM 가중치만 사용, 학습은 공식 데이터로만 |
| 코드 제출 | `script.py` + `requirements.txt` + `model/`, zip 10GB 이하 | `tools/build_package.py`로 조립 |
| 서버 환경 | Python 3.11, transformers 4.57.6 고정, **추론 30분 제한**, 인터넷 차단 | 로컬 `.venv`를 서버 버전에 맞춤 (`env/check_env.py`) |
| 제출 횟수 | 하루 5회, 실행 오류도 차감 | 진단용 제출 없음. 로컬 지표로 결정 |

**사용자가 정한 원칙**
- 제출 횟수를 최소화하고, 실제 개선일 때만 제출
- train·test 선박 중복과 `call_sign` 조회는 **쓰지 않음** (2026-10-27 사용자 재확인, 주최 측 문의 답변 대기). 기존 v9_callsign은 현재 개선 후보에서 제외
- 모든 작업 단계를 [portfolio_log.md](portfolio_log.md)에 기록 (작업 내용 + 어려운 점·개선점·배운 점)

### 일정
- 팀 병합 마감 2026-10-26 · **리더보드 마감 2026-11-01** · 본선 발표 2026-11-10

---

## 2. 현재 상태 한눈에

| 항목 | 값 |
|---|---|
| 2026-10-30 입력 표현·4B전용모델 | 15,756쌍씩의토큰검사에서잘림0. 같은1,200그룹으로compact/characters를각두fold에서학습했으나모두testlike0.907212<v12 0.908779로기각. 4B전용사전학습모델도동일48척에서17→15정답으로기각했다. 입력모델4개·2,400업데이트와실행검사완료, 양자화·전체5-fold·새ZIP·실제제출없음. [입력](explicit_input_20261030.md), [4B진단](reranker4b_probe_20261030.md), [결산](../experiments/representation_iteration_20261030.json) |
| 2026-10-30 저차원 보정 | 내부 선박 검증 점수로3/6계수 보정기4개를 적합했다. 같은fold0·2의testlike0.917877→조건부0.913327/공통0.912963으로모두기각했다. 조건부확률손실은줄었지만정답률은하락했다. [기록](calibration_20261030.md) |
| 2026-10-30 학습 후보 확대 | 같은1,200질의·600업데이트의BGE8/20후보모델을fold1·4에서각각학습했다. 새20후보0.906713은v12 0.908779·8후보대조군0.907088보다낮아기각했다. 원본유지, 변형1회복/8손실. 실제저장4모델순서검사·8개체크포인트CPU상태확인, 전체5-fold·새ZIP·실제제출없음. [기록](listwise_training_20261030.md) |
| **최고 Public** | **0.9065951735** (v12, 2026-10-28 사용자 보고, 서버 **9분 51초**) |
| 목표 | **리더보드 0.97**, 호출부호·train/test 동일 선박 조회 제외 |
| 제출 횟수 | 9회 (baseline, v2, v3, v4, v6, v7, v8, v11, v12) |
| v7 로컬 지표 | test 유사 0.9020 · OOF 0.9965 · hard 0.8440 |
| **후속 평가 결과** | v8: CatBoost 증강 시드 7·8·9 각 ×3 + LightGBM 시드 7·8. test 유사 **0.9050** · OOF 0.9965 · hard 0.8462, **Public 0.8987** |
| 신규 로컬 후보 v10 | v8 + 지도학습 Qwen3-Reranker-0.6B 25%. 5-fold test 유사 **0.9074** · OOF **0.9977** · hard **0.8503**. Public 미측정. 최종 학습·패키지 검사 완료, [ZIP](../dist/v10_neural.zip) 5.83GB |
| 신규 후보의 한계 | v8 대비 test 유사 +0.0024의 선박 bootstrap 95% 구간 [-0.0030, +0.0079], 5개 fold 중 1개 하락. 기존 후보 검색 매퍼 예외가 남는 비교이며 0.97 미달 |
| 제출 v11 | 음역 입력·학습 fold 혼동 후보 보강. 5-fold test 유사 **0.9088** · hard **0.8541** · OOF **0.9977**. Public **0.9039635945**, 서버 **9분 31초**(사용자 보고). [ZIP](../dist/v11_pairviews.zip) 5.83GB |
| 제출 v12 | v8 75% + v11 Qwen 12.5% + 발음 키 BGE 인코더 12.5%. 5-fold test 유사 **0.9119** · hard **0.8579** · OOF **0.9977**. Public **0.9065951735**, 서버 **9분 51초**(2026-10-28 사용자 보고). v11 대비 **+0.0026315790**, 시간 **+20초**. [ZIP](../dist/v12_encoder.zip) **6.98GB**, 로컬 3,000행 **177.0초**, 행 독립성·실제 두 판별기 호출·CRC 검사 통과. [실험 기록](encoder_iteration_20261027.md) |
| 2026-10-28 후속 검증 | 학습 그룹 확대, 신경망 40%, 고정 추가 시드의 가중치/확률 평균을 실제 검증. 모두 각 두 fold에서 testlike 하락으로 기각. 새 5-fold·최종 모델·ZIP·실제 제출 없음. v12 유지. [비교 결과](encoder_extension_20261028.md) |
| 추천 방향 실제 실행 | 약칭·결측 구성 A1, 양성 복원 A2, 새 이름 사전 적응 B, 학습형 문자 정렬 C를 실제 평가했으나 네 안 모두 사전 조건 미달. 새 최종 모델·ZIP 없음. [결산](learned_alignment_20261028.md) |
| 2026-10-28~29 새 구조 실행 | 역방향186,300이름 쌍과 직접 LambdaRank 네 구성을 두 fold에서 평가했지만 v12의0.917877보다 낮은0.902993~0.905300으로 모두 기각. 별도4B 공동 비교 진단도 어려운60척에서21→17척으로 탈락했다. [실행 결과](reverse_name_ranking_20261028.md), [결산](../experiments/reverse_iteration_20261029.json) |
| 2026-10-29 내부 교차 예측 결합 | 새 내부 신경망8개·메타15,031그룹으로4계열을 실제 평가했다. 최고 토큰 통계 후보 testlike0.916929는 같은 두 fold의 v12 대비-0.000948. hard0.866003→0.871533이지만 단어 생략 한글0.846501→0.828442로 하락해 전체 기각. 53,020후보의 선택기 재로딩·행 순서·후보 순서 검사와 기존13개 소스·ZIP SHA 보존 확인. 새 최종 모델·ZIP·실제 제출 없음. [실행 결과](nested_combiner_20261029.md), [결산](../experiments/nested_iteration_20261029.json) |
| 2026-10-29 모호한 음성 손실 | fold1·4에서 손실3조건·모델6개를 학습했다. 모호성 완화0.907739는 같은 두fold v12의0.908779보다-0.001040, 원래 손실 대조군 대비+0.000125에 그쳐 기각. 초기 파라미터·학습 토큰 동일, 실제 모델6개의 질의18개 행/후보 순서 검사와 소스30개·기존v12 소스/ZIP 보존 확인. 재학습 대조군과 과거v12의 차이는 원인 미확정으로 기록. [실행 결과](ambiguity_loss_20261029.md), [결산](../experiments/encoder_ambiguity_v1_final.json) |
| 2026-10-29 바이트 이름 전사·전방 확률 | 공식 쌍877/852개로 내부검증·재학습4회,1,215업데이트를 수행했다. ByT5 원래확률0.904412·길이정규화0.898605는 같은fold1·4의v12 0.908779보다 낮아 모두 기각. 별도기존Kanana 전방확률 진단도48척에서17→15로 하락했다. 새최종모델·ZIP·실제제출 없음. [ByT5](byte_names_20261029.md), [전방진단](forward_name_probe_20261029.md), [결산](../experiments/name_iteration_20261029.json) |
| 2026-10-29 조건부 제원 밀도비 | 공식 학습 선박4,680/4,681척의 제원 맥락으로 조건부·주변·섞은 맥락 모델6개를 학습했다. 같은fold1·4 testlike0.906311은v12 대비−0.002467, 대조군 대비−0.002259로 기각. 실제 한글 변형 오답324개 중182개는 같은 인코딩 제원 후보에 가리고51개는상위20 밖이다. 정답 사용 가상 보정0.926812는 이 고정 메커니즘만의 진단으로 실제 성능·대회 상한이 아니다. [기록](context_density_20261029.md), [결산](../experiments/context_iteration_20261029.json) |
| 최신 구조 진단·연구 | BGE12.5%만 완벽하게 교체하는 로컬 오라클0.947200, 신경망 합계25% 완벽 교체0.973554. 실제 달성 점수나 Public 상한이 아니다. UPR식 후보 조건부 음역 점수, Jina v3.5/Set-Encoder식 후보 공동 비교, 조건부 최종 선택과 교사 증류를 제안. [조사와 실행안](research_097_redesign_20261028.md) |
| 목표 대비 진단 | v7/v8 top5 정답 선택 오라클도 0.9651/0.9668로 0.97 미달. top20 오라클은 0.9904/0.9890. 이는 실제 모델 성능이 아님 |
| 남은 과제 | v12 Public은 로컬 testlike보다 약 0.0053 낮고 목표 0.97까지 **0.0634048265** 남음. 로컬 순증 신뢰 구간은 0 포함, 별도 1,596행 진단은 1행 순감. 톤수가 남은 약칭 회귀와 기존 후보 검색 매퍼 예외도 남음. [실행 결과와 한계](encoder_iteration_20261027.md) 참고 |

### 점수 흐름

| 버전 | 핵심 변경 | Public | test 한글 추정* |
|---|---|---|---|
| baseline | 공식 베이스라인 (후보 검색 + RandomForest) | 0.3616 | ≈ 0.10 |
| v2 | LightGBM LambdaRank 재순위 | 0.4981 | ≈ 0.28 |
| v3 | 한글 → 영문 **발음 키** | 0.7408 | ≈ 0.63 |
| v4 | **노이즈 증강** (톤수 결측·오타·약칭) | 0.8289 | ≈ 0.76 |
| v6 | **LLM 음역** (Kanana-1.5-2.1B) | 0.8979 | ≈ 0.855 |
| v7 | 약칭 검색 + 사전정보 특징 + **CatBoost 앙상블** | **0.8998** | ≈ 0.858 |
| v8 | CatBoost 증강 시드 7·8·9 앙상블 | 0.8987 | ≈ 0.857 |
| v11 | 음역 포함 지도학습 판별기 + 혼동 후보 학습 | **0.9039635945** | ≈ 0.864 |
| v12 | 발음 키 다국어 인코더 + Qwen 고정 결합 | **0.9065951735** | ≈ 0.867 |

\* 비한글 test 정확도를 0.998로 두고 Public에서 역산

---

## 3. 시스템 구조 (v8 기준, v11 추가 단계)

```
입항 기록 1행
 │
 ├─ [1단계] 후보 검색  src/retrieval.py   (질의마다 약 180개, recall 한글 1.000 / 비한글 0.999)
 │    · 영문 문자 n-gram TF-IDF (단어 경계 / 띄어쓰기 제거) · 정확 일치 · 총톤수 근접
 │    · 한글: 발음 키(자음 골격) TF-IDF + 키 정확 일치          src/hangul.py
 │    · 한글: LLM 음역 4개(빔)로 영문 검색                       src/llm_translit.py
 │    · 약칭: 남은 단어를 모두 포함하는 선박 중 톤수가 가까운 순 (톤수 없으면 선종·국적 일치 순)
 │
 ├─ [2단계] 쌍 특징 78개  src/features.py
 │    이름 유사도 · 발음 키 유사도 · LLM 빔 유사도 · 단어 포함률 · 톤수 간격
 │    · 선종·국적 매핑 확률 · 후보 사전정보(입항 선박에 흔한 국적·선종, 크기)
 │
 └─ [3단계] 재순위 앙상블  src/ensemble.py
      CatBoost YetiRank ×3 + LightGBM LambdaRank(증강 시드 7) + LightGBM(증강 시드 8)
      → 질의 그룹 안에서 softmax 정규화 → 가중 합 → 1등 vessel_id
```

- **LLM**: `kakaocorp/kanana-1.5-2.1b-instruct-2505`, few-shot 12개, 빔 4, 배치 1. 로컬 경로 `D:/KSH/hf_models/kakaocorp__kanana-1.5-2.1b-instruct-2505`
- 추론 시간의 대부분은 LLM 생성 (한글 행만, 같은 이름은 실행 중 메모이제이션)

v11은 위 v8 기준 점수의 상위 20개에 추가 판별 단계를 둔다. 한글 기록의 원문·최대 4개 음역·제원을 학습한 Qwen3-Reranker-0.6B에 넣고 기존 확률 75% + 신경망 확률 25%로 결합한다. 비한글 입력은 기존 점수를 유지한다. 승자가 바뀔 수 없는 확률 간격이면 추가 계산을 생략한다. 고정 두 판별기 평균은 0.9093이었으나 fold 순증 조건(2/5) 미달로 선택하지 않았다.

v12는 별도의 BAAI/bge-reranker-v2-m3에 원문·규칙 발음 키·음역·제원을 학습한다. 기존 v8 75%와 v11 Qwen/새 인코더 각 12.5%를 결합한다. 한글 입력만 상위 20개를 판별하며 승자 보존 경계도 유지한다. 일반 인코더는 hard 하락으로 기각했고, 발음 키 보강 후 고정 평균만 초기 두 fold와 전체 조건을 통과했다. 자세한 근거는 [최신 실험](encoder_iteration_20261027.md)에 있다.

---

## 4. 어떻게 여기까지 왔나 (단계별 흐름)

각 단계는 **"측정 → 원인 발견 → 대응"** 순서였습니다.

| # | 발견한 것 | 대응 | 효과 |
|---|---|---|---|
| 1 | 한글이 점수의 70%인데 베이스라인은 한글 후보 recall부터 68.6% | 검색 + LambdaRank 재순위 2단계 구조 | Public 0.36 → 0.50 |
| 2 | 한글 이름은 영어·로마자·병음·일본어·알파벳 읽기가 섞여 있음 | 규칙 기반 **발음 키** (`모닝 → MNN = Morning`) | OOF 한글 0.41 → 0.97, Public 0.74 |
| 3 | CV 0.97인데 test 한글 0.63. train 한글은 **톤수 결측이 0%**, 약칭도 없음 | **스트레스 테스트**로 약점 측정 → **노이즈 증강** | 톤수 결측 시 0.54 → 0.86, Public 0.83 |
| 4 | 규칙으로 안 되는 음역 관습이 남음 | **LLM 음역** 5종 비교 → Kanana 선택, 배치 1로 결정성 확보 | OOF 0.995, Public 0.898 |
| 5 | 남은 차이가 어디서 오는지 모름 | 제출 4개 점수를 시나리오 혼합으로 **역추정** (비음수 최소제곱) | test 한글 ≈ 원본 44% / 톤수 결측 14% / 단어 생략 23% / 둘 다 19% |
| 6 | 약칭 행의 정답이 후보에 없음 (17~24%) | **단어 포함 + 톤수 근접 검색기** | recall 0.83 → 0.96, 하지만 정확도는 그대로 (자매선 → 정보 한계) |
| 7 | CV가 포화되어 비교가 안 됨 | 5의 구성을 재현한 **test 유사 평가 세트** (Public과 ±0.003 일치) + 오프라인 블렌딩 | CatBoost가 최고 알고리즘, v7 Public 0.8998 |

### 효과가 없었던 시도 (다시 하지 말 것)

| 시도 | 결과 | 이유 |
|---|---|---|
| LightGBM 하이퍼파라미터·목적함수 변형 (잎 수, 반복, xendcg, 이진 분류, 한글 가중) | ±0.002 | 입력이 같으면 같은 곳에서 틀림 |
| LightGBM 변형끼리 앙상블 | +0.002 | 위와 같은 이유 |
| Kanana LoRA 파인튜닝 | 검색 상위 10위 0.932 → 0.838 | 과적합 (학습 쌍 1,619개) |
| 약칭 방식 혼합 증강 (첫 단어만, 마지막 단어만, 앞 60%) | hard 0.837 → 0.832 | 정보 한계 |
| test 구성에 맞춘 증강 프로필 (`testmix`) | test 유사 0.8958 (기존 0.8954) | 오타 섞인 다양한 증강이 오히려 일반화에 유리 |
| 두 번째 LLM (Midm) | 미적용 | 서버 추론 약 31분 예상 → 30분 제한 초과 |

---

## 5. 평가 체계 (결정은 이것으로 함)

| 지표 | 만드는 법 | 쓰임 |
|---|---|---|
| **OOF** | 선박(`call_sign`) 단위 5-fold. `experiments/folds.csv` | 깨끗한 데이터 성능. 0.995 이상으로 포화 |
| **hard** | 검증 행에 노이즈(오타·약칭·톤수 결측)를 입힌 고정 복사본 5,382행 | 노이즈 강건성 |
| **test 유사 (testlike)** | 한글 1,191행 × 톤수 결측 / 단어 생략 / 둘 다, 역추정한 test 구성으로 가중 + OOF 비한글 | 기존 회귀 확인용. v8은 Public보다 약 0.006 높아 단독 모델 선택 지표로는 부족 |

- 순위 모델·증강·범주 특징 재계산은 fold 경계를 지킴 (증강·변형 행은 원본과 같은 fold에서만 평가). 단, 후보 검색의 범주 정렬 매퍼는 전체 train으로 적합하는 예외가 있어 독립 평가 경로 보강 필요 ([후속 분석 4절](performance_next_steps.md#4-평가-해석에서-고칠-점))
- **Public 해상도는 미확인**: 실제 test 행 수와 한글 비중이 비공개이므로 한글 1행을 0.0023점으로 고정할 근거가 없음. 기존 로컬 +0.005 기준은 운영상 기준이며 통계적으로 보장된 임계값이 아님
- 약칭 + 톤수 결측의 특정 규칙 정확도는 0.425, 실제 이름이 생략된 행의 v7 정확도는 0.3948. **0.425는 정보이론적 상한이 아님**. 전체 시나리오에는 이름이 유지된 행도 26.2% 포함되어 전체 정확도는 0.5357

---

## 6. 코드 지도

### src/
| 파일 | 역할 |
|---|---|
| `common.py` | 데이터 로딩, 이름 정규화, 한글 판별, 대회 지표 |
| `hangul.py` | 한글·영문 → 발음 키 (규칙 기반) |
| `retrieval.py` | 1단계 후보 검색 (`CatalogRetriever.generate`) |
| `features.py` | 2단계 쌍 특징 78개 (`FeatureBuilder`), 선종·국적 매퍼 (`CategoryMapper`) |
| `llm_translit.py` | LLM 음역 (`Transliterator`), 프롬프트·few-shot, 모델 벤치마크 |
| `llm_cache.py` | 이름 → LLM 빔 캐시 (학습·검증용 사전 생성) |
| `augment.py` | 노이즈 증강 (`augment`), 약칭 함수들, 증강 프로필 |
| `validate.py` | fold 분할, recall 평가 |
| `train_ranker.py` | `Context` (후보·특징 캐시, fold 학습), 단일 LightGBM 학습 |
| `experiment.py` | `Lab`: OOF·hard·testlike 동시 평가, 학습기 목록 `LEARNERS`, 점수 저장 |
| `blend.py` | 저장된 점수로 앙상블 조합을 오프라인 평가 |
| `ensemble.py` | 앙상블 멤버 정의 `SPECS`, 전체 데이터 학습, 추론용 `Ensemble` |
| `stress.py` | 시나리오별 스트레스 테스트 (400행 표본) |
| `llm_finetune.py` | LoRA 파인튜닝 (채택 안 함) |
| `match_features.py`, `feature_experiment.py` | 추가 특징 28개와 혼동 후보 재학습 실험. 신규 최종 후보에는 미반영 |
| `strict_hangul_experiment.py` | 학습 fold만으로 범주 매퍼·후보·특징을 생성하는 한글 검증 경로 |
| `neural_reranker.py` | 학습/검증 선박을 분리한 Qwen3 후보 쌍 판별기의 LoRA 학습·평가 |
| `neural_pair.py` | 병합 가중치 추론, v8과의 25% 결합, 승자 보존 경계로 계산 생략 |
| `pair_evidence.py`, `pair_views.py` | v11의 행별 음역 포함 인코딩과 단독/고정 평균 추론. 최종 선택은 evidence 단독 |
| `encoder_pair.py`, `encoder_phonetics.py`, `encoder_views.py` | v12의 다국어 인코더·발음 키 입력·Qwen과의 고정 결합 추론 |
| `encoder_experiment.py`, `encoder_phonetic_experiment.py` | 일반/발음 키 인코더의 학습 fold 전용 후보 선별과 병합 후 5-fold 평가 |

### tools/
| 파일 | 역할 |
|---|---|
| `script_template.py` | 제출용 `script.py` 원본 (`encoder_views.json` → `pair_views.json` → `neural.json` → `matching.json` → `ensemble.json` → 단일 모델 순으로 구성 선택) |
| `build_package.py` | `submit/<이름>/` 조립 (src 모듈, 모델 파일, LLM 가중치) |
| `make_submit.py` | zip 생성 + 서버 구조로 실행 + 형식·행 독립성·시간 검사 |
| `download_models.py` | LLM 가중치 다운로드 |
| `train_neural_final.py` | 공식 전체 train 원본·증강으로 후보 쌍 판별기를 학습하고 병합 가중치로 내보냄 |
| `audit_neural_cv.py` | 동일 v8 후보에서 5-fold 집계, few-shot 선박 제외, paired bootstrap |
| `check_package_neural.py` | 실제 신경망 호출 행의 패키지 셔플·단독 실행 일치 검사 |
| `audit_pair_iteration.py`, `select_pair_iteration.py` | v10 대비 동일 후보 비교·불확실성·사전 후보 조건 적용 |
| `pair_fresh_stress.py`, `release_pair_iteration.py` | 별도 변형 진단 / 기준 통과 후 로컬 최종 학습·패키지 실행 검사. 외부 제출 없음 |

### 데이터·산출물
| 경로 | 내용 |
|---|---|
| `experiments/cache/` | 후보·특징 캐시 (`*_features_<해시>.joblib`), LLM 캐시 `llm_kanana_73ef4cc1.json` |
| `experiments/scores/` | 학습기별 후보 점수 (블렌딩 재료) |
| `experiments/experiment_results.jsonl` | 모든 실험 결과 한 줄씩 |
| `experiments/artifacts/ranker_v7/` | v7 앙상블 모델 파일 |
| `submit/`, `dist/` | 제출 패키지 폴더와 zip |

### 실험 스위치 (환경 변수)
| 변수 | 기본 | 의미 |
|---|---|---|
| `VM_TOKEN_SEARCH` | 1 | 0이면 약칭 검색기 끔 (v6와 같은 후보) |
| `VM_AUG_SEED` | 7 | 증강 시드 (배깅용) |
| `VM_ABBREV_MIX` | 0 | 1이면 약칭 방식 혼합 증강 |
| `VM_AUG_PROFILE` | default | `testmix` 증강 프로필 |
| `VM_THREADS` | 12 | CatBoost 스레드 수 |

**캐시 키 주의**: 특징 캐시 파일명은 `FEATURE_NAMES` + LLM 캐시 이름 + `RETRIEVAL_VERSION`의 해시입니다. **검색 방식을 바꾸면 `retrieval.py`의 `RETRIEVAL_VERSION`을 올려야** 예전 캐시를 잘못 재사용하지 않습니다.

---

## 7. 자주 쓰는 명령 (프로젝트 루트, `.venv/Scripts/python.exe`)

```bash
# 실험: 학습기 5-fold → OOF / hard / testlike 기록 + 점수 저장 (새 이름이 있으면 --build-hard 로 LLM 생성 허용)
python -m src.experiment --learners lgb_base cat_yeti400 --tag _prior
VM_AUG_SEED=8 VM_THREADS=16 python -m src.experiment --learners cat_yeti400 --tag _prior_a8

# 앙상블 조합 평가 (저장된 점수 사용, 재학습 없음)
python -m src.blend "cat_yeti400_prior*3+lgb_base_prior+lgb_base_prior_a8" --modes z prob

# 최종 앙상블 학습 (멤버 이름@시드 = 그 증강 시드로 학습)
python -m src.ensemble --members cat_yeti400 lgb_base lgb_base@8 --weights 3 1 1 --mode prob --out experiments/artifacts/ranker_v7

# 패키지 조립과 검사
python tools/build_package.py v7_ens --hangul-mode ranker --artifacts experiments/artifacts/ranker_v7 --llm-dir D:/KSH/hf_models/kakaocorp__kanana-1.5-2.1b-instruct-2505
python tools/make_submit.py submit/v7_ens
```

- 실험 이름과 점수 파일: `<학습기 이름><--tag>` (예: `cat_yeti400_prior_a8`)
- `experiment.py`의 학습기와 `ensemble.py`의 `SPECS`는 **같은 설정을 따로 정의**하므로, 새 학습기를 추가할 때 두 곳 모두 고쳐야 함

### 실행 시간과 자원 (로컬: RTX 5060 Ti 16GB, RAM 32GB, 16스레드)
| 작업 | 시간 |
|---|---|
| 특징 캐시 생성 (train / 증강 / hard / testlike) | 약 2.5 / 5 / 2 / 2분 (단독 실행 시) |
| LightGBM 5-fold | 3~6분 |
| CatBoost YetiRank 5-fold (400회) | 40~66분 |
| 전체 데이터 앙상블 학습 | 약 15분 |
| LLM 생성 | 이름당 약 0.45초 |

- **무거운 작업은 2개까지만 동시에**: CatBoost(7GB) + 실험 + LLM을 함께 돌리면 RAM 부족(페이징 파일 오류)

---

## 8. 다음 단계

2026-10-27부터 목표는 **0.97**이다. 최신 진단과 새 실험 설계는 [performance_rethink_20261028.md](performance_rethink_20261028.md), 공식 근거와 기존 결과는 [target_097_plan.md](target_097_plan.md)를 참고한다.

1. 약칭·결측 구성, 양성 복원, 새 이름 사전 적응, 학습형 문자 정렬과 최신 역방향 네 구성·4B 교사 진단은 모두 실제 실행했고 기준에 미달했다. 상세 기록과 사전 조건을 확인해 같은 설정을 반복하지 않는다.
2. 모호한 음성 손실·ByT5 이름 전사 두 설정·기존Kanana 전방확률·조건부 제원 밀도비까지 실제 실행해 기각했다. 원래 손실 재학습과 과거v12의 차이는 아직 원인을 확정하지 못했다. [제원 실험의 분리 가능성 진단](context_density_20261029.md)을 보면 같은 두fold의실제한글변형오답324행 중182행은동일한인코딩제원에가리고51행은상위20밖이다. 같은제원집단안에서실제로구별할수있는이름근거나후보누락을해결하는새가설이필요하며, 현재0.97달성경로는확보하지못했다. 실패 후 같은 가중치·정렬 기준을 다시 탐색하지 않는다.
3. 기존 v8 후보 검색 범주 매퍼 예외가 남는 비교와 검색까지 분리한 진단을 구분한다. 검증에는 정답을 강제로 추가하지 않으며 별도 진단·전체5-fold의 사전 기준을 통과한 뒤 최종 학습을 진행한다.
4. 후속 제출 여부는 행 독립성·시간 검사와 함께 판단한다. v12 실제 서버 시간은 9분51초지만 실제 test 행 수는 비공개다. 로컬3,000행 벤치마크로 서버 전체 시간을 보장하지 않는다.

---

## 9. 주요 교훈 (발표·포트폴리오용)

1. **입력 신호가 모델보다 중요했다**: 발음 키 +0.24, 증강 +0.09, LLM +0.07 vs 학습기 변경 ±0.002
2. **CV가 포화되면 새 평가 축을 만든다**: 스트레스 테스트 → hard → test 유사 세트
3. **제출은 실험이다**: 제출 기록을 연립방정식처럼 풀어 숨겨진 test 구성을 추정
4. **정확도가 안 오르면 정보가 충분한지부터 본다**: 관측 필드 충돌과 모델 실수를 구분한다. 특정 규칙의 정확도를 모든 모델의 상한으로 해석하지 않는다
5. **같은 특징의 모델끼리는 앙상블 효과가 작다**: 다른 알고리즘(CatBoost)을 섞어야 효과
6. **성능 최적화가 규칙 위반이 될 수 있다**: LLM 배치 생성의 비결정성 → 배치 1, 자동 행 독립성 검사
