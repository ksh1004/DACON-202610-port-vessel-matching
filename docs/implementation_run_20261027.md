# 실제 학습을 통한 성능 개선 실험

작업일: 2026-10-27. 사용자 요청에 따라 특징 추가, 혼동 후보 추가 학습, 지도학습 후보 판별기를 구현·학습·검증했다. 호출부호 조회, train/test 동일 선박 조회, 외부 선박 데이터는 사용하지 않는다. 5-fold 집계, 최종 모델 학습, 제출 ZIP 실행 검사를 완료했다. 대회 제출은 수행하지 않았다.

## 비교 조건

기준은 v8이다. CatBoost 증강 시드 7·8·9에 각각 가중치 3, LightGBM 시드 7·8에 각각 가중치 1을 준 질의별 확률 앙상블이다. 동일한 검증 행과 후보 ID·순서로 비교하며 검증 후보에 정답을 추가하지 않는다.

기존 v8의 검증 후보에는 전체 train으로 적합한 검색 범주 정렬 매퍼가 쓰인 예외가 남아 있다. 아래 점수는 이 과거 후보를 공유한 비교다. 새 신경망의 학습 후보·범주 매퍼는 검증 선박을 제외해 생성했다. 학습/검증 선박 중복은 없다. 별도의 독립 검색 경로도 구현했다.

testlike는 기존 44/14/23/19%의 합성 시나리오 혼합이며 실제 test 구성으로 확인된 적이 없다. 선박 단위 bootstrap도 이 가정이나 반복적인 모델 선택의 편향을 제거하지 못한다. 실제 test를 읽거나 제출하지 않았다. Public 최고는 기존 v7의 0.8998이며 목표 0.97은 아직 달성하지 않았다.

## 특징·트리 모델: 완료된 5-fold 결과

| 모델 | testlike | hard | OOF |
|---|---:|---:|---:|
| 기존 LightGBM | 0.8954 | 0.8405 | 0.9941 |
| 추가 특징 LightGBM | 0.8994 | 0.8417 | 0.9941 |
| 추가 특징 + 상위 20개 추가 LightGBM | 0.9008 | 0.8438 | 0.9941 |
| 혼동 후보 추가 CatBoost 단독 | 0.9030 | 0.8466 | 0.9977 |
| 기존 v8 | 0.9050 | 0.8462 | 0.9965 |
| v8 + 혼동 후보 CatBoost | 0.9058 | 0.8476 | 0.9965 |

`src/match_features.py`는 카탈로그 IDF, 단어 순서, 음역 빔별 지지, 비슷한 이름 후보의 톤수 구별 등 28개 특징을 추가한다. 기존 78개와 분리해 총 106개로 실험했다. `src/feature_experiment.py`는 학습 fold에만 적합한 1차 모델로 혼동 후보를 고른다. 점수와 후보 ID·순서·소스 해시를 함께 저장한다.

v8 + 혼동 후보 CatBoost의 testlike 순증은 +0.000872, 선박 단위 paired bootstrap 95% 구간은 [-0.001007, +0.002763]이다. 큰 개선의 근거로 부족해 최종 후보로 채택하지 않는다. 한글 가중치 8의 추가 실험도 v8 블렌딩에서 0.9046~0.9047로 개선되지 않았다. [집계 근거](../experiments/matching_result_audit.json)

## 독립 후보 검색

`src/strict_hangul_experiment.py`에서 범주 매퍼 적합·후보 생성·검색 순위 특징까지 학습 fold로 분리했다. 신경망 학습에 사용하는 5개 fold 후보를 생성했다. 다음은 고정 fold 4의 한글 전용 모델 비교다.

| 특징 | testlike 한글 | hard 한글 |
|---|---:|---:|
| 기본 78개 | 0.8565 | 0.8159 |
| 추가 포함 106개 | 0.8473 | 0.8087 |

추가 특징의 이득은 이 독립 경로에서 유지되지 않았다. 한글 전용 모집단으로 학습하므로 전체 모집단을 쓰는 v7/v8과 점수 자체를 직접 비교하지 않는다.

## 지도학습 후보 판별기

공식 [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) 범용 가중치의 리비전 `e61197ed45024b0ed8a2d74b80b4d909f1255473`을 고정했다. 선박 전용 외부 데이터나 원격 실행 코드는 사용하지 않는다.

- 입력: 기록과 후보의 이름·총톤수·선종·국적, 기록의 국제총톤수. 호출부호·sample_id·vessel_id·정답 ID는 넣지 않는다.
- 학습: 공식 train 원본·증강 중 한글 원본 행, 정답 1개 + 음성 7개, 2,400개 그룹, 고정 1 epoch. 음성은 문자열·음역·톤수 특징으로 어려운 후보 3개와 나머지 무작위 4개다.
- LoRA rank 8, alpha 16, dropout 0.05, 학습률 1e-4, 그룹 배치 2, 최대 256토큰, 시드 42다. 학습/검증 선박을 분리한다.
- 메모리/속도 조정으로 fold 0·1은 gradient checkpointing을 끄고, fold 2·3·4는 켰다. 학습량·시드·손실·배치·선택 비율은 동일하다. 최종 전체 학습은 checkpointing을 끈다. 연산 경로가 모두 동일한 반복 실험은 아니라는 한계도 남는다.
- 추론: 실제 입력 이름에 한글이 있으면 v8 상위 20개를 판별하고 기존 확률 75% + 신경망 확률 25%로 결합한다. 이 비율과 학습량을 fold 결과에 맞춰 조정하지 않았다.
- 배포: LoRA를 병합한 가중치를 사용해 PEFT 설치가 필요 없다. 한 질의의 후보만 함께 처리한다. 톤수 표현을 정규화해 단독 CSV의 dtype 차이에 대응한다.
- 상위 20개 내 재정규화한 기존 상위 두 확률 차이가 1/3보다 크면, 어떤 신경망 분포에서도 승자가 바뀔 수 없어 추가 계산을 생략한다. 독립 fold 4에서 1,662행 중 1,379행(83.0%)이 해당했고 실제 계산과 승자 일치를 확인했다.

초기 독립 fold 4의 한글 testlike는 0.8565 → 0.8683, hard 한글은 0.8159 → 0.8321이었다. 병합 후에는 0.8684 / 0.8339였다. BF16 병합 전후 연산 차이가 있어 최종 검증과 배포는 병합 모델로 통일한다. 동일한 v8 기준에서는 한글 testlike 0.8663 → 0.8748, hard 한글 0.8267 → 0.8267이다. 약한 기준 모델 대비 개선 폭을 최종 성과로 사용하지 않는다.

### 전체 5-fold 집계

[neural_cv_audit.json](../experiments/neural_cv_audit.json)에 최종 수치·선박 단위 불확실성·few-shot 제외 결과를 저장했다. 한글이 모두 사라진 변형 20행은 실제 배포 규칙에 맞춰 v8 예측을 유지한다.

| 지표 | 기존 v8 | v10 신경망 25% 결합 | 차이 |
|---|---:|---:|---:|
| testlike | 0.904959 | 0.907375 | +0.002416 |
| hard | 0.846180 | 0.850294 | +0.004114 |
| 원본 OOF | 0.996517 | 0.997693 | +0.001175 |
| testlike 한글 | 0.865004 | 0.868455 | +0.003451 |
| OOF 비한글 | 0.998187 | 0.998187 | 0 |

원본 한글 1,191행, hard 한글 원본의 변형 2,382행, testlike 변형 3,573행을 모두 평가했다. 전체 7,146행 중 경계 조건으로 추가 계산을 생략한 행은 5,725행(80.1%)이다. 실제 비한글 입력은 v8 경로를 유지한다.

한글 시나리오별 정확도는 원본 0.995802 → 0.997481, 톤수 결측 0.973971 → 0.979849, 단어 생략 0.809404 → 0.812762, 둘 다 0.549118 → 0.554996다. Few-shot 예시 선박 전체를 제외하면 testlike는 0.904562 → 0.907055다.

예측 변화도 함께 확인했다. 원본 OOF는 2행을 회복하고 정답을 잃은 행은 0개다. hard는 53행 회복·39행 손실, testlike 변형은 111행 회복·93행 손실이다. 회복 수만 세면 개선을 크게 보게 되므로 손실을 함께 기록한다. [변화 집계](../experiments/neural_change_audit.json)

testlike 순증의 선박 단위 paired bootstrap 95% 구간(2,000회)은 **[-0.003029, +0.007896]**이다. fold별 순증은 +0.006459 / +0.002947 / +0.003900 / **-0.007230** / +0.005913이다. 평균은 개선됐지만 다섯 fold에 일관되지는 않고, 0을 포함하는 불확실성도 남는다. 따라서 **검증용 신규 후보**로 패키지를 준비하되 기존 Public 최고를 대체하는 확정 모델이나 0.97 달성 모델로 선언하지 않는다.

## 기각한 무학습 방식

`src/closed_name_score.py`에서 Kanana로 카탈로그 이름의 조건부 토큰 확률을 계산했다. 고정 소규모 진단이다.

| 시나리오 20행씩 | 기존 정답 수 | 토큰 확률 단독 | 기존 75% + 토큰 확률 25% |
|---|---:|---:|---:|
| 톤수 결측 | 19 | 20 | 20 |
| 실제 단어 생략 | 15 | 1 | 15 |
| 실제 단어 생략 + 결측 | 8 | 0 | 7 |

약칭의 생략된 단어를 강하게 벌점 주므로 채택하지 않는다. 한 질의 후보 사이에서만 KV 캐시를 공유한다. 전체 시퀀스 직접 계산과의 수치 일치와 다른 질의 처리 후 재계산 일치를 확인했다.

## 실행 검증 및 산출물

- 특징의 차원·유한값·ID 입력 무시·단어 순서 정렬·행 순서 불변을 검사했다. 독립 후보 생성은 직렬/스레드 실행의 후보 ID·순서·특징이 같았다.
- 후보 메타데이터만 읽는 `tools/blend_matching.py`로 v7/v8을 재현했다. 2단계 트리 점수는 로그확률로 저장해 중복 softmax를 피한다.
- `tools/check_neural_runtime.py`: 병합·저장·재로딩한 점수와 저장 OOF가 정확히 일치했다. 배포 블렌딩·그룹 분할·셔플·단독 행 실행도 일치했다. [로그](../experiments/run_neural_runtime_check.log)
- `tools/train_neural_final.py`는 전체 공식 학습 데이터로 동일한 고정 설정의 최종 모델을 학습한다. `tools/build_package.py`는 신경망·v8·Kanana를 오프라인 패키지로 조립한다.
- `tools/make_submit.py`로 3,000행 실행·300행 부분집합·단독 행·형식·ZIP 크기를 검사한다. train 유래 실행 표본의 정확도를 검증 성능으로 사용하지 않는다.
- `tools/check_package_neural.py`는 별도 프로세스에서 패키지를 읽고 실제 신경망이 호출된 스트레스 행의 셔플·단독 실행 일치를 검사한다.

최종 모델은 공식 전체 train에서 만든 원본·증강 후보 그룹 중 고정 시드로 선택한 2,400개 그룹을 학습했다. 학습·병합·내보내기는 로컬 9.35분이었다. [설정·소스 및 데이터 해시](../experiments/artifacts/ranker_v10_neural/neural.json), [학습 로그](../experiments/run_train_final_neural.log). 배포 폴더에는 병합 가중치를 넣고 학습용 LoRA 어댑터를 제외한다.

최종 ZIP은 [dist/v10_neural.zip](../dist/v10_neural.zip), 폴더는 [submit/v10_neural](../submit/v10_neural)이다. 크기 5,833,583,258 bytes(5.83GB, 5.43GiB), 파일 41개로 10GB 제한 이하다. 호출부호 조회표·학습 어댑터는 포함하지 않는다. Qwen 원본 모델 카드, 선언된 Apache-2.0 라이선스, 수정 내역을 포함했다.

| 실행 검사 | 결과 |
|---|---|
| ZIP을 임시 폴더에 풀고 3,000행 실행 | PASS, 로컬 179.6초 |
| 출력 컬럼·행 수·고유 sample_id·카탈로그 ID | PASS |
| 셔플 부분집합 300행 + 단독 행 3회 | PASS, 전체 실행과 예측 일치 |
| 실제 신경망이 호출되는 스트레스 50행 | PASS, 31행에서 추가 신경망 호출 |
| 스트레스 전체 셔플 + 신경망 호출 행 단독 2회 | PASS, 예측 일치 |
| 추론 소스 해시·78개 특징 스키마·조회표 없음 | PASS |
| 최종 ZIP CRC·중복 엔트리 없음·SHA-256 기록 | PASS |

실행 검사 후 출처·라이선스 3개 파일만 ZIP에 추가했다. 기존 추론 코드·가중치 엔트리의 CRC와 크기가 모두 유지됨을 확인하고 최종 ZIP 전체 CRC를 재검사했다. [배포 메타데이터](../experiments/release_v10_neural.json), [일반 실행 로그](../experiments/run_check_v10_neural.log), [실제 신경망 경로 검사 로그](../experiments/run_check_v10_targeted.log).

train 유래 3,000행 실행에서 출력된 0.9994는 학습 데이터 점수이므로 위 5-fold 성과에 사용하지 않는다. 실제 test 크기는 비공개이며 서버 시간도 미측정이라 로컬 179.6초로 서버의 전체 30분 제한 통과를 보장하지 않는다. **신규 후보의 Public은 미측정, 0.97은 미달**이다.

## 재현 명령

```powershell
.venv/Scripts/python.exe -m src.feature_experiment --models rich150 hard20 hardcat
.venv/Scripts/python.exe -m src.strict_hangul_experiment --folds 0 1 2 3 4 --learners base
# 각 fold 실행. fold 4의 최초 어댑터 재사용 경로는 개별 로그 참고.
.venv/Scripts/python.exe -m src.neural_reranker --fold 0 --tag neural_v8_cv --max-groups 2400 --no-checkpoint --skip-pretrained --merge-eval --bound-prune --v8-oof-eval
.venv/Scripts/python.exe -m tools.audit_neural_cv
.venv/Scripts/python.exe -m tools.train_neural_final --out experiments/artifacts/ranker_v10_neural
.venv/Scripts/python.exe tools/build_package.py v10_neural --hangul-mode ranker --artifacts experiments/artifacts/ranker_v10_neural --llm-dir D:/KSH/hf_models/kakaocorp__kanana-1.5-2.1b-instruct-2505
.venv/Scripts/python.exe tools/make_submit.py submit/v10_neural --n 3000 --subset 300 --singles 3
.venv/Scripts/python.exe -m tools.check_package_neural submit/v10_neural
```
