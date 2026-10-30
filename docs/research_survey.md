# 관련 연구·모델 조사 (2026-10-26)

> **최신 조사(2026-10-28):** 아래는 당시 판단이며, 이후 지도학습 신경망으로 실제 Public이 개선됐고 새로운 구조적 제약도 확인했다. BGE12.5%만 교체하는 로컬 오라클0.947200과 후보 조건부 음역·공동 후보 비교의 후속안을 [새 연구 문서](research_097_redesign_20261028.md)에 정리했다. 아래의 “대부분 입력 정보 부족”, “효과0~0.002” 등은 입증된 일반 상한이나 최신 권고가 아니다.

> 2026-10-27 주석: 아래의 “정보 한계”는 특정 노이즈 가설과 규칙으로 관측한 결과이며 엄밀한 상한이 아니다. 호출부호 제외는 현재 사용자 방침으로, 명시적 금지 조항을 확인했다는 의미가 아니다. 최신 목표 분석과 실제 후보 선택 시험은 [target_097_plan.md](target_097_plan.md)를 참고한다.

> 목적: 지금 파이프라인(발음 키 + LLM 음역 + GBDT 재순위)에 없는 새로운 신호나 방법이 있는지 찾기
> 결론 먼저: **이름 유사도를 더 잘 재는 방법은 여럿 있지만, 남은 손실의 원인(약칭 행의 정보 부족)을 푸는 방법은 찾지 못했음.** 직접 시험한 Symphonym은 보완 효과가 작음

## 1. 조사한 분야와 결과

| 분야 | 대표 자료 | 핵심 내용 | 이 대회에 적용 |
|---|---|---|---|
| 한·영 음역 / 역음역 | Oh & Choi (COLING 2002), Kang & Choi (ACL 2000), Kang & Kim (COLING 2000), Koo (HMM 외래어 음운) | 발음 단위 정렬, 결정 트리·HMM·신경망 음역. 대부분 2000년대 초반 | 우리 발음 키와 LLM이 이미 같은 역할. 외래어 사전·용례 데이터는 **외부 데이터라 사용 불가** |
| 문자 체계를 넘는 음성 임베딩 | **Symphonym** (Gadd, arXiv 2601.06932, 2026) | IPA 조음 자질로 학습한 교사 모델을 문자 단위 학생 모델(830만 매개변수)로 증류. 37개 문자 체계, 한국어 포함. 가중치 CC BY 4.0, 코드 BSD-3 | **직접 시험함** (2장) |
| 문자 단위 Siamese 네트워크 | Neculoiu et al. (2016) | 문자 BiLSTM 쌍둥이 망으로 이름 정규화 | 학습 쌍이 382척뿐이라 처음부터 학습하기엔 데이터 부족 |
| 소형 언어모델 기반 엔티티 매칭 | Steiner et al. (arXiv 2409.08185), Peeters et al. (EDBT 2025), "Deep Dive Into Cross-Dataset EM" (EDBT 2025) | 레코드 쌍을 문장으로 만들어 소형 LLM을 미세조정하면 큰 모델 프롬프팅과 비슷한 성능. 학습 데이터에 구조화된 설명을 넣으면 일부 모델에서 개선 | 이름+톤수+선종+국적 쌍 분류기로 쓸 수 있음. 하지만 우리 오답은 입력에 정보가 없는 경우라 효과가 제한적일 것 |
| 교차 인코더 재순위 | Qwen3-Reranker-0.6B (Apache 2.0, transformers 4.51 이상) | 질의-문서 쌍 점수, 100개 이상 언어 | 서버(transformers 4.57.6)에서 동작 가능. 미세조정하면 2번과 같은 역할 |
| 다국어 의미 임베딩 | BGE-M3 및 한국어 미세조정판 | 의미 검색용. 한 연구에서 한국어 질의의 재현율이 영어보다 크게 낮았고(61.5 vs 88.5), 음역 개체명에서 약함 | 음역 매칭에는 부적합 |
| 로마자 변환·발음 변환 도구 | uroman (ISI, ACL 2018), g2pK, Epitran, PanPhon | 모든 문자를 라틴 문자로 옮기거나 IPA로 바꿔 문자열 거리 적용 | 발음 키와 역할이 겹침. Epitran·PanPhon은 서버에 없음 |
| 선박 신원 매칭 실무 | Global Fishing Watch (등록부 매칭) | 이름은 신뢰도가 낮아 호출부호·IMO·국적 등 여러 필드를 함께 봄. 개명 선박이 가장 어려움 | 우리도 이름+톤수+선종+국적을 함께 씀. 호출부호·IMO는 이 대회에서 쓸 수 없음 |
| 같은 대회 공개 저장소 | GitHub `haneul55/-AI-` | 로마자·발음 정규화 + LightGBM LambdaRank, 특징 44개. 선박 단위 CV 0.941, 리더보드 기록 없음. `call_sign` 조회 사용 | 우리와 같은 계열의 방법이고 더 약함. 새 아이디어 없음 |

## 2. Symphonym 직접 시험

- 내려받은 코드를 실행하지 않고, 구조를 [src/symphonym.py](../src/symphonym.py)로 다시 작성해 가중치(safetensors)와 어휘(JSON)만 읽음. 가중치를 엄격 모드로 불러와 구조가 일치함을 확인
- train 한글 370척 (few-shot 선박 제외), 이름만으로 카탈로그 78,595척 검색

| 검색 방법 | 1위 정확도 | 상위 10위 재현율 |
|---|---|---|
| 발음 키 (규칙) | 0.549 | 0.800 |
| LLM 음역 (Kanana) | 0.714 | 0.924 |
| **Symphonym** | 0.214 | 0.400 |
| 발음 키 ∪ LLM | | 0.962 |
| 발음 키 ∪ LLM ∪ Symphonym | | 0.973 (Symphonym만 찾은 선박 4척) |

- 지명으로 학습한 모델이라 선박 이름(영어 단어 위주)에서는 약함. 보완 효과는 370척 중 4척
- 속도: 카탈로그 전체 임베딩 15초(CPU), 질의는 이름당 수 밀리초 → 추론 시간 부담은 작음
- 규칙 측면: 공개 사전학습 가중치이지만 지명 전용 모델이라 "범용" 가중치로 볼 수 있는지 애매함 → 쓴다면 주최 측 확인 필요

## 3. 판단

- 남은 손실의 95%는 약칭 행이고, 그중 대부분은 **행에 있는 정보로는 구분할 수 없는 자매선**(이전 분석의 정보 상한). 조사한 방법은 모두 "이름 유사도를 더 잘 재는 방법"이라 이 한계를 넘지 못함
- 효과가 있을 수 있는 경우는 test에 우리가 재현하지 못한 노이즈(예: 한글 오타)가 있을 때뿐
- 시도할 가치 순서
  1. Symphonym 유사도를 재순위 특징으로 추가 (구현 1시간, 검증 1시간, 예상 효과 0~0.002)
  2. Qwen3-Reranker-0.6B를 (질의, 후보) 쌍으로 미세조정해 상위 후보만 재채점 (1~2일, 효과 불확실)
  3. 나머지(uroman, g2pK, BGE-M3)는 기존 신호와 겹치거나 부적합

## 참고 자료

- [Symphonym (arXiv 2601.06932)](https://arxiv.org/abs/2601.06932) · [모델 카드 docuracy/symphonym-v7](https://huggingface.co/docuracy/symphonym-v7/blob/main/README.md) · [Zenodo 배포본](https://zenodo.org/records/22767194)
- [Fine-tuning Large Language Models for Entity Matching (arXiv 2409.08185)](https://arxiv.org/abs/2409.08185) · [Entity Matching using Large Language Models (EDBT 2025)](https://openproceedings.org/2025/conf/edbt/paper-81.pdf) · [A Deep Dive Into Cross-Dataset Entity Matching with LLMs (EDBT 2025)](https://openproceedings.org/2025/conf/edbt/paper-224.pdf) · [Match, Compare, or Select? (COLING 2025)](https://aclanthology.org/2025.coling-main.8.pdf)
- [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) · [Qwen3 Embedding 기술 보고서](https://arxiv.org/html/2506.05176v1)
- [Learning Text Similarity with Siamese Recurrent Networks](https://aclanthology.org/W16-1617.pdf)
- [An English-Korean Transliteration Model Using Pronunciation and Contextual Rules (COLING 2002)](https://aclanthology.org/C02-1099.pdf) · [English-Korean Automatic Transliteration/Back-transliteration (ACL 2000)](https://aclanthology.org/www.mt-archive.info/ACL-2000-Kang.pdf) · [A Statistical Model of Korean Loanword Phonology](https://www.sjsu.edu/faculty/hahn.koo/research/papers/koo_jk19.pdf)
- [Retrieval-augmented generation in multilingual settings (BGE-M3 한국어 결과)](https://arxiv.org/html/2407.01463v1)
- [uroman (ACL 2018)](https://aclanthology.org/P18-4003/) · [g2pK](https://pypi.org/project/g2pK/)
- [Global Fishing Watch: Matching Broadcasts to Vessel Registries](https://globalfishingwatch.org/data/matching-broadcasts-vessel-registries/)
- [GitHub haneul55/-AI-](https://github.com/haneul55/-AI-)
