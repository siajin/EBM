# E113. short32 중단25epoch의21LA·DF 전체평가와 독립ROC 검산

**실행 상태:** 전체평가·독립 검산 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

원본240의조건부3/2초정렬이 같은epoch의B/C보다단음일반화를개선할수있다.

## 그렇게 설계한 이유

초기5epoch의미확정상태를갱신하고 사용자가지정한25epoch의최종완료결과를 남긴다.

대조와 통제의 근거: B/C의기본입력·점수정의는같지만별도평가실행난수다. [E026](../E026/README.md)과독립학습반복으로세지않는다. target에서사후최저epoch를고른평가가아니다.

## 모델·데이터·표본

[E026](../E026/README.md)과같은한학습궤적의CLS25/TrainState25. 19LA71,237은저장된중간평가,21LA148,176·DF533,928은별도고정25 전체평가. checkpoint SHA53c6e963d76608a8b2490f531834822ace79832c315af0a15ec2ed1b77b7e8be.

## 실험 내용과 상세 절차

1. 25epochcheckpoint commit·SHA와19LA완료를확인한뒤 사용자승인으로 해당학습만중단했다. 학습은재개하지않는다.
2. 별도고유root에서 실제두GPU·rank별4worker smoke와19LA저장점수일치후21LA→DF전체평가를수행했다.
3. 앞64600cap·비반복·마스크·batch8·seed1729·ASP잡음1e-5·학습OC중심cosine을사용했다. optimizer갱신0회다.
4. 이번출판작업에서완료표식과metrics를읽고 파생점수만로컬메모리로받아682,104행의index·UID유일성·class분모·cap·유한값을검사하고동점ROC를독립재계산했다. 원음·모델·SSL출력은수신하지않았다.
5. 같은epoch B25/C25와비교하며 역사적24025 반복평가는별도조건으로표시한다.

## 측정과 검증

21LA정상14,816/위조133,360·DF정상14,869/위조519,059. independent_eer차이<1e-10%p. checkpoint SHA와완료run.exit0를대조.

## 결과

19LA1.2100611828687935%,21LA11.308488302339532%,DF5.427399287107404%. B25는0.788579/5.009748/2.512431%,C25는0.652617/5.248950/2.730513%. 세집합모두short32가나빴다.

## 해석·실패한 부분·한계

해당설정·seed·25epoch에서개선가설이지지되지않았다. 기존4/3항재가중과3/2항추가중어느변화가원인인지분리할수없다. 50epoch완료나모든저장점전체평가로보고하지않는다.

## 배운 점

최종지정25epoch는세집합모두B/C25보다나빴다. 초기5epoch와최신완료를구분하고실패관측도보존한다.

## 실험 코드

- [EBM/tools/evaluate_short32_fixed25.py](code/EBM/tools/evaluate_short32_fixed25.py)
- [EBM/tools/analyze_short32_fixed25.py](code/EBM/tools/analyze_short32_fixed25.py)

코드의 파일별 원본 위치·SHA256은 [source_manifest.json](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [EBM/tools/evaluate_short32_fixed25.py](../../../tools/evaluate_short32_fixed25.py)
- [EBM/tools/analyze_short32_fixed25.py](../../../tools/analyze_short32_fixed25.py)
- [artifacts/short32_train_20261111_v2/stop25_eval_status.json](../../_shared/references/artifacts/short32_train_20261111_v2/stop25_eval_status.json)
- [최신25epoch 전체점수 독립검산](evidence/independent_audit.json)
- [최신 metrics_21LA.json](evidence/metrics_21LA.json)
- [최신 metrics_21DF.json](evidence/metrics_21DF.json)
- [평가완료 표식](evidence/complete.json)

## 연결

다른 관련 가설: [반복 입력과 학습·추론의 상호작용](../../02_repetition/README.md) · [정상 다양성·도메인 지지와 정상쌍 학습](../../03_bonafide_diversity/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
