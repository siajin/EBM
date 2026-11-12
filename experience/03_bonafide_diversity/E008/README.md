<a id="e008-기존-학습들의-배치정상-쌍-수학습-진행량-비교"></a>

# 기존 학습들의 배치·정상 쌍 수·학습 진행량 비교

**실행 상태:** 기록 분석 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

배치와 쌍 수의 영향을 갱신 횟수·같은 UID 쌍 비율까지 보면 이전 성능 차이를 더 정확히 설명할 수 있다.

## 그렇게 설계한 이유

배치 증가 또는 GPU 간 정상 쌍 공유의 결과를 갱신 수·학습률·쌍 구성까지 함께 해석한다.

대조와 통제의 근거: 첫 배치 실패 320은 성능 비교에서 제외한다. 원래 loss1 +loss2 합 계는 서로 다른 배치·학습 단계의 직접적인 일반화 지표가 아니다.

## 모델·데이터·표본

[배치 80·80epoch의 초기 반복 입력 학습](../E001/README.md)~[배치 240·원래 반복 입력·GPU 별 정상 쌍 학습](../../02_repetition/E005/README.md)의 보관 설정·손실 로그·EER 기록. 새 음성 추론·학습 없음.

## 실험 내용과 상세 절차

1. 실험별 보관 소스와 설정 차이를 대조한다. 배치 80/160/240/320,epoch80/50, 학습률 1e-6/1e-5를 구분한다.
2. 정상 수 n 에서 로컬 n² 쌍과 같은 원음 대 각 비율 1/n을 계산한다. 전역 공유는 두 GPU 정상 합계의 제곱으로 계산한다.
3. 각 epoch의 배치 수 323/162/108/81 예정과 두 Adam 호출을 구분하고, 기록된 EER의 공통 epoch와 관측 최소를 계산한다.
4. 손실 로그는 한 epoch 평균이 아닌 약 10%마다 기록한 9개 창의 loss1 +loss2 임을 확인한다.

## 측정과 검증

공통 epoch 차이·관측 최소·학습률 로그·노출쌍 수를 기술한다. 통제된 여러 seed 배치 ablation은 아니다.

## 결과

GPU 간 공유는 쌍 수를 늘렸지만 19LA 최소는 로컬보다 높고 21LA·DF 최소는 낮았다. 배치 240도 모든 집합에서 일관된 우월성을 보이지 않았다.

## 해석·실패한 부분·한계

동일 발화 쌍 비중과 다른 원음 쌍 비중이 배치 크기에 따라 바뀐다. 배치 크기 하나가 성능 변화의 원인이라는 주장은 하지 않는다.

## 배운 점

손실 로그의 집계 단위와 optimizer 횟수를 맞춰야 한다. 쌍 수와 독립 원음 다양성은 같은 양이 아니다.

## 실험 코드

- [소스: plot_ebm_all_runs_20261104.py](code/artifacts/analysis/plot_ebm_all_runs_20261104.py)
- [소스: plot_ebm_bpl_comparison.py](code/artifacts/analysis/plot_ebm_bpl_comparison.py)
- [소스: plot_ebm_diagnosis_20261104.py](code/artifacts/analysis/plot_ebm_diagnosis_20261104.py)
- [소스: plot_ebm_three_runs_paper.py](code/artifacts/analysis/plot_ebm_three_runs_paper.py)

코드의 파일별 원본 위치·SHA256은 [원본 파일 위치와 해시 목록](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [설정·결과 근거](../../_shared/references/artifacts/analysis/ebm-all-runs-comparison-20261104.json)
- [설정·결과 근거](../../_shared/references/artifacts/analysis/ebm-performance-diagnosis-20261104.json)

## 연결

다른 관련 가설: [반복 입력과 학습·추론의 상호작용](../../02_repetition/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
