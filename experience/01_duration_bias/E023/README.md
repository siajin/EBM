# E023. 가변 길이 학습과 기존 모델의 21LA 길이·공격별 오류 재분석

**실행 상태:** 기존 점수 분석 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

가변길이 학습의 성능 저하는 짧은 음성 또는 특정 공격 하나만으로 설명되지 않을 수 있다.

## 그렇게 설계한 이유

성능 저하가 짧은 구간에만 있는지, A10·A11 또는 임계값 이동으로 설명되는지 확인한다.

대조와 통제의 근거: 전체 21LA 에서 정한 임계값은 사후 진단용이다. 정상 571개 의 ≤2초 와 507개 의<2초는 다른 구분이다. 동일 epoch와 사후 최소 epoch를 섞지 않는다.

## 모델·데이터·표본

[E022](../E022/README.md)의 21LA 전체 점수, B20/C20, 참고 B45/C30·원래 24050의 완료 점수. 모든 발화 ID·원음 길이를 맞춘다.

## 실험 내용과 상세 절차

1. 원음 길이를 ≤1,(1,2],(2,3],(3,4.0375],>4.0375초로 나눈다.
2. 각 모델의 전체 21LA EER 근사 임계값 하나를 고정하고 길이별 정상/위조 오류 분모와 전체 오류 기여를 계산한다.
3. 길이 내부에서 다시 계산한 EER과 전체 임계값 오류를 별도 표로 둔다. 개발 임계값이 남은 참조 모델에는 그 결과도 병기한다.
4. 공격·코덱 metadata를 연결해 길이 × 공격 교차 표를 만들고, 같은 원음 길이 공통 지지 집합의 재가중 EER도 계산한다.

## 측정과 검증

전체 EER 재현, 모든 길이별 클래스 분모/오류합, 공격별 누락·점수 분포. 길이 구간 안의 두 클래스가 없으면 EER은 NA 다.

## 결과

가변 길이 20epoch21LA EER 8.908218%를 재현했다. 길이 조정 민감도에서도 학습 구성 차이와 공격·코덱 조합이 남아 ‘짧아서만’이라는 단일 설명은 성립하지 않았다.

## 해석·실패한 부분·한계

학습 원인의 인과 검증이 아니다. 이 항목은 새 변형 추론 없이 기존 점수·metadata를 분석한 것이다.

## 배운 점

길이 내부 오류·전체 임계값·공격×코덱을 따로 봐야 한다. 단일 전체 EER로 원인을 확정할 수 없다.

## 실험 코드

- [EBM/tools/analyze_coverage20_length_shortcut.py](code/EBM/tools/analyze_coverage20_length_shortcut.py)

코드의 파일별 원본 위치·SHA256은 [source_manifest.json](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [EBM/tools/analyze_coverage20_length_shortcut.py](../../../tools/analyze_coverage20_length_shortcut.py)
- [artifacts/coverage20_length_shortcut_20261111/plan.json](../../_shared/references/artifacts/coverage20_length_shortcut_20261111/plan.json)
- [artifacts/coverage20_length_shortcut_20261111/analysis.json](../../_shared/references/artifacts/coverage20_length_shortcut_20261111/analysis.json)
- [artifacts/coverage20_length_shortcut_20261111/findings.json](../../_shared/references/artifacts/coverage20_length_shortcut_20261111/findings.json)

## 연결

다른 관련 가설: [반복 입력과 학습·추론의 상호작용](../../02_repetition/README.md) · [정상 다양성·도메인 지지와 정상쌍 학습](../../03_bonafide_diversity/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
