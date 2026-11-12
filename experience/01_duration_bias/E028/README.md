# E028. 학습·개발·평가 자료의 원음 길이와 공격별 클래스 구성

**실행 상태:** 메타데이터 분석 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

원음 길이와 공격·출처 구성을 보면 반복과 단음 노출의 클래스 차이를 수량화할 수 있다.

## 그렇게 설계한 이유

모델에 넣은 고정 길이와 원음 길이를 분리하고 짧은 정상·위조의 노출 차이를 확인한다.

대조와 통제의 근거: 원음 전체 길이이며 무음 제거 후 발화량이 아니다. 정확한 2초 경계·4.0375초/3.0275초 경계·0.5초 bin 등은 산출 물별 정의를 유지한다.

## 모델·데이터·표본

학습 자료 25,380·개발 24,844·19LA 71,237·21LA 148,176·DF 533,928의 총 803,565개 원음 메타데이터. 필요 시 실제 header와 대조.

## 실험 내용과 상세 절차

1. 자료 분할을 유지하고 정상/위조별 원음 샘플 수의 히스토그램·누적분포·분위수·2초 미만/이하 비율을 계산한다.
2. 샘플러로 가중하지 않은 고유 원음 분포와 학습 중 실제 노출 분포를 구별한다.
3. 21LA는 공식 공격 ID·코덱·전송 조건, DF는 출처·공격·코덱, Train은 화자·공격별로 교차 집계한다.
4. C 학습에서 상한보다 긴 원음의 개수·고유 샘플 수와 B 대비 추가 사용량도 계산한다.

## 측정과 검증

클래스별 분모·길이 분포 겹침·KS·Wasserstein 거리·출처/공격별 비율. 새 탐지기 성능 실험이 아니다.

## 결과

학습 자료 ≤2초는 정상 109/2,580 =4.2248%, 위조 3,207/22,800 =14.0658%.21LA ≤2초는 정상 571/14,816 =3.8539%, 위조 49,193/133,360 =36.8874%.

## 해석·실패한 부분·한계

파일 개수의 클래스 불균형만으로 클래스별 정규화 EER이 기계적으로 악화되는 것은 아니다. 길이별 난이도·공격 구성과 함께 해석한다.

## 배운 점

원음 분포·샘플러 노출·유효 발화량은 다르다. <2초와≤2초 분모도 구분해야 한다.

## 실험 코드

- [EBM/tools/plot_class_distributions.py](code/EBM/tools/plot_class_distributions.py)

코드의 파일별 원본 위치·SHA256은 [source_manifest.json](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [artifacts/repetition_shortcut_20261111/original_length_distribution.csv](../../_shared/references/artifacts/repetition_shortcut_20261111/original_length_distribution.csv)
- [artifacts/repetition_shortcut_20261111/training_original_lengths.csv](../../_shared/references/artifacts/repetition_shortcut_20261111/training_original_lengths.csv)
- [artifacts/ebm_C_length_class_distribution_20261109/analysis.json](../../_shared/references/artifacts/ebm_C_length_class_distribution_20261109/analysis.json)
- [artifacts/class_distributions_20261108/validation.json](../../_shared/references/artifacts/class_distributions_20261108/validation.json)
- [artifacts/segment_group_shifts_20261110/analysis.json](../../_shared/references/artifacts/segment_group_shifts_20261110/analysis.json)
- [artifacts/duration_zoom_20261109/validation.json](../../_shared/references/artifacts/duration_zoom_20261109/validation.json)

## 연결

다른 관련 가설: [반복 입력과 학습·추론의 상호작용](../../02_repetition/README.md) · [정상 다양성·도메인 지지와 정상쌍 학습](../../03_bonafide_diversity/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
