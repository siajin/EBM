# E037. 매 epoch 모든 원음을 사용하면서 기존 기대 선택 비율을 유지하는 샘플러

**실행 상태:** 학습 자료 재생·검증 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

모든 원음을 먼저 한 번 넣고 추가 확률을 보정하면 epoch 길이와 기대분포를 유지하며 전수 방문할 수 있다.

## 그렇게 설계한 이유

복원 추출로 한 epoch에 일부 원음을 못 보는 문제를 추가 학습 배치 없이 해결한다.

대조와 통제의 근거: 108배치 유지. 두 GPU 합계 OC 정상 2,580 +12 방문·위조 22,800 +528 방문·오른쪽 정상 2,580 +12 방문. 각 GPU가 전체 원음을 별도로 보는 것은 아니다.

## 모델·데이터·표본

OC 정상 2,580·OC 위조 22,800·BPL 오른쪽 정상 2,580의 세 stream. 두 GPU 합집합에서 전수 사용한다.

## 실험 내용과 상세 절차

1. 각 stream 에서 모든 원음을 한번씩 포함한다.
2. 총슬롯 T·원음 수 N·기존 원음 확률 p_i에 대해 부족 T −N 슬롯의 추가 확률을(Tp_i −1)/(T −N)로 정하고 추가 추출 후 전체 순서를 섞는다.
3. 모든 원음에서 Tp_i ≥1 인지 검사한다. 불가능하면 원음 확률이나 epoch 길이를 몰래 바꾸지 않고 중단한다.
4. 왼쪽 OC 정상과 오른쪽 BPL 정상은 독립 순서·독립 추가 추출을 사용한다. 길이·음성활동은 이전 조건부 정책을 그대로 쓴다.
5. 원음 선택 50epoch, 실제 구간 전후 각 10epoch285,120개 와 2worker 배치를 재생한다.

## 측정과 검증

원음 전수 포함·기대 확률 일치·실제 길이/VAD 잔차·화자/공격 TV·원음 시간 사용률.

## 결과

모든 50epoch 원음 100% 사용.epoch1 시간 사용률은 OC 정상 39.24 →50.21%, 위조 38.68 →50.01%, 정상 OC +BPL61.19 →71.70%. 실제 10epoch 목표 길이 최대 잔차는 0.347/0.102/0.471%p.

## 해석·실패한 부분·한계

원음 100% 사용은 원음의 모든 시간 구간·긴 연속 문맥 100% 사용이 아니다. 이 샘플러 검증 자체의 EER은 없으며 학습에 연결한 결과는 [E021](../E021/README.md)이다.

## 배운 점

원음 방문100%를 원음 시간·문맥100% 사용으로 부를 수 없다. 유한 배치의 길이 잔차도 보고해야 한다.

## 실험 코드

- [EBM/positive_bpl.py](code/EBM/positive_bpl.py)
- [EBM/tools/validate_source_coverage.py](code/EBM/tools/validate_source_coverage.py)

코드의 파일별 원본 위치·SHA256은 [source_manifest.json](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [artifacts/source_coverage_20261110/verified_summary.json](../../_shared/references/artifacts/source_coverage_20261110/verified_summary.json)
- [artifacts/source_coverage_20261110/final_audit.json](../../_shared/references/artifacts/source_coverage_20261110/final_audit.json)
- [EBM/POSITIVE_BPL.md](../../../POSITIVE_BPL.md)
- [EBM/positive_bpl.py](../../../positive_bpl.py)

## 연결

다른 관련 가설: [반복 입력과 학습·추론의 상호작용](../../02_repetition/README.md) · [정상 다양성·도메인 지지와 정상쌍 학습](../../03_bonafide_diversity/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
