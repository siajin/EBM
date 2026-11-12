# E082. 2019 LA 학습 음성의 XLS-R 마지막 층 전체 특징 직접 저장

**실행 상태:** 실제 추출·수신 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

최종160차원 이전의 XLS-R시간특징을 직접보존하면 후속표현 분석을 재사용할 수 있다.

## 그렇게 설계한 이유

최종 160차원 요약이 아닌 마지막 Transformer의 시간 ×1,024 특징을 나중 분석용으로 저장한다.

대조와 통제의 근거: 초기서버 저장 안과 DF 추출 안은 실행 전 대체됐다. 실제 최종 저장은 노트북 전용이다. 과거 문서의‘대기’보다 최종 드라이브 표식이 우선한다.

## 모델·데이터·표본

2019LA 학습 25,380개. 고정 사전 학습 XLS-R, 앞 64,600샘플/부족하면 순환 반복,201 시간 프레임 ×1,024차원.

## 실험 내용과 상세 절차

1. 훈련 증강·무작위 crop·추가 평균 pooling·양자화 없이 각 음성을 원래 평가 입력으로 만든다.
2. 두 GPU가 각 12,690개를 float32 [12690,201,1024]로 추출한다. EBM 후처리 160차원은 계산 대상이 아니다.
3. 서버 메모리 →FIFO →SSH →사용자가 지정한 F: 외장 드라이브로 직접 전송한다. 서버에 hidden 배열 파일을 저장하지 않는다.
4. 원래 eager 출력 128개 QC·두 GPU 일치·블록 UID/bytes/SHA·수신 후 전체 파일 재읽기 검사를 완료 표 식으로 확인한다.

## 측정과 검증

총 shape[25380,201,1024]·dtypefloat32·rank 별 shape·파일 SHA·index 정렬·complete0.

## 결과

F:/IRLab/XLSR/19LA_train_last_20261105_01의 status/summary가 complete 이고 완료 표 식 0을 이번 문서화에서 읽어 확인했다. 특징 원본 배열 20.895GB는 이번에 다시 추출하거나 전체 재해시하지 않았다.

## 해석·실패한 부분·한계

특징 데이터 생성이지 새 검출기 성능 실험이 아니다. 160차원 h/z 캐시와 shape·역할을 혼동하지 않는다.

## 배운 점

데이터생성과 탐지기성능은 별개다. 배열shape·수신해시·완료표식이 필요하며 대용량특징은 Git에 포함하지 않는다.

## 실험 코드

- [artifacts/geometry_experiments/xlsr_last_layer_extract.py](code/artifacts/geometry_experiments/xlsr_last_layer_extract.py)
- [artifacts/geometry_experiments/receive_xlsr_last_layer.py](code/artifacts/geometry_experiments/receive_xlsr_last_layer.py)

코드의 파일별 원본 위치·SHA256은 [source_manifest.json](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [docs/experiments/19la_train_xlsr_last_20261105.md](../../_shared/references/docs/experiments/19la_train_xlsr_last_20261105.md)
- [artifacts/geometry_experiments/xlsr_last_layer_extract.py](../../_shared/workspace/artifacts/geometry_experiments/xlsr_last_layer_extract.py)
- [artifacts/geometry_experiments/receive_xlsr_last_layer.py](../../_shared/workspace/artifacts/geometry_experiments/receive_xlsr_last_layer.py)
- [artifacts/experiment_methods_audit_20261111/last_hidden_export_receipt.json](../../_shared/references/artifacts/experiment_methods_audit_20261111/last_hidden_export_receipt.json)

## 연결

다른 관련 가설: 주 가설에서만 분류한다.

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
