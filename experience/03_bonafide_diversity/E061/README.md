<a id="e061-정상-음성의-출처-차이가-코덱반복-없는-부분에서도-남는-지-조사"></a>

# 정상 음성의 출처 차이가 코덱·반복 없는 부분에서도 남는 지 조사

**실행 상태:** 기존 특징·점수 분석 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

코덱·반복 불필요 조건에서도 정상 출처 간 오류 차이가 남을 수 있다.

## 그렇게 설계한 이유

DF 정상 취약성이 단순 추가 코덱이나 반복 필요 여부만으로 없어지는지 본다.

대조와 통제의 근거: nocodec은 평가 metadata의 조건이며 원천 녹음에 압축이 전혀 없었다는 뜻이 아니다. 제한 집단의 화자·내용·길이도 바뀐다.

## 모델·데이터·표본

배치 160 고정 모델의 DF 정상 14,869개 와 보존된 개발/19LA 점수. 원래 최종 h/z와 학습 방향을 사용한다.

## 실험 내용과 상세 절차

1. 19LA 에서 고정된 진단 임계값 0.8619597055730815로 정상 오류를 정의한다.
2. 전체·nocodec·원음 ≥4.0375초로 반복 불필요·이들의 교집합 등 제한 집단을 만들고 출처별 오류율·화자별 범위를 계산한다.
3. 코덱 × 길이 구간에 공통 지지가 있는 셀에서 출처 분포를 표준화하고 coverage를 보고한다.
4. LayerNorm 전후 특징·학습 정상/위조 방향과의 거리도 비교한다. 원음의 무음 비율을 원인 특징으로 사용하지 않는다.

## 측정과 검증

출처별 정상 거부율·화자 군집 구간·제한 전후 분모·표현 기하.

## 결과

전체 정상 거부는 ASVspoof0.5668%, VCC201865.6085%, VCC202037.5556%. nocodec 에서도 0.2591/60.7143/30.8571%로 출처 차이가 남았다.

## 해석·실패한 부분·한계

출처 자체가 마이크나 발성의 인과 변수로 식별된 것은 아니다. 이후 DF 자체 EER 임계값 분석과 이 고정 19LA 임계값 오류를 구별한다.

## 배운 점

단순 코덱이나반복 여부만으로 출처 차이를 설명하지 못했다. 출처를 특정 마이크 원인으로 확정할 수 없다.

## 실험 코드

- [소스: diagnose_normal_source_gap.py](code/artifacts/geometry_experiments/diagnose_normal_source_gap.py)

코드의 파일별 원본 위치·SHA256은 [원본 파일 위치와 해시 목록](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [설정·결과 근거](../../_shared/references/artifacts/geometry_cache/local_analysis/normal_source_mechanisms_v1/protocol.json)
- [설정·결과 근거](../../_shared/references/artifacts/geometry_cache/local_analysis/normal_source_mechanisms_v1/summary.json)
- [소스: diagnose_normal_source_gap.py](../../_shared/workspace/artifacts/geometry_experiments/diagnose_normal_source_gap.py)

## 연결

다른 관련 가설: [길이·유효 발화량과 앞뒤 저에너지 편향](../../01_duration_bias/README.md) · [잔여 방향·좌표·거리·기하와 변형 반응](../../05_residual_geometry/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
