<a id="e062-정상-도메인-오류와-연결되는-실제-음향-특성-48개-측정"></a>

# 정상 도메인 오류와 연결되는 실제 음향 특성 48개 측정

**실행 상태:** 서버 원음 측정 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

음량·스펙트럼·음높이 등 실제 음향 통계가 정상 도메인 오류를 설명할 수 있다.

## 그렇게 설계한 이유

길이·출처 이외에 음량·스펙트럼·음높이 특성이 오류와 연결되는지 살핀다.

대조와 통제의 근거: 무음 시간·무음 비율 설명은 사용 자제 외 조건이다. 상대 에너지 gate는 음향 측정 안정화용이며 무음 분포 가설 검증이 아니다.

## 모델·데이터·표본

학습 자료 정상 화자별 26개 =520개. DF의 nocodec 정상 전체와 출처별 정상 진단 목록 합집합 1,699개, 출처별 위조 400개 씩 1,200개. 총 3,419개 원음.

## 실험 내용과 상세 절차

1. 메타데이터 ID 해시로 표본을 고정하고 원음을 서버에서만 읽는다.16kHz 모노·처음 최대 64,600샘플, 반복 전 신호를 측정한다.
2. 25ms(400샘플)창·10ms(160샘플)간격에서 RMS가 최대 프레임의 −25dB 이상인 프레임만 안정적인 음향 측정에 쓴다. 그 프레임 수나 비율은 원인 변수로 내보내지 않는다.
3. 음량 4개, 스펙트럼 12개, MFCC1~13의 평균/표준편차 26개, 음높이·주기성 4개, 시간 변화 2개를 측정한다. FFT512·Hann 창,80Hz 이상 정규화 스펙트럼과 24개 melfilter를 쓴다.
4. 음높이는 1,024샘플 창·320샘플 간격의 자기 상관 기반 정규화 차이에서 32~266샘플 지연을 찾는다. 첫 localtrough<0.15·주기성 ≥0.75·유효 3 프레임 이상 규칙을 쓴다.
5. 합성 100/200/400Hz 음높이·배율·스펙트럼 불변 검사와 실제 유한 값·누락음을 검수한다. 첫 pitch 추정 방식 오류는 수정한 v2 특징으로 완료했다.

## 측정과 검증

원음 header·특징 정의·source 별 분포·변형 전 점수와의 상관. 다음 [출처·길이를 통제한 음향 특징 블록별 정상 오류 예측](../E063/README.md)의 추가 오류 예측 입력으로 사용한다.

## 결과

3,419개 측정 완료. 주 분석 정상은 DFnocodec1,156개이며 48 음향 특징을 추가해도 출처·길이를 넘는 안정적 오류 예측 개선을 확인하지 못했다.

## 해석·실패한 부분·한계

음향 관측 연관으로 특정 마이크·악센트·음색을 유일한 원인으로 확정하지 않는다. 파형을 인위적으로 조작해 해당 속성만 분리한 실험은 아니다.

## 배운 점

48개 특징을 측정했지만 안정적인 추가 설명을 얻지 못했다. 측정 실패·제한도 결과의 일부로 남겨야 한다.

## 실험 코드

- [소스: normal_acoustic_features.py](code/artifacts/geometry_experiments/normal_acoustic_features.py)

코드의 파일별 원본 위치·SHA256은 [원본 파일 위치와 해시 목록](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [소스: normal_acoustic_features.py](../../_shared/workspace/artifacts/geometry_experiments/normal_acoustic_features.py)
- [설정·결과 근거](../../_shared/references/artifacts/geometry_cache/local_analysis/normal_cause_audit_v1/protocol.json)
- [설정·결과 근거](../../_shared/references/artifacts/geometry_cache/local_analysis/normal_cause_audit_v1/feature_QC_recovery.json)
- [설정·결과 근거](../../_shared/references/artifacts/geometry_cache/local_analysis/normal_cause_audit_v1/causal_conclusions.json)

## 연결

다른 관련 가설: 주 가설에서만 분류한다.

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
