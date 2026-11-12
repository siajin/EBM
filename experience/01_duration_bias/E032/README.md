<a id="e032-여러-목표-길이의-앞뒤-구간-생성과-원음-우선-추출의-노출-분포"></a>

# 여러 목표 길이의 앞·뒤 구간 생성과 원음 우선 추출의 노출 분포

**실행 상태:** 메타데이터 기대 분포 계산 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

원음 먼저 균등 선택하면 조각 균등 선택의 노출 편향을 바꿀 수 있다.

## 그렇게 설계한 이유

짧은 조각이 많아지는 이유와 조각 균 등 추출/원음 균 등 추출의 차이를 측정한다.

대조와 통제의 근거: 같은 원음의 서로 다른 길이 구간은 중복 내용이므로 독립 새 음성으로 세지 않는다. 이 조각 카운트 버전과 이후 VAD 후보 중복 제거 버전도 다르다.

## 모델·데이터·표본

[학습·개발·평가 자료의 원음 길이와 공격별 클래스 구성](../E028/README.md) 원음 전체, 각 자료·클래스를 따로 분석. 원음 파일을 새로 만들지 않는다.

## 실험 내용과 상세 절차

1. 목표 길이 0.5초 간격마다 한 원음에서 floor(D/L)개의 앞 기준 비중복 구간을 센다.
2. 양방향 버전은 뒤 기준 floor(D/L)개를 더해 2floor(D/L)로 센다. 뒤 기준은 파형 역전이 아니며 반대 쪽 남는 꼬리를 덮는다.
3. 앞/뒤에서 같은 좌표가 나오더라도 이 버전에서는 중복을 유지한다.0.5초 미만 원음은 통째로 한 번 보존한다.
4. 모든 조각을 균등 선택하는 분포와, 클래스 안의 원음을 먼저 균등 선택한 뒤 그 원음의 후보 조각을 균등 선택하는 분포를 정확 기대 확률로 계산한다.

## 측정과 검증

길이별 생성 조각 수·2초 미만 비율·평균 선택 길이·원음/화자/출처비율이동.

## 결과

학습 자료 정상은 조각 균 등에서 2초 미만 75.0899%, 원음 우선에서 78.0081%였다. 원음 균 등으로 바꿔도 길이를 균등하게 고른 것이 아니다.

## 해석·실패한 부분·한계

메타데이터 상의 노출 분포다. 이 조각을 실제 학습한 효과를 직접 측정한 값은 아니다.

## 배운 점

원음 균등이어도 짧은 후보가 많아 길이가 균등해지지는 않는다. 후보 수와 추출 확률을 따로 설계해야 한다.

## 실험 코드

- [소스: bidirectional_audio_segments.py](code/EBM/tools/bidirectional_audio_segments.py)
- [소스: analyze_segment_length_histograms.py](code/EBM/tools/analyze_segment_length_histograms.py)
- [소스: analyze_original_first_length_sampling.py](code/EBM/tools/analyze_original_first_length_sampling.py)
- [소스: analyze_segment_group_shifts.py](code/EBM/tools/analyze_segment_group_shifts.py)

코드의 파일별 원본 위치·SHA256은 [원본 파일 위치와 해시 목록](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [설정·결과 근거](../../_shared/references/artifacts/segment_length_histograms_20261109/analysis.json)
- [설정·결과 근거](../../_shared/references/artifacts/segment_length_histograms_bidirectional_20261109/analysis.json)
- [설정·결과 근거](../../_shared/references/artifacts/original_first_length_sampling_20261110/analysis.json)
- [설정·결과 근거](../../_shared/references/artifacts/segment_group_shifts_20261110/analysis.json)

## 연결

다른 관련 가설: 주 가설에서만 분류한다.

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
