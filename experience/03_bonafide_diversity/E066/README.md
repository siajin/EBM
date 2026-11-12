<a id="e066-df-자체-eer-부근-오류-꼬리와-길이최근접음향-특성-재검토"></a>

# DF 자체 EER 부근 오류 꼬리와 길이·최근접·음향 특성 재검토

**실행 상태:** 기존 점수·특징 재분석 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

19LA고정 임계값의 정상꼬리와 DF 자체EER을 만드는 오류는 다른 집단일 수 있다.

## 그렇게 설계한 이유

19LA 기준 고정 임계값 오류와 실제 DF EER을 만드는 어려운 정상을 구별한다.

대조와 통제의 근거: 정상만 골라 제외한 EER은 모델 개선이 아니다. 개발에서 고른 배포 임계값도 아니다.

## 모델·데이터·표본

배치 160 모델 DF 전체 533,928개 와 기존 음향·최근접캐시. 기존 19LA 임계값과 다른 사후 DF 임계값 −0.9145589682를 사용한다.

## 실험 내용과 상세 절차

1. DF 전체 ROC 에서 가까운 실현 가능 EER 운영 점을 정해 정상 오류 수를 센다.
2. 그 정상들의 원음 길이·출처·반복 필요 여부를 집계하고,2초 이상 정상만 남김/반복 불필요 정상만 남김의 구성 변경 EER을 계산한다.
3. 음향 특징이 있는 nocodec 오류 정상과 DF 진단 내 오류 정상에서 최근접 지지·음향 오류 예측을 별도로 반복한다.
4. 학습 자료 길이 분포 재가중과 화자 내 길이 상관, 기존 입력 개입의 같은 표본 EER도 분석한다.

## 측정과 검증

실제 EER 꼬리 정상 수·길이 분포·참조 최근 접·활동 제한 음향 모델의 표본 수·paired 처리 EER.

## 결과

DF 전체 EER 3.241828%, 정상 오류 482개 중 479개가 반복 필요·312개가 2초 미만·437개가 3초 미만이었다. nocodec 정상 오류는 36개 뿐이었다.

## 해석·실패한 부분·한계

원음 길이와 반복 패턴 원인은 이 관찰로 분리되지 않는다. 오류 수적은 음향 검정 의무 효과를 보편 부정으로 읽지 않는다.

## 배운 점

DF EER꼬리는 매우짧고 반복필요 집단에 집중됐다. 정상 일부를 제외한 EER을 모델 개선으로 부를 수 없다.

## 실험 코드

- [소스: analyze_normal_error_tail.py](code/artifacts/geometry_experiments/analyze_normal_error_tail.py)

코드의 파일별 원본 위치·SHA256은 [원본 파일 위치와 해시 목록](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [소스: analyze_normal_error_tail.py](../../_shared/workspace/artifacts/geometry_experiments/analyze_normal_error_tail.py)
- [설정·결과 근거](../../_shared/references/artifacts/geometry_cache/local_analysis/normal_cause_audit_v1/EER_tail_complete.json)
- [설정·결과 근거](../../_shared/references/artifacts/geometry_cache/local_analysis/normal_cause_audit_v1/actual_EER_tail_neighborhoods.json)

## 연결

다른 관련 가설: [길이·유효 발화량과 앞뒤 저에너지 편향](../../01_duration_bias/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
