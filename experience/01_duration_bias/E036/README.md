<a id="e036-구간만-직접-읽는-방식의-속도발화량원음-사용률-진단"></a>

# 구간만 직접 읽는 방식의 속도·발화량·원음 사용률 진단

**실행 상태:** 실제 파형·로더진단완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

선택 구간만 읽으면 파형을 바꾸지 않고 I/O 비용을 줄일 수 있으며 실제 정보 노출도 계량할 수 있다.

## 그렇게 설계한 이유

원음 전체를 해독한 뒤 자르는 비용을 줄이고 선택된 조각의 정보 사용량을 측정한다.

대조와 통제의 근거: 동일 원음의 겹친 구간을 시간 사용량에 중복 더 하지 않는다. CPU 읽기 속도를 전체 학습 속도로 부르지 않는다.

## 모델·데이터·표본

학습 자료 25,380개 와 실제 첫 epoch28,512 조각. 전체 학습 50epoch의 원음/구간 선택을 재생한다.

## 실험 내용과 상세 절차

1. SoundFile header 에서 원음 동일성을 확인한 뒤 seek로 선택 구간만 float64 해독한다.
2. 전체 해독 후 같은 구간을 자르는 원래 방식과 실제 raw512개·증강 64쌍을 bit-exact로 대조한다.
3. warm-cache 읽기 3회 와 실제 epoch1 선택 빈도로 I/O 시간을 비교한다.
4. 원음별 선택 구간의 합집합 시간을 원음 시간으로 나눠 사용률을 계산한다. VAD2로 증강 전 음성 프레임과 100ms 미만 발화 판정을 보조 분석한다.

## 측정과 검증

읽기 시간·파형 동일성·원음 ID 방문률·원음 시간 합집합 사용률·활동량.

## 결과

읽기 1.594배, 시간 37.3% 감소. 기존 복원 추출 50epoch 평균 원음 시간 사용률 OC 정상 63.28%·위조 64.03%, 정상 OC +BPL86.61%.0.5초 조각의 100ms 미만 판정은 정상 20.64%·위조 12.07%.

## 해석·실패한 부분·한계

짧은 조각 수가 늘어도 실제 발화 정보가 늘었다는 뜻은 아니다. VAD는 자동 추정이고 이 실행은 새 성능 학습을 하지 않았다.

## 배운 점

bit-exact 읽기 개선은 확인했지만 전체 학습 속도 향상과 다르다. 조각 수보다 원음 시간 합집합 사용률이 중요하다.

## 실험 코드

- [소스: positive_bpl.py](code/EBM/positive_bpl.py)
- [소스: analyze_train_crop_io.py](code/EBM/tools/analyze_train_crop_io.py)

코드의 파일별 원본 위치·SHA256은 [원본 파일 위치와 해시 목록](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [설정·결과 근거](../../_shared/references/artifacts/train_crop_io_activity_20261110/results/analysis.json)
- [설정·결과 근거](../../_shared/references/artifacts/train_crop_io_activity_20261110/verified_summary.json)
- [설정·결과 근거](../../_shared/references/artifacts/train_crop_io_activity_20261110/state.json)
- [소스: positive_bpl.py](../../../positive_bpl.py)

## 연결

다른 관련 가설: [반복 입력과 학습·추론의 상호작용](../../02_repetition/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
