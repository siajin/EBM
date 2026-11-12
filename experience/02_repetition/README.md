# 반복 입력과 학습·추론의 상호작용

## 연구 질문과 설계 배경

순환 반복이 모델에 내용 외의 문맥·반복량 단서를 만들며 학습과 추론의 일치가 성능에 영향을 주는가?

기존240은 반복 입력, B는 비반복 입력에서 유리했다. 반복의 보편적 해로움은 입증되지 않았다. 파형반복과 후기 특징복제는 다른 결과를 냈다.

## 실험별 상세 기록

각 항목에 가설, 설계 이유, 데이터·통제, 절차, 실제 결과, 실패·한계, 배운 점, 코드와 근거를 담았다. 검증·캐시생성·설계만인 항목을 독립 성능 실험으로 세지 않는다.

|실험|내용|실행 상태|
|---|---|---|
|[E005](E005/README.md)|배치 240·원래 반복 입력·GPU 별 정상 쌍 학습|학습 완료|
|[E006](E006/README.md)|학습 길이 상한을 유지하면서 짧은 음성 반복을 제거한 학습(B)|학습 50epoch 완료|
|[E009](E009/README.md)|두 고정 50epoch 모델을 같은 비반복 입력으로 전체 평가|평가 완료|
|[E010](E010/README.md)|두 반복 제거 학습 모델의 모든 저장 epoch 평가와 최소 EER 비교|평가 완료|
|[E012](E012/README.md)|같은 768개 원음의 반복 여부를 두 모델에 교차 적용|추론·분석 완료|
|[E013](E013/README.md)|원래 배치 240 모델의 반복 구간과 오류 분포 조사|기존 기록 분석 완료|
|[E014](E014/README.md)|개발 전체와 21LA 전체에서 반복/비반복 입력을 직접 교차 평가|추론·독립 검산 완료|
|[E015](E015/README.md)|자연적으로 2초 이하인 음성에서 오류 전이·임계값 이동·공격별 회복 분석|E014 저장 점수 분석 완료|
|[E016](E016/README.md)|긴 원음의 같은 1초/2초 구간에서 반복량·접합부·반사대조|추론·분석 완료|
|[E071](E071/README.md)|출처별 같은 원음의 순환 반복·반사채움·비반복과 긴 원음 절단 대조|실제 추론·독립 검수 완료|
|[E072](E072/README.md)|특징 복제의 표준편차 배율과 평균·표준편차 경로 교체|실제 내부 개입·수학 검수 완료|
|[E073](E073/README.md)|파형을 먼저 반복하는 경우와 추출한 특징을 뒤에서 반복하는 경우의 8조건|실제 내부 개입·검산 완료|
|[E081](E081/README.md)|네 모델·전체 체크포인트의 반복/비반복 공통 평가 준비와 실행 전 취소|검사·분석 재현만 완료; 추가 GPU 평가 취소|
|[E091](E091/README.md)|파형 크기·반복·0 채움·반사·가장자리 절단의 초기 교차 개입|기존 실제 결과 확인·문서화 완료|
|[E092](E092/README.md)|저에너지 가장자리 제거 후 반복·0 채움 위치를 달리한 대조|기존 실제 결과 확인·문서화 완료|
|[E095](E095/README.md)|이전 모델에서 특징 복제와 통계 요약의 길이 배율을 분리|기존 실제 결과 확인·문서화 완료|
|[E109](E109/README.md)|같은 저에너지 내부 개입을 다섯 저장 모델에 반복한 대조|기존 실제 결과 확인·문서화 완료|

## 다른 가설과 공유하는 근거

[E001](../03_bonafide_diversity/E001/README.md) · [E002](../03_bonafide_diversity/E002/README.md) · [E003](../03_bonafide_diversity/E003/README.md) · [E008](../03_bonafide_diversity/E008/README.md) · [E020](../03_bonafide_diversity/E020/README.md) · [E021](../01_duration_bias/E021/README.md) · [E022](../01_duration_bias/E022/README.md) · [E023](../01_duration_bias/E023/README.md) · [E024](../01_duration_bias/E024/README.md) · [E025](../01_duration_bias/E025/README.md) · [E026](../01_duration_bias/E026/README.md) · [E027](../01_duration_bias/E027/README.md) · [E028](../01_duration_bias/E028/README.md) · [E035](../01_duration_bias/E035/README.md) · [E036](../01_duration_bias/E036/README.md) · [E037](../01_duration_bias/E037/README.md) · [E038](../01_duration_bias/E038/README.md) · [E050](../05_residual_geometry/E050/README.md) · [E067](../01_duration_bias/E067/README.md) · [E084](../01_duration_bias/E084/README.md) · [E088](../01_duration_bias/E088/README.md) · [E093](../01_duration_bias/E093/README.md) · [E094](../01_duration_bias/E094/README.md) · [E097](../01_duration_bias/E097/README.md) · [E098](../01_duration_bias/E098/README.md) · [E100](../01_duration_bias/E100/README.md) · [E101](../01_duration_bias/E101/README.md) · [E102](../01_duration_bias/E102/README.md) · [E105](../01_duration_bias/E105/README.md) · [E106](../01_duration_bias/E106/README.md) · [E107](../03_bonafide_diversity/E107/README.md) · [E108](../05_residual_geometry/E108/README.md) · [E110](../01_duration_bias/E110/README.md) · [E113](../01_duration_bias/E113/README.md)

같은 checkpoint·음성·캐시를 재사용한 분석은 독립 학습이나 독립 표본으로 중복 합산하지 않는다.

[전체 목록](../README.md)
