# 정상 다양성·도메인 지지와 정상쌍 학습

## 연구 질문과 설계 배경

새 도메인 EER 악화가 위조 변화보다 정상 지지의 이동에 민감하고, 정상 다양성 확대가 해결의 단서인가?

점수 교차와 최근접 분석은 정상 도메인 문제를 지지한다. 정상 다양성을 독립 조작한 장기 통제학습은 없으므로 다양성 부족을 원인으로 확정하지 않는다.

## 실험별 상세 기록

각 항목에 가설, 설계 이유, 데이터·통제, 절차, 실제 결과, 실패·한계, 배운 점, 코드와 근거를 담았다. 검증·캐시생성·설계만인 항목을 독립 성능 실험으로 세지 않는다.

|실험|내용|실행 상태|
|---|---|---|
|[E001](E001/README.md)|배치 80·80epoch의 초기 반복 입력 학습|학습 완료|
|[E002](E002/README.md)|배치 160·50epoch·GPU 별 정상 쌍 학습|학습 완료|
|[E003](E003/README.md)|배치 160에서 두 GPU의 정상 특징을 공유하는 학습|학습 완료|
|[E004](E004/README.md)|전체 배치 320으로 늘린 실제 학습 시도|실패: 첫 배치 메모리 부족|
|[E008](E008/README.md)|기존 학습들의 배치·정상 쌍 수·학습 진행량 비교|기록 분석 완료|
|[E020](E020/README.md)|정상 두 묶음만 사용한 108회 양성쌍 손실의 단기 학습|단기 학습·진단 완료|
|[E060](E060/README.md)|정상 점수와 위조 점수의 데이터셋을 따로 교체하는 교차 평가|캐시교차 평가·재현 완료|
|[E061](E061/README.md)|정상 음성의 출처 차이가 코덱·반복 없는 부분에서도 남는 지 조사|기존 특징·점수 분석 완료|
|[E062](E062/README.md)|정상 도메인 오류와 연결되는 실제 음향 특성 48개 측정|서버 원음 측정 완료|
|[E063](E063/README.md)|출처·길이를 통제한 음향 특징 블록별 정상 오류 예측|추가 진단 분류기 교차검증 완료|
|[E064](E064/README.md)|정상 예시를 더 잘 대표하는 중심·이동·정규화 점수 7 종|정상만 이용한 기준 추정·평가 완료|
|[E065](E065/README.md)|틀린 정상 특징이 학습 위조 또는 평가 정상의 어디에 가까운지 조사|최근접 참조 분석 완료|
|[E066](E066/README.md)|DF 자체 EER 부근 오류 꼬리와 길이·최근접·음향 특성 재검토|기존 점수·특징 재분석 완료|
|[E086](E086/README.md)|이전 80epoch 모델의 정상 기준 각도와 160차원 분포|기존 실제 결과 확인·문서화 완료|
|[E089](E089/README.md)|정상 특징으로 여러 평균 기준 벡터를 만든 초기 실험|기존 실제 결과 확인·문서화 완료|
|[E090](E090/README.md)|낮은 정상 점수 꼬리의 출처·코덱·길이·음향·화자 진단|기존 실제 결과 확인·문서화 완료|
|[E096](E096/README.md)|단순 길이·저에너지·에너지·스펙트럼 단서를 읽는 분류기의 전이|기존 실제 결과 확인·문서화 완료|
|[E099](E099/README.md)|다른 출처·라벨의 저에너지 프레임을 이식한 실험|기존 실제 결과 확인·문서화 완료|
|[E107](E107/README.md)|고정 BPL 학습 모델의 실제 증강 쌍·저에너지 이식 반응 검토|기존 실제 결과 확인·문서화 완료|

## 다른 가설과 공유하는 근거

[E005](../02_repetition/E005/README.md) · [E006](../02_repetition/E006/README.md) · [E007](../01_duration_bias/E007/README.md) · [E009](../02_repetition/E009/README.md) · [E010](../02_repetition/E010/README.md) · [E011](../01_duration_bias/E011/README.md) · [E014](../02_repetition/E014/README.md) · [E015](../02_repetition/E015/README.md) · [E017](../01_duration_bias/E017/README.md) · [E021](../01_duration_bias/E021/README.md) · [E022](../01_duration_bias/E022/README.md) · [E023](../01_duration_bias/E023/README.md) · [E026](../01_duration_bias/E026/README.md) · [E028](../01_duration_bias/E028/README.md) · [E034](../01_duration_bias/E034/README.md) · [E035](../01_duration_bias/E035/README.md) · [E037](../01_duration_bias/E037/README.md) · [E043](../05_residual_geometry/E043/README.md) · [E046](../05_residual_geometry/E046/README.md) · [E048](../05_residual_geometry/E048/README.md) · [E050](../05_residual_geometry/E050/README.md) · [E051](../05_residual_geometry/E051/README.md) · [E052](../05_residual_geometry/E052/README.md) · [E053](../05_residual_geometry/E053/README.md) · [E055](../05_residual_geometry/E055/README.md) · [E059](../04_norm_information/E059/README.md) · [E067](../01_duration_bias/E067/README.md) · [E068](../05_residual_geometry/E068/README.md) · [E069](../05_residual_geometry/E069/README.md) · [E074](../05_residual_geometry/E074/README.md) · [E075](../05_residual_geometry/E075/README.md) · [E076](../05_residual_geometry/E076/README.md) · [E078](../01_duration_bias/E078/README.md) · [E085](../05_residual_geometry/E085/README.md) · [E087](../05_residual_geometry/E087/README.md) · [E093](../01_duration_bias/E093/README.md) · [E094](../01_duration_bias/E094/README.md) · [E097](../01_duration_bias/E097/README.md) · [E100](../01_duration_bias/E100/README.md) · [E103](../01_duration_bias/E103/README.md) · [E104](../01_duration_bias/E104/README.md) · [E105](../01_duration_bias/E105/README.md) · [E106](../01_duration_bias/E106/README.md) · [E108](../05_residual_geometry/E108/README.md) · [E109](../02_repetition/E109/README.md) · [E110](../01_duration_bias/E110/README.md) · [E111](../06_spoof_centered/E111/README.md) · [E112](../06_spoof_centered/E112/README.md) · [E113](../01_duration_bias/E113/README.md)

같은 checkpoint·음성·캐시를 재사용한 분석은 독립 학습이나 독립 표본으로 중복 합산하지 않는다.

[전체 목록](../README.md)
