# 코사인 외 norm과 정규화 전 통계

## 연구 질문과 설계 배경

코사인이 같은 경우에도 norm·정규화 전 평균·스케일이 판별 신호를 갖는가?

내부 판독에서 정규화 전 특징의 이득이 관측됐지만 Train→DF 전이는 대부분 악화했다. LayerNorm의 비가역적 정보 손실을 증명한 것은 아니다.

## 실험별 상세 기록

각 항목에 가설, 설계 이유, 데이터·통제, 절차, 실제 결과, 실패·한계, 배운 점, 코드와 근거를 담았다. 검증·캐시생성·설계만인 항목을 독립 성능 실험으로 세지 않는다.

|실험|내용|실행 상태|
|---|---|---|
|[E040](E040/README.md)|초기 개발 자료에서 점수·크기·옆 방향 특징을 읽는 분류기 비교|추가 분류기 학습·평가 완료|
|[E041](E041/README.md)|초기 학습 자료 분류기와 선택 좌표 분류기의 DF 전이 평가|추가 분류기·전이 평가 완료|
|[E045](E045/README.md)|LayerNorm 전후 특징·평균·표준편차를 읽는 내부 분류기|추가 분류기·내부 교차검증 완료|
|[E059](E059/README.md)|기존 OC gradient·방향 정규화·정규화 전 특징 복원·변형 반응의 네 후속 분석|캐시 계산·검산 완료|

## 다른 가설과 공유하는 근거

[E043](../05_residual_geometry/E043/README.md) · [E049](../05_residual_geometry/E049/README.md) · [E050](../05_residual_geometry/E050/README.md) · [E051](../05_residual_geometry/E051/README.md) · [E052](../05_residual_geometry/E052/README.md) · [E053](../05_residual_geometry/E053/README.md) · [E054](../05_residual_geometry/E054/README.md) · [E055](../05_residual_geometry/E055/README.md) · [E057](../05_residual_geometry/E057/README.md) · [E058](../05_residual_geometry/E058/README.md) · [E072](../02_repetition/E072/README.md) · [E074](../05_residual_geometry/E074/README.md) · [E075](../05_residual_geometry/E075/README.md) · [E076](../05_residual_geometry/E076/README.md) · [E087](../05_residual_geometry/E087/README.md) · [E095](../02_repetition/E095/README.md) · [E103](../01_duration_bias/E103/README.md) · [E104](../01_duration_bias/E104/README.md) · [E108](../05_residual_geometry/E108/README.md)

같은 checkpoint·음성·캐시를 재사용한 분석은 독립 학습이나 독립 표본으로 중복 합산하지 않는다.

[전체 목록](../README.md)
