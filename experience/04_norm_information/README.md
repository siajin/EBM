# 코사인 외 norm과 정규화 전 통계

## 연구 질문과 설계 배경

코사인이 같은 경우에도 norm·정규화 전 평균·스케일이 판별 신호를 갖는가?

내부 판독에서 정규화 전 특징의 이득이 관측됐지만 Train→DF 전이는 대부분 악화했다. LayerNorm의 비가역적 정보 손실을 증명한 것은 아니다.

## 실험별 상세 기록

각 항목에 가설, 설계 이유, 데이터·통제, 절차, 실제 결과, 실패·한계, 배운 점, 코드와 근거를 담았다. 검증·캐시생성·설계만인 항목을 독립 성능 실험으로 세지 않는다.

|실험|실행 상태|
|---|---|
|[초기 개발 자료에서 점수·크기·옆 방향 특징을 읽는 분류기 비교](E040/README.md)|추가 분류기 학습·평가 완료|
|[초기 학습 자료 분류기와 선택 좌표 분류기의 DF 전이 평가](E041/README.md)|추가 분류기·전이 평가 완료|
|[LayerNorm 전후 특징·평균·표준편차를 읽는 내부 분류기](E045/README.md)|추가 분류기·내부 교차검증 완료|
|[기존 OC gradient·방향 정규화·정규화 전 특징 복원·변형 반응의 네 후속 분석](E059/README.md)|캐시 계산·검산 완료|

## 다른 가설과 공유하는 근거

[DF 안에서 점수·크기를 통제한 잔여 방향의 판별 정보](../05_residual_geometry/E043/README.md) · [학습 정상 공분산으로 거리 방향을 바꾸되 정상 기준 벡터는 고정](../05_residual_geometry/E049/README.md) · [방향 벡터 전체의 조건별 분산과 길이·패딩 연관](../05_residual_geometry/E050/README.md) · [같은 정상 기준 각도에서 손실이 구별하지 않는 방향의 수학 검사](../05_residual_geometry/E051/README.md) · [고정 특징 위에서 정적·입력별 좌표 가중치와 같은 크기 신경망 학습](../05_residual_geometry/E052/README.md) · [정상·위조 라벨로 대각 거리 가중치만 학습하는 대조](../05_residual_geometry/E053/README.md) · [원래 점수에 단조인 기본 식과 방향/잡음 잔차의 학습 자료 →평가 전이](../05_residual_geometry/E054/README.md) · [두 평가 자료 내부의 방향·정규화 전 특징·같은 차원 잡음 대조 9 종](../05_residual_geometry/E055/README.md) · [기존 모델의 층 가중치·최종 시간 가중치를 균등하게 바꾸는 개입](../05_residual_geometry/E057/README.md) · [같은 음성의 백색잡음 20/10dB와 Opus32kbps 재압축 개입](../05_residual_geometry/E058/README.md) · [특징 복제의 표준편차 배율과 평균·표준편차 경로 교체](../02_repetition/E072/README.md) · [틀린 정상의 옆 방향이 학습 정상 주축에 집중되는지 분석](../05_residual_geometry/E074/README.md) · [학습 정상이 드물게 차지하는 방향 축부터 통째로 제거](../05_residual_geometry/E075/README.md) · [저점유 방향 제거 152~158개를 한 축씩 늘리는 정밀 비교](../05_residual_geometry/E076/README.md) · [특징 추출 단계별 분리와 같은 원음의 긴·짧은 특징 비교](../05_residual_geometry/E087/README.md) · [이전 모델에서 특징 복제와 통계 요약의 길이 배율을 분리](../02_repetition/E095/README.md) · [자연적인 앞 저에너지 길이를 읽는 특징 방향: −40dB 정의](../01_duration_bias/E103/README.md) · [자연적인 앞 저에너지 길이를 읽는 특징 방향: −20dB와 안정성](../01_duration_bias/E104/README.md) · [저에너지 시간 위치와 나머지 XLS-R 특징을 교체·제외한 내부 개입](../05_residual_geometry/E108/README.md)

같은 checkpoint·음성·캐시를 재사용한 분석은 독립 학습이나 독립 표본으로 중복 합산하지 않는다.

[전체 목록](../README.md)
