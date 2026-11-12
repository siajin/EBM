# 잔여 방향·좌표·거리·기하와 변형 반응

## 연구 질문과 설계 배경

단일 정상 중심과의 각도 밖에 판별 신호가 남고, 이를 읽거나 조절해 전이 성능을 높일 수 있는가?

내부CV에서 방향 신호는 남았지만 고정 후처리의 독립 전이 이득은 확립하지 못했다. 좌표·회전·매칭 지지·관찰 밀도와 수학적 자유도를 구분한다.

## 실험별 상세 기록

각 항목에 가설, 설계 이유, 데이터·통제, 절차, 실제 결과, 실패·한계, 배운 점, 코드와 근거를 담았다. 검증·캐시생성·설계만인 항목을 독립 성능 실험으로 세지 않는다.

|실험|내용|실행 상태|
|---|---|---|
|[E039](E039/README.md)|고정 모델의 정규화 전후 특징 추출·전체 평가 캐시 구축|추출·검증 완료|
|[E042](E042/README.md)|원래 코사인을 유지한 방향 잔차와 공분산 거리의 전이 대조|추가 분류기·캐시 평가 완료|
|[E043](E043/README.md)|DF 안에서 점수·크기를 통제한 잔여 방향의 판별 정보|추가 분류기·내부 교차검증 완료|
|[E044](E044/README.md)|기존 코사인에서 좌표를 제거하거나 1~80개만 남기는 실험|캐시 점수 개입 완료|
|[E046](E046/README.md)|정상 기준 밖의 방향 군집과 출처·코덱·길이의 연관|캐시 군집 분석 완료|
|[E047](E047/README.md)|좌표 선택의 전체 169조건과 회전 좌표계 대조|캐시 점수 개입 완료|
|[E048](E048/README.md)|같은 화자·코덱·점수·크기의 정상/위조 방향 분포 매칭|계산 완료·표본 부족으로 주가 설 판정 불가|
|[E049](E049/README.md)|학습 정상 공분산으로 거리 방향을 바꾸되 정상 기준 벡터는 고정|캐시거리 평가 완료|
|[E050](E050/README.md)|방향 벡터 전체의 조건별 분산과 길이·패딩 연관|캐시 통계 완료|
|[E051](E051/README.md)|같은 정상 기준 각도에서 손실이 구별하지 않는 방향의 수학 검사|합성 수치·미분 검사 완료|
|[E052](E052/README.md)|고정 특징 위에서 정적·입력별 좌표 가중치와 같은 크기 신경망 학습|추가 모듈 학습·전이 평가 완료|
|[E053](E053/README.md)|정상·위조 라벨로 대각 거리 가중치만 학습하는 대조|추가 모듈 학습·전이 평가 완료|
|[E054](E054/README.md)|원래 점수에 단조인 기본 식과 방향/잡음 잔차의 학습 자료 →평가 전이|추가 분류기 학습·전이 평가 완료|
|[E055](E055/README.md)|두 평가 자료 내부의 방향·정규화 전 특징·같은 차원 잡음 대조 9 종|추가 분류기·내부 교차검증 완료|
|[E056](E056/README.md)|개발에서 고른 고정 후처리를 전체 자료로 확장 평가|추출·적용 평가 완료|
|[E057](E057/README.md)|기존 모델의 층 가중치·최종 시간 가중치를 균등하게 바꾸는 개입|실제 추론 완료|
|[E058](E058/README.md)|같은 음성의 백색잡음 20/10dB와 Opus32kbps 재압축 개입|실제 추론·전처리 검수 완료|
|[E068](E068/README.md)|위조 특징의 주축과 두 위조 사이 연속 경로의 정상 최근접 진입|기하 분석·연속 검산 완료|
|[E069](E069/README.md)|위조 연결 경로 주변의 정상·위조 밀도와 빈 구간|기하 밀도 분석 완료|
|[E070](E070/README.md)|같은 변형 전 점수 표본의 전체 가능 매칭과 이미 저장된 변형 반응의 범위|기존캐시매칭·산술 검증 완료|
|[E074](E074/README.md)|틀린 정상의 옆 방향이 학습 정상 주축에 집중되는지 분석|기존캐시 통계·독립 검수 완료|
|[E075](E075/README.md)|학습 정상이 드물게 차지하는 방향 축부터 통째로 제거|기존캐시 점수개입·독립 검수 완료|
|[E076](E076/README.md)|저점유 방향 제거 152~158개를 한 축씩 늘리는 정밀 비교|기존캐시 점수개입·독립 검수 완료|
|[E082](E082/README.md)|2019 LA 학습 음성의 XLS-R 마지막 층 전체 특징 직접 저장|실제 추출·수신 완료|
|[E083](E083/README.md)|원래 160차원 관계를 보존하거나 왜곡을 밝힌 2차원·구면 시각화|보조 기하 분석·그림 검증|
|[E085](E085/README.md)|기존 변형 결과에서 정상 안정성과 위조의 정상 방향 이동을 따로 검토|기존 반응 점수의 탐색적 재분석 완료|
|[E087](E087/README.md)|특징 추출 단계별 분리와 같은 원음의 긴·짧은 특징 비교|기존 실제 결과 확인·문서화 완료|
|[E108](E108/README.md)|저에너지 시간 위치와 나머지 XLS-R 특징을 교체·제외한 내부 개입|기존 실제 결과 확인·문서화 완료|

## 다른 가설과 공유하는 근거

[E012](../02_repetition/E012/README.md) · [E016](../02_repetition/E016/README.md) · [E020](../03_bonafide_diversity/E020/README.md) · [E026](../01_duration_bias/E026/README.md) · [E040](../04_norm_information/E040/README.md) · [E041](../04_norm_information/E041/README.md) · [E045](../04_norm_information/E045/README.md) · [E059](../04_norm_information/E059/README.md) · [E060](../03_bonafide_diversity/E060/README.md) · [E061](../03_bonafide_diversity/E061/README.md) · [E064](../03_bonafide_diversity/E064/README.md) · [E065](../03_bonafide_diversity/E065/README.md) · [E067](../01_duration_bias/E067/README.md) · [E071](../02_repetition/E071/README.md) · [E072](../02_repetition/E072/README.md) · [E073](../02_repetition/E073/README.md) · [E078](../01_duration_bias/E078/README.md) · [E079](../01_duration_bias/E079/README.md) · [E080](../01_duration_bias/E080/README.md) · [E086](../03_bonafide_diversity/E086/README.md) · [E089](../03_bonafide_diversity/E089/README.md) · [E090](../03_bonafide_diversity/E090/README.md) · [E094](../01_duration_bias/E094/README.md) · [E095](../02_repetition/E095/README.md) · [E097](../01_duration_bias/E097/README.md) · [E098](../01_duration_bias/E098/README.md) · [E099](../03_bonafide_diversity/E099/README.md) · [E100](../01_duration_bias/E100/README.md) · [E101](../01_duration_bias/E101/README.md) · [E102](../01_duration_bias/E102/README.md) · [E103](../01_duration_bias/E103/README.md) · [E104](../01_duration_bias/E104/README.md) · [E107](../03_bonafide_diversity/E107/README.md) · [E109](../02_repetition/E109/README.md) · [E111](../06_spoof_centered/E111/README.md) · [E112](../06_spoof_centered/E112/README.md)

같은 checkpoint·음성·캐시를 재사용한 분석은 독립 학습이나 독립 표본으로 중복 합산하지 않는다.

[전체 목록](../README.md)
