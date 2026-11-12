# 정상 다양성·도메인 지지와 정상쌍 학습

## 연구 질문과 설계 배경

새 도메인 EER 악화가 위조 변화보다 정상 지지의 이동에 민감하고, 정상 다양성 확대가 해결의 단서인가?

점수 교차와 최근접 분석은 정상 도메인 문제를 지지한다. 정상 다양성을 독립 조작한 장기 통제학습은 없으므로 다양성 부족을 원인으로 확정하지 않는다.

## 실험별 상세 기록

각 항목에 가설, 설계 이유, 데이터·통제, 절차, 실제 결과, 실패·한계, 배운 점, 코드와 근거를 담았다. 검증·캐시생성·설계만인 항목을 독립 성능 실험으로 세지 않는다.

|실험|실행 상태|
|---|---|
|[배치 80·80epoch의 초기 반복 입력 학습](E001/README.md)|학습 완료|
|[배치 160·50epoch·GPU 별 정상 쌍 학습](E002/README.md)|학습 완료|
|[배치 160에서 두 GPU의 정상 특징을 공유하는 학습](E003/README.md)|학습 완료|
|[전체 배치 320으로 늘린 실제 학습 시도](E004/README.md)|실패: 첫 배치 메모리 부족|
|[기존 학습들의 배치·정상 쌍 수·학습 진행량 비교](E008/README.md)|기록 분석 완료|
|[정상 두 묶음만 사용한 108회 양성쌍 손실의 단기 학습](E020/README.md)|단기 학습·진단 완료|
|[정상 점수와 위조 점수의 데이터셋을 따로 교체하는 교차 평가](E060/README.md)|캐시교차 평가·재현 완료|
|[정상 음성의 출처 차이가 코덱·반복 없는 부분에서도 남는 지 조사](E061/README.md)|기존 특징·점수 분석 완료|
|[정상 도메인 오류와 연결되는 실제 음향 특성 48개 측정](E062/README.md)|서버 원음 측정 완료|
|[출처·길이를 통제한 음향 특징 블록별 정상 오류 예측](E063/README.md)|추가 진단 분류기 교차검증 완료|
|[정상 예시를 더 잘 대표하는 중심·이동·정규화 점수 7 종](E064/README.md)|정상만 이용한 기준 추정·평가 완료|
|[틀린 정상 특징이 학습 위조 또는 평가 정상의 어디에 가까운지 조사](E065/README.md)|최근접 참조 분석 완료|
|[DF 자체 EER 부근 오류 꼬리와 길이·최근접·음향 특성 재검토](E066/README.md)|기존 점수·특징 재분석 완료|
|[이전 80epoch 모델의 정상 기준 각도와 160차원 분포](E086/README.md)|기존 실제 결과 확인·문서화 완료|
|[정상 특징으로 여러 평균 기준 벡터를 만든 초기 실험](E089/README.md)|기존 실제 결과 확인·문서화 완료|
|[낮은 정상 점수 꼬리의 출처·코덱·길이·음향·화자 진단](E090/README.md)|기존 실제 결과 확인·문서화 완료|
|[단순 길이·저에너지·에너지·스펙트럼 단서를 읽는 분류기의 전이](E096/README.md)|기존 실제 결과 확인·문서화 완료|
|[다른 출처·라벨의 저에너지 프레임을 이식한 실험](E099/README.md)|기존 실제 결과 확인·문서화 완료|
|[고정 BPL 학습 모델의 실제 증강 쌍·저에너지 이식 반응 검토](E107/README.md)|기존 실제 결과 확인·문서화 완료|

## 다른 가설과 공유하는 근거

[배치 240·원래 반복 입력·GPU 별 정상 쌍 학습](../02_repetition/E005/README.md) · [학습 길이 상한을 유지하면서 짧은 음성 반복을 제거한 학습](../02_repetition/E006/README.md) · [반복 없는 학습에서 긴 입력의 상한만 제거한 학습](../01_duration_bias/E007/README.md) · [두 고정 50epoch 모델을 같은 비반복 입력으로 전체 평가](../02_repetition/E009/README.md) · [두 반복 제거 학습 모델의 모든 저장 epoch 평가와 최소 EER 비교](../02_repetition/E010/README.md) · [원음 길이별 오류와 임계값 민감도 분석](../01_duration_bias/E011/README.md) · [개발 전체와 21LA 전체에서 반복/비반복 입력을 직접 교차 평가](../02_repetition/E014/README.md) · [자연적으로 2초 이하인 음성에서 오류 전이·임계값 이동·공격별 회복 분석](../02_repetition/E015/README.md) · [클래스별 길이 분포를 맞추되 공격·코덱 비율을 보존한 EER](../01_duration_bias/E017/README.md) · [정상·위조 길이와 음성활동을 맞추고 독립 정상 묶음을 정렬한 학습](../01_duration_bias/E021/README.md) · [중단한 가변 길이 학습의 저장 20epoch를 21LA·DF 전체 평가](../01_duration_bias/E022/README.md) · [가변 길이 학습과 기존 모델의 21LA 길이·공격별 오류 재분석](../01_duration_bias/E023/README.md) · [원음 2초 이하 정상에만 3.0275초·2초 손실을 더한 EBM-Repeat 학습](../01_duration_bias/E026/README.md) · [학습·개발·평가 자료의 원음 길이와 공격별 클래스 구성](../01_duration_bias/E028/README.md) · [140개 클래스·화자·공격 그룹의 0.5~4초 공동 길이 균형](../01_duration_bias/E034/README.md) · [목표 길이 안에서 정상·위조의 음성활동 비율까지 맞추는 정책](../01_duration_bias/E035/README.md) · [매 epoch 모든 원음을 사용하면서 기존 기대 선택 비율을 유지하는 샘플러](../01_duration_bias/E037/README.md) · [DF 안에서 점수·크기를 통제한 잔여 방향의 판별 정보](../05_residual_geometry/E043/README.md) · [정상 기준 밖의 방향 군집과 출처·코덱·길이의 연관](../05_residual_geometry/E046/README.md) · [같은 화자·코덱·점수·크기의 정상/위조 방향 분포 매칭](../05_residual_geometry/E048/README.md) · [방향 벡터 전체의 조건별 분산과 길이·패딩 연관](../05_residual_geometry/E050/README.md) · [같은 정상 기준 각도에서 손실이 구별하지 않는 방향의 수학 검사](../05_residual_geometry/E051/README.md) · [고정 특징 위에서 정적·입력별 좌표 가중치와 같은 크기 신경망 학습](../05_residual_geometry/E052/README.md) · [정상·위조 라벨로 대각 거리 가중치만 학습하는 대조](../05_residual_geometry/E053/README.md) · [두 평가 자료 내부의 방향·정규화 전 특징·같은 차원 잡음 대조 9 종](../05_residual_geometry/E055/README.md) · [기존 OC gradient·방향 정규화·정규화 전 특징 복원·변형 반응의 네 후속 분석](../04_norm_information/E059/README.md) · [원음 길이별 정상 평균 기준과 점수 보정·무작위 중심 대조](../01_duration_bias/E067/README.md) · [위조 특징의 주축과 두 위조 사이 연속 경로의 정상 최근접 진입](../05_residual_geometry/E068/README.md) · [위조 연결 경로 주변의 정상·위조 밀도와 빈 구간](../05_residual_geometry/E069/README.md) · [틀린 정상의 옆 방향이 학습 정상 주축에 집중되는지 분석](../05_residual_geometry/E074/README.md) · [학습 정상이 드물게 차지하는 방향 축부터 통째로 제거](../05_residual_geometry/E075/README.md) · [저점유 방향 제거 152~158개를 한 축씩 늘리는 정밀 비교](../05_residual_geometry/E076/README.md) · [화자가 다른 정상 발화 정렬과 같은 원음의 보수적 길이 일관성 결합 안](../01_duration_bias/E078/README.md) · [기존 변형 결과에서 정상 안정성과 위조의 정상 방향 이동을 따로 검토](../05_residual_geometry/E085/README.md) · [특징 추출 단계별 분리와 같은 원음의 긴·짧은 특징 비교](../05_residual_geometry/E087/README.md) · [학습 원음의 저에너지·음량 분포와 클래스 단서의 초기 조사](../01_duration_bias/E093/README.md) · [저에너지 구간의 길이·에너지를 유지한 미세 파형 구조 개입](../01_duration_bias/E094/README.md) · [같은 본문·총길이·에너지에서 앞뒤 저에너지 배치만 바꾼 실험](../01_duration_bias/E097/README.md) · [원음 전체를 보존하고 앞 또는 뒤에만 저에너지를 추가한 엄격 대조](../01_duration_bias/E100/README.md) · [자연적인 앞 저에너지 길이를 읽는 특징 방향: −40dB 정의](../01_duration_bias/E103/README.md) · [자연적인 앞 저에너지 길이를 읽는 특징 방향: −20dB와 안정성](../01_duration_bias/E104/README.md) · [앞 저에너지의 존재·지속 길이 규칙과 모델 점수의 관계](../01_duration_bias/E105/README.md) · [앞 저에너지 100ms 이하·500ms 이상인 자연 표본의 양 클래스 평가](../01_duration_bias/E106/README.md) · [저에너지 시간 위치와 나머지 XLS-R 특징을 교체·제외한 내부 개입](../05_residual_geometry/E108/README.md) · [같은 저에너지 내부 개입을 다섯 저장 모델에 반복한 대조](../02_repetition/E109/README.md) · [원음 전체와 실제 긴·짧은 증강 입력의 저에너지 위치 분포 전수 조사](../01_duration_bias/E110/README.md) · [정상 중심과 스푸프 중심을 명시적으로 전환하는 원본240 기반 구현](../06_spoof_centered/E111/README.md) · [역할 평가의 점수 부호·smoke 범위·checkpoint 용도 오류 감사](../06_spoof_centered/E112/README.md) · [짧은 정상 음성의 3.0275초·2초 정렬 학습 중단25epoch의21LA·DF 전체평가와 독립ROC 검산](../01_duration_bias/E113/README.md)

같은 checkpoint·음성·캐시를 재사용한 분석은 독립 학습이나 독립 표본으로 중복 합산하지 않는다.

[전체 목록](../README.md)
