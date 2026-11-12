# EBM-Repeat · 코덱 제외 재학습 준비

**상태: 코드 준비만 완료. 이 실행 이름의 별도 학습·완료 EER은 없다.**

이 브랜치는 [EBM-Repeat](https://github.com/siajin/EBM/tree/models/repeat)의 원래 학습 흐름을 유지하고, 코덱 변형을 끈 상태의 재학습을 준비한 코드다. 학습 완료 모델과 구별하기 위해 `setup/`에 둔다.

## 가설과 준비 목적

입력의 반복 여부를 비교할 때 코덱 변형까지 함께 바뀌면 차이의 원인을 해석하기 어렵다. 코덱을 제외한 반복 입력 기준을 따로 준비해 비교 조건을 정리하고 기존 결과를 보존하려 했다. 실제 재학습을 실행하지 않았으므로 코덱 제외의 성능 효과를 검증한 결과는 없다.

## 수정한 부분

|기준|수정|
|---|---|
|EBM-Repeat 복원 코드|이미 온라인 코덱 호출 두 곳이 비활성화된 상태|
|[arguments.py](arguments.py)|기존 결과를 덮어쓰지 않도록 결과 이름만 새로 지정|
|문서|재학습 준비 범위와 당시 실행 계획 기록|

긴 64,600·짧은 48,440샘플의 반복·절단, 클래스별 로더, 정상 전체쌍 손실, 두 단계 Adam, 배치 240·학습률 1e-5·seed 1·50epoch 계획은 유지했다. 기준 복원 브랜치와 비교한 Python 변경은 결과 이름뿐이다.

## 결과와 해석

|항목|상태|
|---|---|
|코드와 실행 이름 준비|완료|
|이 이름의 별도 본학습|미실행|
|저장된 새 모델·새 EER|없음|
|과거 EBM-Repeat 결과|별도 보관 결과이며 이 준비 브랜치의 재학습 결과가 아님|

과거 실제 코덱 적용 여부는 미확정이다. 이 브랜치의 코드에서 코덱을 껐다는 사실로 과거 모델의 증강 이력을 소급 확정하지 않는다. [보관된 반복형 모델의 실제 기록](https://github.com/siajin/EBM/blob/main/experience/02_repetition/E005/README.md)

## 원본 출처와 전체 연구

원본 구현은 [kimho1wq/EBM](https://github.com/kimho1wq/EBM)이다. 원논문 설명·인용은 [원본 README](https://github.com/siajin/EBM/blob/main/docs/UPSTREAM_README.md), 라이선스는 [LICENSE](LICENSE)를 따른다. 이 페이지의 결과는 이 저장소의 실험 기록이다.

[모델 비교와 가설별 연구](https://github.com/siajin/EBM) · [전체 평가 결과](https://github.com/siajin/EBM/blob/main/experience/RESULTS.md) · [용어 안내](https://github.com/siajin/EBM/blob/main/experience/TERMINOLOGY.md)
