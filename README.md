# EBM-FullContext

**전체 문맥형 EBM** — 비반복 처리를 유지하면서 긴 **학습** 입력에 원음 전체를 사용하는 모델.

**상태:** seed 1에서 처음부터 50epoch 학습·공식 평가 전체 완료. EBM-NoRepeat의 학습 가중치를 이어 쓴 모델이 아니다.

## 결과

두 모델 모두 앞 64,600샘플 상한·비반복·마스크·배치 8·학습 OC 중심 코사인으로 평가했다. EER 단위는 %다.

|고정 50epoch 모델|2019 LA|2021 LA|2021 DeepFake|
|---|---:|---:|---:|
|EBM-NoRepeat|0.679810|4.792117|2.723788|
|EBM-FullContext|0.584636|5.163467|2.881368|

|5~50epoch의 10개 저장점 관측 최소|2019 LA(epoch)|2021 LA(epoch)|2021 DeepFake(epoch)|
|---|---:|---:|---:|
|EBM-FullContext|0.488400 (40)|4.622076 (30)|2.152549 (5)|

고정 50epoch에서 2019 LA는 개선됐지만 2021 LA·DeepFake는 악화했다. 집합별 사후 최소를 한 모델의 성능으로 합치지 않는다.

## 가설

반복을 제거한 뒤에도 긴 발화를 앞 4.0375초로 자르면 이후 발화 내용이 사라진다. 긴 원음을 더 활용하면 표현과 일반화가 개선될 수 있다는 가설을 시험했다. 비교 기준은 EBM-NoRepeat다.

## 수정한 부분

|코드|변경과 이유|
|---|---|
|[arguments.py](arguments.py)|학습용 `cap_long=False`와 새 결과 이름. 긴 입력의 64,600샘플 상한을 해제해 원음 전체 사용|
|[data_loaders.py](data_loaders.py)|평가용 로더에는 `cap_long=True`를 명시. 학습 문맥 변경이 평가 문맥 변경과 섞이지 않게 함|

짧은 입력은 최대 48,440샘플(3.0275초)의 연속 무작위 구간이고 짧은 원음은 반복하지 않는다. 모델·손실·샘플러·optimizer·기울기 재계산 코드는 EBM-NoRepeat와 같다. 실제 두 브랜치의 실행 Python 차이는 위 두 파일이다.

## 공통 학습 설정

|항목|설정|
|---|---|
|학습 데이터|ASVspoof 2019 LA Train 25,380개: 정상 2,580·위조 22,800|
|모델|동결 XLS-R + 학습 가능한 bridge·attention·160차원 임베딩·OC 중심|
|배치|두 GPU 합계 240, 일반 배치에서 GPU별 정상 12·위조 108|
|정상쌍|GPU별 정상 긴 입력×짧은 입력 전체쌍, 일반 배치 12×12=144쌍|
|최적화|학습률 1e-5·seed 1·50epoch, backend→OC 중심의 두 단계 Adam|
|손실|클래스 가중 OCSoftmax + 정상쌍 학습(BPL)|


물리 배치 8, 유효 길이 마스크, RawBoost·코덱 비활성과 원래 두 단계 갱신도 유지했다. **원음 전체 사용은 긴 학습 입력에 대한 설명이며 평가 입력의 상한을 없앤 것이 아니다.**

## 해석·실패한 부분·배운 점

더 긴 학습 문맥은 모든 도메인의 일관된 개선으로 이어지지 않았다. 한 seed의 결과이며 길이가 달라지면 내용·증강 실현값·pooling·난수 소비도 달라진다. 같은 seed가 서로 다른 입력 길이의 난수 텐서까지 같게 만들지는 않는다.

이 결과는 주어진 입력·증강·최적화 조건의 관측이다. 긴 문맥 자체의 보편적 유익·해로움이나 순수 길이의 인과 효과로 확대하지 않는다. [전체 설계·학습·결과](https://github.com/siajin/EBM/blob/main/experience/01_duration_bias/E007/README.md) · [두 비반복 모델의 모든 저장점 평가](https://github.com/siajin/EBM/blob/main/experience/02_repetition/E010/README.md)

## 원본 출처와 전체 연구

원본 구현은 [kimho1wq/EBM](https://github.com/kimho1wq/EBM)이다. 원논문 설명·인용은 [원본 README](https://github.com/siajin/EBM/blob/main/docs/UPSTREAM_README.md), 라이선스는 [LICENSE](LICENSE)를 따른다. 이 페이지의 결과는 이 저장소의 실험 기록이다.

[모델 비교와 가설별 연구](https://github.com/siajin/EBM) · [전체 평가 결과](https://github.com/siajin/EBM/blob/main/experience/RESULTS.md) · [용어 안내](https://github.com/siajin/EBM/blob/main/experience/TERMINOLOGY.md)
