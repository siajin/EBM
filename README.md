# EBM-NoRepeat

**비반복형 EBM** — 긴 입력의 4.0375초 상한은 유지하고, 짧은 원음의 순환 반복을 제거한 모델.

**상태:** seed 1에서 처음부터 50epoch 학습·공식 평가 전체 완료.

## 결과

앞 64,600샘플 상한·비반복·유효 길이 마스크·배치 8·학습 OC 중심 코사인으로 평가했다. EER 단위는 %다.

|고정 50epoch 모델|2019 LA|2021 LA|2021 DeepFake|
|---|---:|---:|---:|
|EBM-Repeat·같은 비반복 평가|1.128484|6.688715|8.877530|
|EBM-NoRepeat|0.679810|4.792117|2.723788|

|5~50epoch의 10개 저장점 관측 최소|2019 LA(epoch)|2021 LA(epoch)|2021 DeepFake(epoch)|
|---|---:|---:|---:|
|EBM-NoRepeat|0.462271 (45)|3.759449 (45)|2.084875 (5)|

각 집합의 최소 epoch는 서로 다르다. 이 최소값들을 단일 모델의 성능으로 합치거나 독립 검증에서 선택한 성능으로 제시하지 않는다.

## 가설

짧은 음성을 목표 길이까지 반복하면 원래 내용 외의 반복량·접합·frontend 문맥이 생긴다. 이를 제거하고 실제 길이를 유지하면 반복 처리에 대한 의존을 줄여 다른 길이와 도메인에 더 잘 일반화할 수 있다는 가설을 시험했다.

## 수정한 부분

|코드|변경과 이유|
|---|---|
|[data_loaders.py](data_loaders.py)|긴 입력은 원음 앞 최대 64,600, 짧은 입력은 최대 48,440샘플의 연속 무작위 구간. 부족한 길이를 반복으로 늘리지 않음|
|[data_loaders.py](data_loaders.py)|서로 다른 길이의 입력을 클래스·길이를 고려한 논리 배치로 구성. 물리 배치에서 필요한 오른쪽 0 채움만 사용|
|[model/b_linear.py](model/b_linear.py)|XLS-R 및 시간 pooling에 유효 길이 마스크 전달. 0 채움을 음성으로 집계하지 않음|
|[native_steps.py](native_steps.py)|물리 배치 8로 특징을 분할 계산하고 기울기를 재계산해 논리 배치의 OC·정상 전체쌍 손실을 보존|
|[main.py](main.py)·[train.py](train.py)|가변 길이 로더·계산·저장 경로 연결. 학습 중 5epoch마다 개발 집합 평가·저장|
|[arguments.py](arguments.py)|비반복·긴 입력 상한 사용. 짧은 입력의 추가 분류 손실과 반복 관련 보조 손실은 0|

## 공통 학습 설정

|항목|설정|
|---|---|
|학습 데이터|ASVspoof 2019 LA Train 25,380개: 정상 2,580·위조 22,800|
|모델|동결 XLS-R + 학습 가능한 bridge·attention·160차원 임베딩·OC 중심|
|배치|두 GPU 합계 240, 일반 배치에서 GPU별 정상 12·위조 108|
|정상쌍|GPU별 정상 긴 입력×짧은 입력 전체쌍, 일반 배치 12×12=144쌍|
|최적화|학습률 1e-5·seed 1·50epoch, backend→OC 중심의 두 단계 Adam|
|손실|클래스 가중 OCSoftmax + 정상쌍 학습(BPL)|


코덱 변형과 추가 HiFi-GAN 입력은 비활성화하고 원래 RawBoost를 사용했다. 긴 입력 OC와 GPU별 정상 긴×짧은 BPL을 유지했으며, GPU 간 정상 특징 공유나 새 보조 손실은 추가하지 않았다.

## 해석·실패한 부분·배운 점

공통 비반복 평가에서는 EBM-Repeat보다 세 집합의 EER이 낮았다. 그러나 과거 반복 평가와 비교하면 모든 집합에서 우월한 것은 아니다. 일부 짧은 정상의 거부 증가와 위조 누락 감소가 함께 나타났으며, 전체 EER과 개발 고정 임계값의 오류도 구분해야 한다.

원래 반복형과는 반복뿐 아니라 샘플러·마스크·메모리 계산 방식도 달라졌다. **반복 제거 하나가 성능 차이의 원인이라는 결론은 내리지 않는다.** 같은 현재 구현으로 다시 학습한 반복 대조군은 실행하지 않았다. [전체 가설·학습·평가 기록](https://github.com/siajin/EBM/blob/main/experience/02_repetition/E006/README.md) · [길이별 오류와 임계값 분석](https://github.com/siajin/EBM/blob/main/experience/02_repetition/E011/README.md)

## 원본 출처와 전체 연구

원본 구현은 [kimho1wq/EBM](https://github.com/kimho1wq/EBM)이다. 원논문 설명·인용은 [원본 README](https://github.com/siajin/EBM/blob/main/docs/UPSTREAM_README.md), 라이선스는 [LICENSE](LICENSE)를 따른다. 이 페이지의 결과는 이 저장소의 실험 기록이다.

[모델 비교와 가설별 연구](https://github.com/siajin/EBM) · [전체 평가 결과](https://github.com/siajin/EBM/blob/main/experience/RESULTS.md) · [용어 안내](https://github.com/siajin/EBM/blob/main/experience/TERMINOLOGY.md)
