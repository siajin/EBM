# EBM-Repeat

**반복형 EBM** — 짧은 원음을 반복해 긴·짧은 고정 길이 입력을 만드는 기준 모델 계열.

**이 브랜치:** 보관 소스 복원 코드. **보관된 모델:** 50epoch 학습 완료. 코드 복원 자체로 새 학습을 실행한 것은 아니다.

## 결과

동일한 보관 50epoch 모델도 평가 입력에 따라 결과가 크게 달랐다. EER 단위는 %이며 낮을수록 좋다.

|평가 입력|2019 LA|2021 LA|2021 DeepFake|
|---|---:|---:|---:|
|과거 고정 길이 반복 평가|0.571040|1.052789|3.477033|
|공통 비반복 평가|1.128484|6.688715|8.877530|

공통 비반복 평가는 앞 64,600샘플 상한·유효 길이 마스크·배치 8·학습 OC 중심 코사인을 사용했다. 과거 반복 평가와 현재 공통 평가의 수치를 섞어 최소값이나 개선율을 만들지 않는다. [공통 입력 평가와 검산](https://github.com/siajin/EBM/blob/main/experience/02_repetition/E009/README.md)

## 가설과 역할

정상 중심과 정상 전체쌍 정렬을 학습한 기준 모델을 보존하고, 이후 반복 제거·긴 문맥 사용의 효과를 비교하려는 출발점이다. 원래 학습의 고정 길이 입력은 배치 구성을 단순하게 만들지만, 짧은 원음에 반복량·접합·문맥 변화를 추가한다. 그 영향을 후속 모델에서 검사했다.

## 입력과 복원한 부분

긴 입력은 64,600샘플(4.0375초), 짧은 입력은 48,440샘플(3.0275초)이다. 원음이 부족하면 순환 반복하고, 길면 긴 입력은 앞부분·짧은 입력은 연속 무작위 구간을 사용한다. 두 입력의 RawBoost는 독립적이다.

|코드|내용|
|---|---|
|[data_loaders.py](data_loaders.py)|클래스별 로더·DistributedSampler·고정 길이 반복과 절단을 복원|
|[train.py](train.py)|원래 전체 연산 그래프와 backend→중심의 두 단계 갱신을 복원|
|[model/b_linear.py](model/b_linear.py)·[model/loss.py](model/loss.py)|원래 모델과 GPU별 정상 전체쌍 손실을 보존|
|[arguments.py](arguments.py)|배치 240·학습률 1e-5·50epoch 설정을 복원|

복원 코드에서는 사용자 지시에 따라 온라인 코덱 호출 두 곳을 비활성화했다. **과거 보관 모델을 실제로 학습할 때 코덱이 적용됐는지는 확정하지 못했다.** 따라서 위 결과는 이 복원 브랜치로 새로 수행한 코덱 비활성 학습의 결과가 아니다.

## 공통 학습 설정

|항목|설정|
|---|---|
|학습 데이터|ASVspoof 2019 LA Train 25,380개: 정상 2,580·위조 22,800|
|모델|동결 XLS-R + 학습 가능한 bridge·attention·160차원 임베딩·OC 중심|
|배치|두 GPU 합계 240, 일반 배치에서 GPU별 정상 12·위조 108|
|정상쌍|GPU별 정상 긴 입력×짧은 입력 전체쌍, 일반 배치 12×12=144쌍|
|최적화|학습률 1e-5·seed 1·50epoch, backend→OC 중심의 두 단계 Adam|
|손실|클래스 가중 OCSoftmax + 정상쌍 학습(BPL)|


## 해석과 한계

반복형 모델은 반복 평가에서, 후속 비반복형 모델은 비반복 평가에서 유리했다. 반복의 보편적 유익·해로움을 단정할 수 없다. 후속 모델과는 샘플러·마스크·계산 방식도 달라 반복 하나의 독립 인과 비교는 아니다. [학습 기록·최소값·한계](https://github.com/siajin/EBM/blob/main/experience/02_repetition/E005/README.md)

재현할 때는 [arguments.py](arguments.py)의 데이터 경로와 결과 이름을 새 환경에 맞게 설정한다. 원래 로거는 같은 이름의 결과 폴더를 삭제하므로 새 고유 이름을 사용한다. 데이터·모델 가중치는 이 브랜치에 포함하지 않는다.

## 원본 출처와 전체 연구

원본 구현은 [kimho1wq/EBM](https://github.com/kimho1wq/EBM)이다. 원논문 설명·인용은 [원본 README](https://github.com/siajin/EBM/blob/main/docs/UPSTREAM_README.md), 라이선스는 [LICENSE](LICENSE)를 따른다. 이 페이지의 결과는 이 저장소의 실험 기록이다.

[모델 비교와 가설별 연구](https://github.com/siajin/EBM) · [전체 평가 결과](https://github.com/siajin/EBM/blob/main/experience/RESULTS.md) · [용어 안내](https://github.com/siajin/EBM/blob/main/experience/TERMINOLOGY.md)
