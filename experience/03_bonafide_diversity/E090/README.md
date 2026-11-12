# E090. 낮은 정상 점수 꼬리의 출처·코덱·길이·음향·화자 진단

**실행 상태:** 기존 실제 결과 확인·문서화 완료

이 문서의 상태는 2026-11-12 출판 검토 기준이다. 기존 문서·state의 시점별 상태와 충돌하면 실제 완료표식과 본문 최신 보완을 우선한다.

## 가설

정상낮은점수꼬리는 길이외에 출처·음향조건과 관련될 수 있다.

## 그렇게 설계한 이유

정상인데 기준 방향에서 벗어난 표본을 어떤 관측 조건이 설명하는지 조사한다.

대조와 통제의 근거: 낮은 점수를 예측하는 새 진단 분류기이며 음성 위조 탐지기를 새로 학습한 것이 아니다.

## 모델·데이터·표본

배치 80·80epoch의 고정 모델(체크포인트 SHA256 30043199664bc84da780654e01fd430235f647b894e7832a52f777c7fee5939a). 별도 명시가 없으면 DF eval에서 seed 1로 정상·위조 각 1,000개를 클래스 안에서 중복 없이 무작위 추출한 같은 2,000개다. 추론 배치 4, worker 2, 원래 앞 64,600샘플·부족하면 순환 반복, 증강 없음이다.

## 실험 내용과 상세 절차

1. 정상 코사인 0.9 미만을 꼬리, 0.2 미만을 심한 꼬리로 정의한다. 실제 분류 오류 정의와 구별한다.
2. 출처·코덱·원음 길이·전체/앞/뒤 저에너지·에너지·화자별 분포를 교차 집계한다.
3. 정상 1,000개에서 화자를 나눈 다섯 분할 예측으로 길이만, 출처만, 출처+길이, 출처+코덱+길이, 음향 추가, XLS-R/시간 요약 PCA 16축 추가 조건의 꼬리 예측 AUROC를 비교한다.
4. 출처 안의 길이 차이, 길이를 조정한 화자 차이와 같은 표본의 파형 개입을 추가로 연결한다. 초기 빈 tail_factors 폴더는 완료 실험으로 세지 않고 결과가 있는 v2만 사용한다.
5. 원실행 코드를 읽어 세부값도 확인했다. 초기 꼬리 예측은 seed 1의 StratifiedGroupKFold 5분할, 범주 one-hot·숫자 중앙값 채움·표준화, C=1·최대 1,500회의 로지스틱이다. 후속은 seed 17·5분할, 길이의 4개 knot·2차 스플라인, C=1·최대 1,000회·클래스 균형 가중치를 썼다. PCA 16축 입력 분류기는 C=0.1이며 PCA·표준화는 학습 분할에서 맞췄다.

## 측정과 검증

발화별 대응 점수·원래 160차원 코사인·클래스별 분모와 저장 검산을 확인한다.

## 결과

꼬리는 정상 163/1,000개였다. 후속 다섯 분할 AUROC는 길이만 0.648423, 출처만 0.862069, 출처+길이 0.896493, 출처+코덱+길이+음향 0.903079였다.

## 해석·실패한 부분·한계

하나의 고정 학습 모델과 탐색 표본의 결과다. 전체 공식 평가 EER, 여러 학습 seed의 효과, 손실을 제거한 재학습의 효과로 해석하지 않는다. 반복 관찰한 표본과 서로 겹치는 후속 분석은 독립 실험 표본으로 합산하지 않는다. 초기·후속 분석의 특징 묶음과 AUROC는 서로 다르다. 클래스별 화자·원발화 의존성과 관찰 교란을 완전히 제거한 인과 분석은 아니다.

## 배운 점

출처의예측력이 길이보다컸다. 꼬리0.9는 학습margin이며 실제오류임계값이 아니다.

## 실험 코드

- [historical_analysis/tail_diagnosis_21df.py](code/tail_diagnosis_21df.py)
- [historical_analysis/tail_factors_21df.py](code/tail_factors_21df.py)
- [historical_analysis/tail_source_followup.py](code/tail_source_followup.py)

코드의 파일별 원본 위치·SHA256은 [source_manifest.json](code/source_manifest.json)에 있다. 공통 의존성과 실행 위치는 [재현 안내](../../REPRODUCIBILITY.md)를 따른다. 역사 snapshot은 당시 실패를 포함한 실제 버전이며 최신 수정 코드와 구분한다.

## 설정·결과·검산 근거

- [artifacts/experiment_methods_audit_20261111/historical_receipts/EBM80_21DF_tail_diagnosis.json](../../_shared/references/artifacts/experiment_methods_audit_20261111/historical_receipts/EBM80_21DF_tail_diagnosis.json)
- [artifacts/experiment_methods_audit_20261111/historical_receipts/EBM80_21DF_tail_factors.json](../../_shared/references/artifacts/experiment_methods_audit_20261111/historical_receipts/EBM80_21DF_tail_factors.json)
- [artifacts/experiment_methods_audit_20261111/historical_receipts/EBM80_21DF_tail_factors_v2.json](../../_shared/references/artifacts/experiment_methods_audit_20261111/historical_receipts/EBM80_21DF_tail_factors_v2.json)
- [artifacts/experiment_methods_audit_20261111/historical_method_source_checks.json](../../_shared/references/artifacts/experiment_methods_audit_20261111/historical_method_source_checks.json)

## 연결

다른 관련 가설: [길이·유효 발화량과 앞뒤 저에너지 편향](../../01_duration_bias/README.md) · [잔여 방향·좌표·거리·기하와 변형 반응](../../05_residual_geometry/README.md)

[가설 목록으로](../README.md) · [전체 실험 목록](../../README.md)
