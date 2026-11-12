# 코드 구조와 재현 안내

## 버전과 출처

저장소 루트에는2026-11-12의 로컬 EBM 개발 소스·tests·tools를 올렸다. 이 현재트리는 과거모든실행의공통소스가 아니다. 실험별 `code/snapshots/<run>/`은 실제서버보관소스이며 SHA256으로 원본을 보존한다. 각 `code/source_manifest.json`의 kind와 origin으로 실제frozen버전·현재분석도구·검증코드를 구분한다. 실패v1도실패당시코드를보존하므로최신수정으로몰래대체하지않았다.

공통분석소스는 `_shared/workspace/artifacts/`·`_shared/workspace/docs/`·`_shared/workspace/historical_analysis/`, 공통현재EBM의Python의존성은 `_shared/workspace/EBM/`에있다. 실험별code에는그실험의실제주요소스를별도복사했다. 모듈경로를유지하려면공통workspace구조에서작업하거나snapshot을독립작업디렉터리로복사한뒤실행한다. 원본의절대경로·결과root는사용자환경에맞게새고유경로로설정한다.

## 환경과 입력

기존환경기준은루트Dockerfile과upstream설명이다. 일부로컬감사는Python3.13/PyTorch2.13 CPU/Gloo,실제서버학습은기존컨테이너에서수행됐다. 정확한라이브러리lock을새로만들어과거전체환경과같다고주장하지않는다. SoundFile·NumPy·PyTorch·Transformers·scikit-learn·SciPy·pandas 및개별도구의SoX/FFmpeg/WebRTC VAD등의의존성은해당소스import·실험명세를따른다.

ASVspoof2019 Train25380(정상2580/위조22800),공식eval19LA71237·21LA148176·DF533928을사용한다. 데이터와pretrainedXLS-R300m·checkpoint·캐시배열은포함하지않았다. 데이터는각자허가된경로로준비하고frontendrevision은실행설정과같은것으로고정해야한다. 과거의허가되지않은대체모델이나새다운로드를이문서가자동실행하지않는다.

## 학습 진입점

|계열|실제 코드|역사 구성과 구분|
|---|---|---|
|초기80/160/전역160/240/실패320|각실험code/snapshots의main.py·arguments.py·train.py|당시고정길이64600/48440·wrap·로컬또는전역BPL|
|EBM-NoRepeat와 EBM-FullContext|E006/E007의frozen snapshot,루트native_steps.py|비반복·mask·분할gradient replay·EBM-NoRepeat와 EBM-FullContext의long상한차이|
|coverage|tools/train_coverage_experiment.py,positive_bpl.py|길이·VAD·coverage정책과독립오른쪽정상stream|
|속도 0.8~1.2배·최대 20% 절단 변형원본240|tools/train_baseline240_mild.py,baseline240_mild.py|실패v1/v2snapshot을각각보존|
|짧은 정상 음성의 3.0275초·2초 정렬 학습|tools/run_short32_experiment.py,short32_original240.py,short32_lifecycle.py|원본240hook·대상행조건부재가중·save-first|
|역할전환|tools/train_role_experiment.py,role_*.py|기본validate;실제GPUsmoke/train미실행|

현재main.py 기본설정은EBM-FullContext/native계열의개발기본값이다. 과거원본240재현은그대로main.py를실행하는대신실험snapshot을사용한다. 같은결과이름재실행시원본로거가결과를삭제할수있으므로재현은새고유root를사용한다. 취소한paired watcher·대기는재활성화하지않는다.

역할전환의코드·metadata사전검사는다음처럼명시적으로확인할수있다(학습실행아님).

```bash
python tools/train_role_experiment.py
```

## 결과 해석 규약

공식EER은높은점수=bona라벨1,동점그룹을유지한ROC의선형보간이다. spoof중심은canonical점수−cos를정확히한번적용한다. 원래bona중심은+cos다. 개발임계값고정오류·target자체EER·층화소표본EER·내부OOFCV·전체eval을서로구분한다. 역할v2체크포인트는역할대응로더로읽어야한다.

## 이 출판 작업의 실행 범위

새학습·새음성추론·GPU작업·원격수정·대기등록0회,추가비용상한0USD. 완료score는파생결과만메모리로받아독립검산했다. 모든SSH연결·전경검사프로세스를종료하고원본·로그·서버코드는그대로보존한다. 저장소에자동실험시작workflow는추가하지않는다.
