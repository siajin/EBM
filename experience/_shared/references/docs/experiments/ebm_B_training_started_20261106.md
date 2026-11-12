> 표시 명칭·링크 이름은 설명형으로 정리했다. 원래 코드·경로·수치는 그대로 보존한다.

> 역사 문서 보존본. 원문 작성 당시의 상태·계획을 포함한다. 최신 상태와 확정 결과는 experience의 각 실험 README를 우선한다.

<a id="b-학습-시작-및-코드-브랜치-저장"></a>

# 4.0375초 상한 비반복 학습 학습 시작 및 코드 브랜치 저장

사용자가 학습 대상은 **4.0375초 상한 비반복 학습 코드**라고 명시했다. Baseline 재실행 준비 브랜치는 만들어졌으나 학습은 실행하지 않았다. 실제 실행은 4.0375초 상한 비반복 학습 한 건뿐이다.

| 항목 | 실제 실행 |
|---|---|
| 브랜치 | `train/ebm-B-zeropad-e50-bs240-20261106` |
| GitHub | [비공개 4.0375초 상한 비반복 학습 학습 브랜치](https://github.com/siajin/EBM/tree/train/ebm-B-zeropad-e50-bs240-20261106) |
| 커밋 | `2c2c6ff65bbadf33b811983f7c648c97fffb81a1` |
| 로컬 | `IRLab/EBM`, 위 브랜치 추적 |
| 서버 | `/home/mentee/code/EBM`, 위 브랜치 추적 |
| 실험명 | `EBM_e50_bs240_lr1e5_B_zeropad_short303` |
| 실행 | 기존 `352153.exp:0` 컨테이너 안에서 `CUDA_VISIBLE_DEVICES=0,1 python -u main.py` |
| GPU | host6/7, TITAN RTX24GiB 두 개 |
| 학습 | 50epoch, global240, GPU당120=정상12+spoof108, seed1, lr1e-5 |
| 입력 | Long최대4.0375초, Short최대3.0275초, 반복 없음,0패딩+마스크 |
| 손실 | 기존Long OC + 로컬12×12 BPL, 새 두 보조 계수0 |
| 증강 | codec-off·ACN-off, RawBoost 유지 |
| 구현 | physical8 분할 재계산, frozen frontend, 논리적 batch 손실 유지 |
| 평가/저장 | 5epoch마다 공식19LA dev 검증과 CLS/TrainState 저장, target-best 선택 없음 |

학습 Python 파일은 이미 양GPU에서 검증했던 4.0375초 상한 비반복 학습 main 커밋 `0b860798c87daef8ea47cd24f15c8aaecb9a8c7e`과 전부 같다. 새 브랜치 커밋은 실행 설명만 추가했다. 원래 baseline과 4.0375초 상한 비반복 학습 main은 별도 브랜치로 보존돼 있다.

실행 전 서버 셸과 GPU 유휴, 결과 폴더 미존재, 현재 branch/commit, 4.0375초 상한 비반복 학습 설정, codec 활성 호출0, 데이터 경로, 실제 컨테이너의GPU2개를 검사했다. 한 번만 실행하도록 launcher에 claim guard를 두었다. 이미 존재하는 결과를 삭제하거나 덮어쓰지 않았다.

운영 파일은 서버 `/home/mentee/analysis/ebm_train_B_20261106_213627/`에 있다.

- `run.sh`, `launch.json`, `preflight.json`: 실행 명령·설정·소스hash.
- `train.log`: 진행 로그. `parent_pid.txt`: 컨테이너 parent PID1619454.
- `run.exit`, `finished.txt`: 정상/오류 종료 후 생성. 시작만 했으므로 현재 완료로 취급하지 않는다.
- 결과: host `/home/mentee/results/CM/LA/EBM_e50_bs240_lr1e5_B_zeropad_short303`.

21:38:52 KST 읽기 전용 확인에서 **1/50epoch,13/108배치**를 완료했다. 최근loss1=8.548,loss2=71.494, padding=0, active_pairs=0. 실제기록에서loss1=oc_long+BPL이 성립한다. `oc_short`는 진단용 기록이며 계수0인 short OC가 목적식에 추가된 것은 아니다.

GPU메모리5540/5286MiB,사용률86/86%,로그갱신나이3.64초,exit파일없음,OOM/NCCL오류/Traceback/비유한손실표식없음이었다. DDP gradient-stride 성능 경고는 있었지만 실제 갱신은 진행됐다. 실행 소스와 로거 백업의Pythonhash가preflight와 모두 같고 작업 트리는clean이었다. 배치 240 고정 길이 반복 입력 학습 결과·checkpoint의hash도불변이다. 초기실행정상확인이지 전체학습완료나성능개선확인은아니다.

로컬 운영 기록: [시작 정보](../../artifacts/ebm_git_private_20261106/B_training_launch.json), [최근 확인](../../artifacts/ebm_git_private_20261106/B_training_last_check.json). 이후 상태는 서버를 읽어서 갱신해야 하며, 이 문서의 진행 숫자는 과거 관측값이다. 실행 중 서버 파일수정·pull·브랜치전환·재시작을 하지 않는다.
