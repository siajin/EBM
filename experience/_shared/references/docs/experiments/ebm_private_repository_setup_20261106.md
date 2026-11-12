> 역사 문서 보존본. 원문 작성 당시의 상태·계획을 포함한다. 최신 상태와 확정 결과는 experience의 각 실험 README를 우선한다.

# EBM 비공개 저장소 연결 완료

**현재 상태 정정:** 아래는 최초 연결 당시 기록이다. 이후 사용자 요청으로 서버만 `baseline/e50-bs240-lr1e5`로 복원했다. 로컬과 GitHub `main`은 B 버전을 유지한다. 서버 baseline은 기존240 백업에서 codec-off와 Git 제외 규칙을 보존한 버전이다. 자세한 현재 상태는 [서버 baseline 복원](ebm_server_baseline_restore_20261106.md)을 따른다.

2026-11-06 사용자의 요청으로 [siajin/EBM](https://github.com/siajin/EBM)을 **Private**로 생성했다. 공개 저장소를 private fork로 바꾸는 방식이 아니라, 원본 이력과 LICENSE를 보존한 별도 비공개 저장소다.

| 구분 | 위치 |
|---|---|
| GitHub | https://github.com/siajin/EBM |
| 로컬 코드 폴더 | `C:\Users\sihwa\PycharmProjects\PythonProject\IRLab\EBM` |
| 서버 코드 폴더 | `/home/mentee/code/EBM` |
| 작업 브랜치 | `main`, 두 폴더 모두 `origin/main` 추적 |
| 원본 저장소 | `upstream` = https://github.com/kimho1wq/EBM |
| 최초 사용자 커밋 | `0b860798c87daef8ea47cd24f15c8aaecb9a8c7e` |

커밋에는 현재 B 코드, 기존 사용자 설정·데이터 경로 변경, ignore 규칙, 통제실험·작업 안내가 포함됐다. 현재 B의 손실·입력 코드는 제한 검증 뒤 변경하지 않았으며 Python 파일 hash도 유지했다. 데이터·음성·모델·결과·백업·인증 파일은 포함하지 않았다. 기존 원본 LICENSE와 커밋 이력은 유지했다.

서버에는 이 저장소 전용 write deploy key를 설정했다. 비밀 키는 서버에만 있고 개인 GitHub token을 서버에 복사하지 않았다. GitHub 호스트 키는 공식 HTTPS API로 받은 키로 확인하며, 서버의22번 연결 timeout 때문에443번 SSH 연결을 사용한다. 로컬은 기존 GitHub CLI 인증을 저장소 범위 credential helper로 사용한다.

검증 당시 GitHub visibility가Private, GitHub main과 양쪽 HEAD가 위 커밋으로 일치하고 작업 폴더는 모두 clean이었다. 서버에서 private origin 읽기·쓰기, 로컬 clone과 origin tracking을 확인했다. 전체 학습을 시작하거나 기존 실험을 중단하지 않았다.

앞으로 코드 수정은 `IRLab/EBM`에서 한다. 수정한 쪽에서 commit/push하고, 다른 쪽은 미커밋 변경과 실행 중 실험을 확인한 뒤 `git pull --ff-only`로 반영한다. 파일 수정만으로 GitHub나 서버에 자동 반영되지는 않는다. 학습 실행은 별도 요청에 따라 기존 컨테이너와 `CUDA_VISIBLE_DEVICES=0,1`로 수행한다.

현재 조건은 저장소의 `docs/CONTROL_B.md`, 작업 절차는 `docs/WORKFLOW.md`에 있다. 상세 연결 검증은 [connection.json](../../artifacts/ebm_git_private_20261106/connection.json)에 보관했다. 서버 연결 전 config와 ignore 백업은 `/home/mentee/analysis/ebm_private_git_20261106_2125/`에 보관했다.
