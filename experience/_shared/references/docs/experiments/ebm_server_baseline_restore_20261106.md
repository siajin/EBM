> 역사 문서 보존본. 원문 작성 당시의 상태·계획을 포함한다. 최신 상태와 확정 결과는 experience의 각 실험 README를 우선한다.

# 서버 EBM batch240 코드 복원

2026-11-06 사용자가 서버 EBM을 `EBM_e50_bs240_lr1e5` 버전으로 돌리도록 요청했다. 서버 작업 폴더를 추가하지 않고 기존 `/home/mentee/code/EBM`을 별도 baseline 브랜치로 전환했다. 학습 실행이나 가중치 로드는 하지 않았다.

| 위치 | 현재 버전 |
|---|---|
| 서버 `/home/mentee/code/EBM` | `baseline/e50-bs240-lr1e5`, commit `cc7d324c7287b4c5e5f8a29dafec303f834f3525` |
| 로컬 `IRLab/EBM` | B 코드의 `main`, commit `0b860798c87daef8ea47cd24f15c8aaecb9a8c7e` |
| 비공개 `siajin/EBM` | 두 브랜치 모두 보존; 기본 브랜치는 `main` |

복원 기준은 서버 `/home/mentee/results/CM/LA/EBM_e50_bs240_lr1e5/backup/scripts`다. 원본34파일 중32개는 byte 단위로 백업과 같고, 아래 두 파일만 의도적으로 다르다.

- `data_loaders.py`: 기존에 요청한 codec-off를 유지하도록 online codec conditional block 두 개를 주석 처리했다. 확률값0.8과 codec 구현은 그대로다. AST 비교로 두 block 제거 이외의 의미상 차이가 없음을 확인했다.
- `.gitignore`: 데이터·결과·모델·인증 파일·로컬 백업을 제외하는 사용자 저장소 규칙을 유지했다.

따라서 보관 소스와 완전히 동일한 codec 활성화 버전이라고 부르지는 않는다. 사용자의 앞선 codec-off 지시를 유지한 baseline 복원이다.

기존 고정 long/short 반복·crop, 클래스별 loader, 원래 OC·로컬 BPL과 두 단계 갱신, 원래 평가 코드·logger를 복원했다. B의 mask·길이 sampler·분할 재계산·새 보조 손실 연결은 서버에서 제거됐다. `native_steps.py`도 이 브랜치에는 없다. epoch50, 전체batch240, lr1e-5, 기존 실험명으로 복원했다. `main`에 보존된 B의 소스나 설정은 바꾸지 않았다.

복원 전 서버는 유휴였고 지정 GPU6/7은 각6MiB·0%였다. 복원 전 추적 파일을 서버 `/home/mentee/analysis/ebm_server_baseline_restore_20261106_2133/before/`에 보관했다. 소스 구문 검사와 실제 짧은/긴 학습 입력의 고정 반복·crop 검사, codec 호출 금지 검사를 수행했다. 과거 결과 폴더 전체의 hash는 복원 전후 동일하며 checkpoint를 수정하지 않았다. [복원 메타데이터](../../artifacts/ebm_git_private_20261106/server_baseline_restore.json)에 기록했다.

기존 logger도 복원했으므로 **같은 실험명으로 실행하면 기존 결과 폴더를 삭제할 수 있다.** 이번에는 실행하지 않았다. 재학습을 별도로 요청받으면 새 결과 이름을 사용해야 한다. 서버를 B로 바꾸려면 브랜치 전환을 명시적으로 결정해야 하며 자동으로 main을 pull하지 않는다.
