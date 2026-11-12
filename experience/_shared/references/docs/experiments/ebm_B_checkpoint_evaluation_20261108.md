> 표시 명칭·링크 이름은 설명형으로 정리했다. 원래 코드·경로·수치는 그대로 보존한다.

> 역사 문서 보존본. 원문 작성 당시의 상태·계획을 포함한다. 최신 상태와 확정 결과는 experience의 각 실험 README를 우선한다.

<a id="b-8개-체크포인트-평가-준비"></a>

# 4.0375초 상한 비반복 학습 8개 체크포인트 평가 준비

> 2026-11-09 변경: 사용자 요청으로 4.0375초 상한 비반복 학습과 원음 전체 비반복 학습 전체20개 체크포인트 평가로 확대해 실행했다. 이 문서의 4.0375초 상한 비반복 학습 8 전용 launcher는 실행하지 않는다. 현재 상태는 artifacts/ebm_B_C_checkpoints_20261109/state.json, 기록은 ebm_B_C_all_checkpoint_evaluation_20261109.md를 확인한다. 정기 예약은 PAUSED다.

사용자 요청: 4.0375초 상한 비반복 학습의 15·20·25·30·35·40·45·50epoch를 각각 19LA·21LA·21DF에서 평가한다. 현재 원음 전체 비반복 학습 학습이 두 GPU를 사용하므로 평가 상태는 **원음 전체 비반복 학습 정상 종료 대기**다. 평가 결과는 아직 없다.

- 서버 root: `/home/mentee/analysis/ebm_B_checkpoints_20261108`
- [상태](../../artifacts/ebm_B_checkpoints_20261108/state.json), [출처·해시](../../artifacts/ebm_B_checkpoints_20261108/provenance.json)
- [평가 코드](../../../../../tools/evaluate_B_checkpoints.py), [실행 도우미](../../../../../tools/launch_B_checkpoints.py)
- 기존 예약 `ebm-c-30-b`를 ACTIVE/30분 그대로 유지하며 4.0375초 상한 비반복 학습 평가·보고를 추가했다. 원음 전체 비반복 학습 종료 후 4.0375초 상한 비반복 학습 평가, 그다음 원래 원음 전체 비반복 학습 고정50 paired 평가/분석을 수행한다. GPU 작업은 동시에 실행하지 않는다.

각 epoch의 CLS와 TrainState가 서버에 있음을 확인했고 CLS SHA와 TrainState SHA를 기록했다. 서버 학습 source·branch·프로세스는 변경하지 않았다. 새 독립 평가 코드는 analysis root에만 업로드했다. 로컬 EBM/tools 신규 파일은 아직 커밋하지 않았다.

평가 조건은 앞64,600samples cap, 비반복, mask, GPU당 batch8, pooling noise1e-5, 공통 RNG다. 두 GPU에서 같은 frozen frontend를 모델별로 재사용하되 hidden states는 메모리에서만 사용한다. Dev24,844·19LA71,237·21LA148,176·21DF533,928개와 8개 score column을 저장한다. Dev 기준 threshold 규칙을 유지하고 target 최저 checkpoint를 최종 모델로 선택하지 않는다.

검증 완료: 로컬/서버 Python syntax, bash syntax, 8개 score column과 두 rank 병합·coverage·완전분리/역분리 EER·dev threshold 집계의 CPU 검증. 실제 GPU smoke는 원음 전체 비반복 학습 실행 중이라 아직 수행하지 않았다. 원음 전체 비반복 학습 exit0/최종checkpoint, GPU 유휴, 기존 컨테이너 shell 유휴, source/모델 해시 검사가 모두 통과해야 실행된다. Smoke는 8개 head strict load·TrainState 일치·Adam step·mask/단독 점수·기존50epoch dev 점수 재현을 검사한다. 성공한 뒤에만 본 평가가 진행된다.

실행 후 `run.exit`, `smoke.exit`, `complete.json`, `metrics.json`, `progress_rank*.json`, `done_*.json`을 확인한다. Claim이나 종료파일이 있으면 중복 실행하지 않는다. 완료 후 독립 ROC·행수·UID·라벨·유한값·해시를 검산하고 8×3 EER와 추이 및 dev 운영점 오류를 보고한다. 원음·SSL hidden·checkpoint·인증은 로컬로 수신하지 않는다.
