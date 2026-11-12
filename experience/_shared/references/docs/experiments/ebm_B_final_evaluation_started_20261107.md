> 표시 명칭·링크 이름은 설명형으로 정리했다. 원래 코드·경로·수치는 그대로 보존한다.

> 역사 문서 보존본. 원문 작성 당시의 상태·계획을 포함한다. 최신 상태와 확정 결과는 experience의 각 실험 README를 우선한다.

<a id="b-학습-완료-검증과-공통-입력-최종-평가"></a>

# EBM-NoRepeat 학습 완료 검증과 공통 입력 최종 평가

## 학습 완료

EBM-NoRepeat 실험은2026-11-07 **16:22:40 KST에50epoch·exit0으로 정상 종료**했다. 시작은11월6일21:36:29KST, 경과시간은67,571초(18시간46분11초)다. `train.log`에서50×108=5,400개의(epoch,batch) 조합을누락없이확인했고, 모든학습metric도5,400행·유한값이다.

전체5,400개step의 `loss1=oc_long+BPL` 최대산술오차는4.77e-7이다. padding손실과active_pairs는항상0이다. 최종checkpoint를기존컨테이너에서strict-load했고CLS_50과TrainState_50의classifier가전부같으며Adam state의모든step이5,400이다. 최종모델가중치와기존240checkpoint는아래파일로고정했다. Target성능에따른선택은하지않는다.

| 모델 | 파일 | SHA256 |
|---|---|---|
| EBM-NoRepeat | EBM-NoRepeat 결과/model/CLS_50.pt | `63ed02a1baef97efa9c8b701814dcccebf3360d441782abe9688d8edb3cdf457` |
| EBM-NoRepeat 학습상태 | EBM-NoRepeat 결과/model/TrainState_50.pt | `3b1fe8ac2ec67e680ac582365e40cd6f34451a42484858f847f55e41a1b1cba0` |
| EBM-Repeat | 기존 결과/model/BestCLS_50_21DF.pt | `84a828eeda71d74212178dc2c2f02e72f8bb79b3573354f6cd9ca181a376d19e` |

50epoch dev점수는학습metric파일에없고5~45epoch의9개기록만있다. checkpoint50은dev검증뒤저장됐지만logger의종료전flush가없어마지막기록이미저장됐을가능성이있다. 점수를추정하거나45epoch값을복사하지않고독립최종평가에서다시계산한다. 이는원인추정과재계산계획이며최종dev점수를이미확인했다는뜻은아니다.

## 실제 평가 프로토콜

21:54:51KST부터기존컨테이너·`CUDA_VISIBLE_DEVICES=0,1`로독립평가를한번시작했다. 학습을재실행하거나서버의학습소스를수정하지않았다. 실행루트는 `/home/mentee/analysis/ebm_B_final_eval_20261107_215451`이다. 로컬상태는 [설정·결과 근거](../../artifacts/ebm_git_private_20261106/B_final_evaluation_state.json), 읽기전용관찰도구는 `artifacts/ebm_git_private_20261106/check_B_final_evaluation.py`다. 진행중에는중복실행하지않는다.

두고정모델에같은원음앞최대64,600sample,반복없는0패딩+mask,rank당batch8을적용한다. 현재dataset코드의dev24,844·19LAeval71,237·21LA148,176·21DF533,928, 총778,185개의동일trial에두score를저장한다. FrozenXLS-R특징은배치내에서두모델이공유하며파일로저장하지않는다. 원길이·UID·speaker·label·score등파생값만보관한다.

Source의pooling은eval에서도noise_std=1e-5이다. 공식최종평가도이값을유지하고두모델에공통RNG를적용한다. QC의배치/단독동일성검사에서만noise를0으로두고검사후다시복원했다. 양GPU QC에서sourceEnrollmentSet과입력동일,엄격한checkpointload,내장forward와수동cosine점수오차0,배치/단독점수최대오차1.19e-7을확인했다. Frontend는facebook/wav2vec2-xls-r-300m,revision `1a640f32ac3e39899438a2931f9924c02f080a54`,feature norm은layer다.

EER는ties를포함한ROC의FPR=FNR교차를선형보간한다. 운영threshold는각모델의dev에서|FPR-FNR|최소,동률이면낮은FPR,그다음높은threshold규칙으로고정한다. Target의정상거부·spoof수락에는이dev threshold만쓴다. Target EER의ROC상threshold는설명통계이며운영threshold나checkpoint선택에사용하지않는다.

## 다음 예약에서 이어갈 작업

`run.exit=0`과`complete.json`을확인하면paired_scores_dev/19LA/21LA/21DF.parquet와metrics.json을검산한다. 각phase의index전수범위·중복UID없음·유한score·원길이cap·csvhash가평가부에서도검사된다. 평가중이면progress_rank0/1.json과done파일을읽고기다리며평가를다시시작하지않는다.

최종분석에서는학습/dev곡선,세target의고정50epochEER,사전지정원길이구간별정상거부·spoof수락과분모,동일UID쌍불확실성,EBM-Repeat의같은EBM-NoRepeat평가입력비교를보고한다. 기존기록의반복평가EER는프로토콜이다른표로구분한다. Codec실제적용미확정·sampler/학습구현차이·한seed라는제약을명시한다. 같은모델을더유리한epoch로바꾸거나target에서threshold를고르지않는다.

21:59:26KST의확인에서양GPU정상·smoke.exit0·fullQC통과,dev를rank0 8,008개/rank1 9,208개까지평가중이었다. 평가코드SHA256은 `65c089cb30333d9ec5af699f79f8e9baefd0727da40149bebfc7e170285ab63b`다. 이숫자는과거관측값이며최종평가·분석완료가아니다. 6시간예약은평가상태를반영하도록갱신했고최종보고후에만종료한다.
