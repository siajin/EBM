> 표시 명칭·링크 이름은 설명형으로 정리했다. 원래 코드·경로·수치는 그대로 보존한다.

> 역사 문서 보존본. 원문 작성 당시의 상태·계획을 포함한다. 최신 상태와 확정 결과는 experience의 각 실험 README를 우선한다.

<a id="b-학습-6시간-점검과-완료-후-분석-예약"></a>

# EBM-NoRepeat 학습 6시간 점검과 완료 후 분석 예약

**후속 변경:** 사용자의 추가 요청으로 이 예약의 현재 주기는 **30분**이다. 같은ID `ebm-b-6`을갱신했으며중복예약은만들지않았다. 최종목적분석에는고정모델×입력정책2×2의제한된추론과centroid각도/잔여방향진단도포함하고, 증거에따라가설지지·반박·미확정을판정한다. 아래6시간표현은최초설정당시기록이다.

사용자가6시간기준 점검과 학습 종료 후 분석을 요청했다. 기존 대화로 돌아오는 heartbeat 예약 `ebm-b-6`을 **ACTIVE**로 생성하고 다시 조회해 확인했다. 중복된 기존 EBM 예약은 확인되지 않았다. [예약 상태](../../artifacts/ebm_git_private_20261106/B_monitor_schedule.json)에 ID와 대상 실험을 기록했다. 시간 간격은6시간이며 완료 여부도 이 간격으로 확인한다.

대상은 EBM-NoRepeat 실험 `EBM_e50_bs240_lr1e5_B_zeropad_short303`, 실행루트 `/home/mentee/analysis/ebm_train_B_20261106_213627`이다. 예약을 위해 서버 학습이나 소스는 수정하지 않았다. 즉시 점검에서도1/50epoch37/108배치, loss 유한값, 오류표식없음, 실행/백업 source hash 일치, EBM-Repeat결과 불변을 확인했다.

점검 시 epoch/배치 증가, 로그 시각, 손실, GPU6/7과 parent/두 rank 상태, 종료코드, dev와checkpoint를 확인한다. 오류·정체를 보고하되 임의 중단·재시작·같은 이름 재실행·코드 수정·브랜치전환을 하지 않는다. 진행 파서가평가나완료상태를정확히나타내지못하면실제로그·metric·checkpoint로보완한다.

정상 종료를 확인하면 exit0·epoch50·CLS_50/TrainState_50·설정/소스를 검증한다. 기존 컨테이너와 `CUDA_VISIBLE_DEVICES=0,1`로 최종 고정50epoch checkpoint를 평가한다. EBM-NoRepeat train은dev만평가하므로 target 점수가 없다면 현재dataset코드의동일19LAeval·21LA·21DF trial 집합에서 평가를 한 번 실행한다. 평가 상태를 저장해 예약이중복실행하지않도록 한다. 원음·SSLhiddenstates는 내보내지 않고 UID/label/원길이/score 파생 기록만 분석한다.

분석은 학습/dev곡선, 최종 target EER, 길이별 정상거부·spoof수락·표본수를 포함한다. 길이구간은 원길이 기준 <2초,2~3.0275초,3.0275~4.0375초,4.0375초 이상이다. 각모델에같은dev threshold선택규칙을적용하고target에서 threshold나epoch를고르지않는다. 가능하면동일UID쌍의불확실성을보고하며빈구간과소표본을명시한다.

기존 `EBM_e50_bs240_lr1e5` 모델도 같은EBM-NoRepeat평가입력으로평가해학습변화와추론입력변화를구분한다. 예전반복평가EER는별도프로토콜로표기한다. 이전실행의codec여부미확정·sampler등의차이가남아단일비교로반복제거의인과효과를증명했다고표현하지않는다. 별도의A재학습이나새보조손실학습은이예약범위에추가하지않는다.

결과와검산자료를docs/experiments의보고서·필요한그림으로저장하고같은대화에보고한다. 완료분석과보고후예약을비활성화한다. 로컬예약이므로PC와앱을켜두고프로젝트·SSH접속을유지해야한다. [공식 예약 안내](https://learn.chatgpt.com/docs/automations)를확인했다.
