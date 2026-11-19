# 정상 전체 교차 BPL만 사용하는 모델 본학습

2026-11-20 사용자가 기존 평가 결과와 논리 검토 후 학습 진행을 승인했다. 구현 단계의 본학습 미시작 기록 이후, 검증된 동일149개 소스로 새 seed1·50epoch 학습을 시작했다. 검증 모델/optimizer/가중치는 재사용하지 않는다.

backend 손실은 기존 스푸프 중심 OC와 정상 전체 교차 BPL의 합이다. 스푸프 BPL 계수0·정상 BPL 계수1이며 pair target1/pull0.9/push0.2, alpha30/20, 중심 배율5, 원래240 batch/증강/반복자와 두 Adam 단계를 유지한다. 비교 기준은 스푸프/정상 모두 전체 교차 BPL인 v3이며 바뀐 학습 계수는 스푸프 BPL 1→0 하나다.

학습 시작 전 CPU163검사와 실제 두 GPU108loader/3갱신·normal144/144/36pairs·spoof0pairs·정상short양수/스푸프short정확0·저장복원·19LA64 독립ROC 검산 완료 기록을 확인했다. 새 실행에서149소스/11제어 및 이전1059보호파일의 해시와 검증 receipt signature가 일치한다. Fresh seed1 초기 파라미터/RNG 동등성을 확인하고, 실제 학습 양rank의 첫 갱신이 검증 첫 갱신과 정확히 일치하는지도 확인했다.

## 실행과 평가 조건

- 50epoch, seed1, global batch240/local120, Adam lr1e-5/weight decay1e-4.
- 5epoch마다 checkpoint 저장 후19LA 공식 eval 전체만 평가한다.
- 학습 종료 후21LA·DF 전체를 평가하며 batch마다 동결 XLSR 출력을 한번 계산해 저장10head에 함께 넣는다.
- wrap64600/no-mask/batch8/common matched_wrap1729/ASP1e-5, bigger score=bonafide.
- v6 전용 `finalize_bona_only.py`로 기존6모델 결과를 보존하면서 새 `bona_only` 열을 추가한다. 동일epoch/고정50/공통15~50 범위 및 전체 관측최저·epoch·후보수를 함께 제공한다.

원본 실행 경로는 plan.json과 pipeline.sh에 기록했다. 새 환경에서는 새 고유 root/result 경로를 설정하고 소스·제어를 다시 고정해야 한다. 기존 실행 root를 수정하거나 재시작하지 않는다.

학습 시작 검증 기록은 이 시점의 실행 확인이며 학습 완료나 최종 EER을 뜻하지 않는다. 첫 본학습 EER은5epoch19LA 전체평가 후에 나온다. ITW 결과를 새 학습이나 checkpoint 선택에 사용하지 않으며, 이번 pipeline에 새 ITW 평가는 포함하지 않는다.

기존 실제 평가/검증의 논리적 차단 문제는 발견되지 않았다. 목적식은 구면 임베딩 공간에서 양립 가능하지만 스푸프 OC의 수축 압력과 공유 파라미터 충돌 가능성은 남으며 성능 향상을 보장하지 않는다. 지출상한0USD·기존 할당GPU6/7만 사용·자동재시도0·전체pipeline168h상한이다.
