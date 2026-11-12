"""Refresh the requested checklist strictly from actual completion evidence."""
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import json
from frozen_common import DEFAULT_ROOT,save

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'docs/experiments/ebm_frozen_model_experiments_20261105.md'
PROTOCOL=DOC.with_name('ebm_frozen_model_protocol_20261105.json')
OUT=DEFAULT_ROOT/'local_analysis/frozen_all_v1'

def main():
    stamp=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M KST')
    done=lambda f:(OUT/f).is_file()
    n0=done('N0_complete.json');head=done('heads_complete.json');a=done('P3_P4/A/complete.json')
    b=all(done('P3_P4/B/'+n+'/complete.json') for n in ['DF_pilot','21LA_pilot'])
    r1=all(done('R1/'+n+'/complete.json') for n in ['Q','19LA_eval','DF_full'])
    r23=all(done('R2_R3/'+n+'/complete.json') for n in ['DF_pilot','21LA_pilot'])
    r23state='완료' if r23 else ('실행 중' if done('R23_submission/launched.json') else '실행 대기')
    items=[
      ('F0','완료','기존 항등식·weighted OC·gradient 검수 완료','없음'),
      ('F1','완료' if n0 else '실행 중','명세의169조건 및 회전 좌표3개 계산·통계 실행','없음' if n0 else '공유 speaker/mask bootstrap10,000회와 전체 검수'),
      ('F2','완료' if done('f2_complete.json') else '실행 중','주 matching DF27쌍/9그룹,21LA10쌍/6그룹; 넓은 caliper 민감도·MMD·방향 moment 포함','표본 기준 미달이므로 결론은 판정 불가; 실행 실패와 구분'),
      ('F3','완료' if done('f3_complete.json') else '실행 중','T 정상 covariance·V 선택·두2만 개 평가 및 CI; V 선택은 양쪽 metric 모두identity','없음' if done('f3_complete.json') else '평가·통계'),
      ('F4','완료' if done('f4_complete.json') else '실행 중','두 평가셋의 class별 vector eta²·padding/duration·조건별 native 오류','없음' if done('f4_complete.json') else '조건별 통계'),
      ('P1','완료' if head else '실행 중','static/dynamic/동일 크기MLP×seed3; T학습/V선택·gate개입·두 평가셋 및 전체21LA 보조 평가','없음' if head else '공유 bootstrap99% CI'),
      ('P2','완료' if head else '실행 중','supervised diagonal metric×seed3; V에서는epoch0 identity가 선택됨','없음' if head else '학습·평가·통계'),
      ('P3','완료' if a and b else '실행 중',('T/V 전이 평가 완료; ' if a else '')+'새 단조 scalar·linear/MLP direction/noise와 데이터셋별5×3 grouped CV','없음' if a and b else '현재 실행 중인 내부CV·OOF 및 통계 완료'),
      ('P4','완료' if b else '실행 중/순차 대기','P3과 같은fold에서 h-stat/h160 증분 및2/160차원 noise 대조','없음' if b else 'P3 공통 baseline 뒤의 CV·OOF 및 통계'),
      ('R1','완료' if r1 else '실행 중','21LA 전체 native 평가 완료; Q·19LA·DF 전체 확장','없음' if r1 else '전체 캐시 검수와 V동결 방법 적용·DF pilot 결합'),
      ('R2',r23state,'두2만 개(각처음2천 개)의4조건 sidecar와 native대응 검사;R1 종료 후 실행','없음' if r23 else '실제 추론·native복제 확인·paired 평가'),
      ('R3',r23state,'V-QC128 ID잠금, 같은base noise20/10dB·Opus32k 재압축의 두2만 개 작업','없음' if r23 else 'V전처리QC·실제 추론·gate/방향/오류 안정성 평가'),
    ]
    lines=['## 0. 진행 체크리스트','',stamp+' 실제 완료 marker 기준. **체크는 실험 수행 완료이며 성능 개선/가설 입증과 다르다.** 사용자의 후속 요청에 따라 미완료 동결 실험 전체를 실행 중이다.','',
        '### native 평가 및 준비','',
        '- [x] 기준 EBM 구조·손실과 T/V/DF·21LA 캐시 검수.',
        '- [x] 21LA 전체148,176개 native 평가: EER1.0535393%.',
        '- [x] 같은20,000개씩의 native 평가: DF3.35%,21LA0.80%.',
        '- [x] DF/21LA 각2만 개, 소규모각2천 개와 V-QC128개 ID잠금.',
        '- [x] 명세·수학/gradient·이전 결과 재사용 점검.','',
        '### F/P/R 실행','',
        '| ID | 진행 | 실제 범위 | 남은 작업/판정 |','| --- | --- | --- | --- |']
    for id,state,scope,remaining in items:lines.append(f'| {id} | {"✅" if state=="완료" else "⏳"} {state} | {scope} | {remaining} |')
    lines += ['','진행 결과: [이번 실제 실행 기록](ebm_frozen_all_results_20261105.md). 완료 근거는 `artifacts/geometry_cache/local_analysis/frozen_all_v1/`의 실험별 완료 JSON과 `verification.json`에 기록한다. 서버 raw 추출이 끝나기 전에는 부분 파일을 완료 캐시로 등록하지 않는다.','',
        '현재 실행 범위는 기존 EBM을 고정한 F/P/R 전체다. 작은 gate/metric/probe의 학습은 포함하며, 새 attention/방향 loss로 EBM 자체를 공동 재학습하는 N3/B/G는 이 묶음에 포함하지 않는다.','']
    text=DOC.read_text(encoding='utf8');start=text.index('## 0. 진행 체크리스트');end=text.index('\n## 1.',start);DOC.write_text(text[:start]+'\n'.join(lines)+text[end:],encoding='utf8')
    p=json.loads(PROTOCOL.read_text(encoding='utf8'));p['scope']='Frozen EBM F0-F4/P1-P4/R1-R3 execution authorized on2026-11-05; native-only substeps counted complete separately'
    p['execution_progress_v1']=dict(updated_KST=stamp,output=str(OUT.relative_to(ROOT)).replace('\\','/'),experiments=[dict(id=i,state=s,scope=c,remaining=r) for i,s,c,r in items],native_evaluations_complete=['DF_pilot20k','21LA_pilot20k','21LA_full'],EBM_joint_retraining=False,all_experiments_completed=all(s=='완료' for _,s,_,_ in items))
    # Keep the earlier evidence-only checklist as a dated historical snapshot.
    if 'progress_checklist' in p:p['progress_checklist']['historical_snapshot_superseded_by']='execution_progress_v1'
    save(PROTOCOL,p);print(stamp,[(i,s) for i,s,_,_ in items])

if __name__=='__main__':main()
