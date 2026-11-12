"""Compose the final Markdown from completed outputs without new experiments."""
from pathlib import Path
import json
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from frozen_common import DEFAULT_ROOT,save,digest

ROOT=Path(__file__).resolve().parents[2];OUT=DEFAULT_ROOT/'local_analysis/frozen_all_v1';DOC=ROOT/'docs/experiments/ebm_frozen_all_results_20261105.md'

def table(frame,columns,rename=None):
    frame=frame[columns].copy()
    if rename:frame=frame.rename(columns=rename)
    def cell(value):
        if isinstance(value,float):return f'{value:.6f}'
        return str(value).replace('|','/')
    return '\n'.join(['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(columns))+' |']+['| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])

def main():
    assert (OUT/'N0_complete.json').exists() and (OUT/'heads_complete.json').exists()
    assert json.loads((OUT/'verification.json').read_text())['status']=='pass_within_completed_scope'
    assert json.loads((OUT/'P3_P4/verification.json').read_text())['status']=='pass'
    stamp=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M KST')
    lines=['# 고정 EBM 전체 실험 결과','',stamp+' 실제 완료 결과. **F0~F4/P1~P4/R1~R3 전체 수행·산출물 검수를 마쳤다.** 완료 표시는 가설의 입증과 다르다. 원래 epoch50/batch160 EBM과 centroid는 고정했고 새 gate/metric/probe만 제한적으로 fit했다. N3/B/G 공동 재학습은 수행하지 않았다.','',
        '주 결과는 DF·21LA의 고정2만 개씩이다. 각2천 개는 그 안에 포함되고,21LA 전체도pilot을 포함한다. DF/21LA는67화자그룹을 공유하며 원발화family는 미해결이다. 이미 관찰한 benchmark를 완전한blind test로 부르지 않는다.','',
        '## native 평가','']
    native=json.loads((OUT/'native_baselines.json').read_text());df=pd.DataFrame([dict(dataset=n,**r) for n,r in native['datasets'].items()]);lines += [table(df,['dataset','n','n_bona','n_spoof','eer_percent','auroc']), '',
        '## F1 채널 의존성과 좌표계 대조','',
        'T의 raw post-LN 공격 macro-Fisher 순위를 고정했다. 원래 좌표169조건,회전좌표3개각43조건을 계산했다. 주25%비교는20개random mask평균 EER와top mask의 차이다. keep효과는random−top,drop효과는top−random이며 양수일수록top선택의집중가설방향이다. 평균score의EER로바꾸지않았고speaker/mask를같이재표집했다.','']
    effects=pd.DataFrame(json.loads((OUT/'F1/paired_effects.json').read_text()));lines += [table(effects,['basis','dataset','action','effect_pp','CI975']), '',
        '특정 좌표번호의 효과를 물리적 의미나 dynamic attention의 필요성과 동일시하지 않는다. 회전은embedding좌표의 분석이며 실제음성개입이아니다.','',
        '## F2 같은 scalar에서의 방향 분포','']
    f2=pd.DataFrame(json.loads((OUT/'F2/summary.json').read_text()));lines += [table(f2,['dataset','condition','pairs','speaker_groups','status']),'',
        '최소200쌍·10그룹 기준에 미달한 조건은 판정불가다. 실행은 끝났지만 방향정보가 없다는 증거는 아니다. 근사matching의MMD를정확한conditional-independence검정으로해석하지않는다.','',
        '## F3 거리와 F4 데이터 조건','',
        'F3의 diagonal/full shrinkage는T정상에서만추정하고V EER로선택했다. 두계열모두gamma1 identity가선택됐다. 전체후보·조건별평가·10,000회paired CI는F3산출물에보존했다. F4의class별vector eta²·padding/duration 연관·조건별오류와작은cell표시를F4에보존했다. 관찰적codec차이는인과효과의증명이아니다.','',
        '## P1/P2 동결 표현 위 학습','']
    heads=pd.read_csv(OUT/'head_metrics.csv');headtable=heads[heads.dataset.isin(['DF_pilot','21LA_pilot'])].groupby(['dataset','model'],as_index=False)[['eer_percent','balanced_logloss']].mean();lines += [table(headtable,['dataset','model','eer_percent','balanced_logloss']), '','seed별EER의평균이며ensemble한EER가아니다. static/metric은V에서epoch0 identity가선택됐다. dynamic은1epoch,MLP는9~11epoch다.','']
    effects=pd.DataFrame(json.loads((OUT/'head_paired_effects.json').read_text()));lines += [table(effects,['dataset','baseline','candidate','mean_delta_EER_pp','CI99']), '',
        '주P1/P2 비교5개에는99%paired CI를적용했다. 이는고정fit·선택에조건부이며이전탐색전체를통제하거나학습표본불확실성까지포함하지않는다.','',
        '## P3 방향 probe와 P4 LayerNorm 이전 정보','',
        '경로A는T학습/V선택후전이,경로B는각평가셋내부의별도5outer×3inner speaker-grouped CV다. B의scaler·ridge·epoch는fold안에서만결정했다. B가좋아도원래EBM의전이성능개선으로부르지않는다.','']
    for route in ['A','B']:
        records=[]
        for name in ['DF_pilot','21LA_pilot']:
            frame=pd.read_csv(OUT/'P3_P4'/route/name/'metrics.csv');frame['dataset']=name;records.append(frame)
        averaged=pd.concat(records).groupby(['dataset','variant'],as_index=False)[['eer_percent','balanced_logloss']].mean()
        lines += ['### 경로 '+route,'',table(averaged,['dataset','variant','eer_percent','balanced_logloss']),'']
    lines += ['h-stat/h160의추가효과는같은scalar+u에2/160차원noise를추가한용량대조와비교한다. probe하나로정보량의완전한손실,채널attention의우월성,새방향loss의학습효과를증명하지않는다. 관련paired LL/EER CI는각경로산출물에있다.','',
        '## R1 평가 범위 확장','']
    r1=[]
    for name in ['Q','19LA_eval','DF_full']:
        folder=OUT/'R1'/name;assert (folder/'complete.json').exists();frame=pd.read_csv(folder/'metrics.csv');frame['dataset']=name;r1.append(frame)
    rt=pd.concat(r1).groupby(['dataset','model'],as_index=False)[['n','eer_percent','auroc']].mean();lines += [table(rt,['dataset','model','n','eer_percent','auroc']), '',
        'DFfull533,928개는기존pilot20,000개와새513,928개의전처리/RNG필드를맞춰결합했다. 이자료에서방법을다시fit하거나선택하지않았다. 전체21LA의동결head평가는head_metrics.csv에포함한다.','',
        '## R2 attention 제거와 R3 입력 변형','',
        '원래layer/time attention2×2 네조건,동일발화noise20/10dB·Opus32k재압축세조건이다. V128개의전처리QC를먼저통과했고각평가셋의처음2천개를거쳐2만개까지수행했다. native복제오차≤1e-5,원래모델state hash전후동일을검사했다.','']
    for name in ['DF_pilot','21LA_pilot']:
        folder=OUT/'R2_R3'/name;assert (folder/'complete.json').exists();metrics=pd.read_csv(folder/'metrics.csv');metrics=metrics[(metrics.subset=='pilot20000')&~metrics.condition.str.contains('_dynamic_')]
        lines += ['### '+name,'',table(metrics,['condition','eer_percent','auroc','bona_false_reject','spoof_false_accept']),'']
    lines += ['paired score차이·방향각·unit-z 변화·PH2 gate변화·V고정오류및CI를R2_R3에보존했다. attention제거는OOD중간표현을만들수있고,Opus는기존압축음성의추가재압축일수있다. 개입조건수만큼독립발화수가늘어난것은아니다.','',
        '## 검수와 재사용','',
        '완료산출물과독립계산은`artifacts/geometry_cache/local_analysis/frozen_all_v1/verification.json`, `P3_P4/verification.json`,각R1/R2_R3의complete.json에있다. 지표는full tied ROC/독립rank AUC,head는독립NumPy수식,OOF는fold내선택·scaler·refit·예측대응을검사했다. 원음·연구실원본소스·checkpoint는로컬로가져오지않았다.','',
        '실행코드와UID/fingerprint·RNG·fold·threshold·head상태를보존했으므로추가재학습이나음성추론없이후속분석에서재사용할수있다.']
    DOC.write_text('\n'.join(lines)+'\n',encoding='utf8')
    save(OUT/'final_report.json',dict(status='complete',KST=stamp,path=str(DOC),sha256=digest(DOC)))
    print('Final result report saved',DOC)

if __name__=='__main__':main()
