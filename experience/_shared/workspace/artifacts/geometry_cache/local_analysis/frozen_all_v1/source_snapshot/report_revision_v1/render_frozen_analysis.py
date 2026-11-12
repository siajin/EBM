"""Render a detailed, evidence-scoped Korean analysis from completed results.

Reads saved results only. No fits, inference, resubmission or process control.
The completion pipeline also uses this renderer, preserving the reviewed
interpretation when future experiment sections become available.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import hashlib
import pandas as pd
from polish_frozen_analysis import polish

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/geometry_cache/local_analysis/frozen_all_v1'
DOC=ROOT/'docs/experiments/ebm_frozen_all_results_20261105.md'

def read(name):return json.loads((OUT/name).read_text(encoding='utf8'))
def done(name):return (OUT/name).is_file()
def number(v,digits=6):return f'{float(v):.{digits}f}'
def ci(v,digits=6):return '['+', '.join(number(x,digits) for x in v)+']'
def table(headers,rows):
    def cell(v):return str(v).replace('|','/')
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(cell,row))+' |' for row in rows])
def link(label,path):return f'[{label}](../../artifacts/geometry_cache/local_analysis/frozen_all_v1/{path})'

def render():
    stamp=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M KST')
    assert done('heads_complete.json') and done('P3_P4/A/verification.json')
    n0=done('N0_complete.json');b_done=all(done('P3_P4/B/'+n+'/complete.json') for n in ['DF_pilot','21LA_pilot']);b_verified=done('P3_P4/verification.json')
    r1=all(done('R1/'+n+'/complete.json') for n in ['Q','19LA_eval','DF_full']);r23=all(done('R2_R3/'+n+'/complete.json') for n in ['DF_pilot','21LA_pilot'])
    n0_verified=done('verification.json') and 'N0_complete.json' in read('verification.json').get('completed_markers',[])
    fully=n0 and n0_verified and b_done and b_verified and r1 and r23
    native=read('native_baselines.json');heads=pd.read_csv(OUT/'head_metrics.csv');contrasts=read('head_paired_effects.json')
    lines=['# EBM 방향·채널 실험 결과와 연구 가설의 해석','',f'검토 기준: {stamp}. '+('F/P/R 전체 수행과 검수를 완료했다.' if fully else '**완료된 결과에 한정한 중간 분석이다. 진행 중인 실험의 최종 결과는 포함하지 않는다.**'),'',
    '## 1. 현재 결과가 말하는 것','',
    '**현재 표현에 추가로 읽을 수 있는 방향 신호가 있다는 DF 내부 진단은 남아 있다. 반면 두 pilot의 주 비교에서는 T에서 학습한 채널 gate의 전이 EER 우월성을 입증하지 못했다.** 두 문장은 모순되지 않는다. 같은 데이터셋 안에서 읽는 문제와 학습 도메인에서 다른 도메인으로 전이하는 문제가 다르기 때문이다.','',
    '- 기존 DF 내부 spline-linear probe에서는 방향 추가가 log-loss와 EER를 개선했다. 이는 DF 정답으로 작은 분류기를 학습한 정보 접근성 진단이다.',
    '- 새 P1의 dynamic gate는 DF 평균 EER가 3.350000%에서 3.342593%로 변했으나, native 대비 개선의 99% 구간이 0을 포함했다. 21LA pilot EER는 0.800000%로 같았다.',
    '- P2 supervised metric은 V에서 초기 identity가 선택됐다. 선택된 점수가 native와 같다는 결과이며, 모든 비등방성 거리의 가능성을 기각한 결과가 아니다.',
    '- P3 전이의 선형 방향 잔차는 balanced log-loss를 작게 줄였지만 DF EER는 seed평균 3.35%에서 3.40%로 나빠졌다. 따라서 “방향 사용은 아무 효과가 없다”도, “탐지 성능이 개선됐다”도 정확한 요약이 아니다.',
    '- 같은 scalar의 엄격한 matching은 DF27쌍,21LA10쌍으로 부족하다. F2의 결론은 방향 차이의 부재가 아니라 **표본 부족에 따른 판정 불가**다.',
    '- 방향은 출처·codec·길이/padding과도 연관된다. 현재 관찰만으로 어떤 축이 위조 고유 신호이고 어떤 축이 shortcut인지 인과적으로 나눌 수 없다.','',
    '아래 수치는 한 epoch50/batch160 checkpoint와 반복 관찰한 benchmark의 탐색 결과다. “실험 완료”와 “가설 지지”를 별도로 판정한다. 원래 EBM을 새 attention 또는 방향 loss로 공동 재학습한 통제 비교는 이번 묶음에 없다.','',
    '## 2. 무엇을 비교했는가','',
    '### 2.1 원래 점수와 추가 방향의 의미','',
    r'길이를 1로 정규화한 최종 표현을 (x=z/\|z\|), 학습된 정상 centroid를 (c)라 두면 원래 점수는 (s=x^Tc)다. 여기서 c도 단위 벡터이며 평가 데이터의 평균으로 바꾸지 않았다. 잔여 성분은 (r=x-sc), 잔여 방향은 (u=r/\|r\|)다. (\|r\|^2=1-s^2)이므로 잔여 크기는 s와 별개의 추가 정보가 아니다.','',
    '**원래 EBM이 방향이나 채널을 전혀 사용하지 않는다는 설명은 틀리다.** 코사인은 centroid와의 방향을 사용하는 점수이고, 원래 모델도 채널 혼합·layer attention·time attention을 학습한다. 정확한 차이는 마지막160차원에 입력별 channel gate가 없고, 최종 점수와 spoof OC 항이 동일 s의 잔여 방위를 별도로 구분하지 않는다는 것이다.','',
    '따라서 사용자의 새 방향 loss 아이디어는 “거리 대신 처음으로 방향을 쓴다”보다 **동일한 centroid 각도 안에서 남는 방위 구조를 어떻게 유용하게 학습할지**로 구체화하는 편이 정확하다. OC도 정상/스푸프의 centroid 각도를 이미 다르게 만들고, BPL은 정상 long/short에만 적용된다. 이 구조적 불변성이 곧 유용 정보 손실이나 성능 저하의 원인임을 뜻하지는 않는다.','',
    '### 2.2 평가 경로','',
    table(['경로','fit/선택','평가','답할 수 있는 질문'],[
      ['Native','추가 fit 없음','같은 표본의 기존 cosine','기준 모델의 점수 성능'],
      ['P1/P2/P3-A 전이','2019LA T에서 fit, V에서 선택','DF/21LA 각20,000개','학습 도메인에서 배운 새 readout이 전이하는가'],
      ['기존 cached_core 및 P3-B/P4','각 평가셋의 outer-train/inner에서만 fit·선택','같은 셋의 held-out speaker','현재 표현에서 다른 화자에게 읽히는 정보가 있는가'],
      ['F1/F3/F4','순위·공분산·PCA는 T, 설정·threshold는 V','평가셋의 고정 점수/통계','현재 점수의 좌표·거리·조건 의존성'],
      ['R2/R3','원래 EBM·새 head·기준 threshold 고정','attention 또는 동일 음성 입력 개입','학습된 함수가 개입에 얼마나 민감한가']]),'',
    '서로 다른 경로의 EER를 같은 모델 개선량으로 빼지 않는다. 특히 DF 내부에서 fit한 1.20%를 T→DF native3.35%와 비교해 배포 성능 개선으로 보고하면 학습 데이터 역할을 혼동하게 된다.','',
    '## 3. 데이터와 native 기준','',
    table(['자료','정상','스푸프','역할'],[
      ['T:2019LA train','2,580','22,800','전이 head·순위·공분산 fit'],['V:2019LA dev 선택 부분','1,351','13,632','설정·epoch·threshold 선택'],
      ['DF_pilot','2,000','18,000','주 평가, 기존에 관찰한 고정 표본'],['21LA_pilot','2,000','18,000','추가 평가, UID hash로 독립적으로 표본 선택'],
      ['21LA 전체','14,816','133,360','pilot을 포함하는 보조 평가'],['Q:2019LA dev 유보 부분','1,197','8,664','추가 적용 범위 확인'],['19LA eval','7,355','63,882','R1 확장 캐시·동결 head 평가 완료']]),'',
    'R2/R3의각2,000개(정상200/spoof1,800)는 해당pilot 안에 포함된다. 전처리 QC128개(32/96)는 V에서 정했다. 표본 선택에 점수나 h/z를 쓰지 않았다. DF/21LA는67개의 asvspoof 화자그룹을 공유하므로 두 평가셋을 완전히 독립된 반복으로 세지 않는다. 원발화·합성·재인코딩 family 연결은 미해결이다.','']
    rows=[]
    for name,r in native['datasets'].items():rows.append([name,number(r['eer_percent']),number(r['auroc'],9),number(r['balanced_logloss']),number(100*r['bona_false_reject'],4),number(100*r['spoof_false_accept'],4)])
    lines += [table(['같은20,000개 표본','EER(%)↓','AUROC↑','Balanced LL↓','V고정 정상거부(%)','V고정 spoof통과(%)'],rows),'',
    '전체21LA native EER는1.0535392921%다. pilot의0.80%와 표본 구성이 다르므로 어느 쪽이 다른 쪽을 대체하지 않는다.','',
    '운용 threshold는 V 정상 오거부율1% 이하 규칙으로 정했다. DF 정상 오거부39.00%,21LA12.40%는 EER가 아니라 이 고정 운용점의 결과다. EER가 낮아도 같은 threshold가 도메인 간에 잘 보정됐다는 뜻은 아니다. LL도 T에서 고정한 확률 링크를 쓰므로 ranking 지표와 calibration을 구분한다.','',
    '## 4. F0: 손실의 불변성이 증명한 범위','',
    '단위 centroid 주위에서 s와 norm을 유지한 채 잔여 방위만 회전하면 weighted OC 값은 바뀌지 않는다. float64 항등식과 방향미분을 확인한 구조적 사실이다. 정상 표본의 방향은 BPL에 영향을 줄 수 있지만 spoof만의 회전은 정상 쌍 BPL에 들어가지 않는다.','',
    '여기서 증명한 것은 embedding 공간에서 목적함수가 식별하지 않는 자유도다. 그 모든 회전 벡터가 encoder·LayerNorm을 거친 실제 음성 표현으로 실현된다는 증명, 실제 정보가 버려졌다는 증명, 새 loss가 더 잘 일반화한다는 증명은 아니다.','',
    '## 5. F1: 채널 선택과 소수 좌표 결과','',
    '### 5.1 이미 검수된 DF 부분집합 실험','',
    table(['점수 좌표','DF EER(%)','DF AUROC','해석'],[
      ['전체160','3.3500','0.995943','기준 cosine'],['T-Fisher 하위10','2.9444','개별 결과 파일 참조','특정 부분집합의 EER 지점 결과'],
      ['T-Fisher 하위2','2.9222','0.987659','낮은 EER와 낮은 AUC가 동시에 관찰됨'],['T-Fisher 하위1(j95)','4.345939','0.967472','1D cosine은 부호±1 두 점수']]),'',
    'Fisher는 T raw post-LN z의 정상/공격별 평균·분산으로 정한 단변량 순위다. “하위”는 T에서 이 기준이 낮다는 뜻이며 무의미한 좌표라는 뜻이 아니다. 같은 mask를 z/c에 적용하고 각각 재정규화한 cosine을 계산했다.','',
    '하위2개의 native 대비 EER 차이 구간은0을 포함했다. 상위2개 EER는6.85%,random2개20회의중앙값은5.5750%였다. 특정 두 좌표가 가능한 것과 아무 두 좌표가 충분한 것은 다르다. 하위2개의 V EER는0.1907%로 native0%보다 높았다.','',
    '전체160개와 하위2개의 ROC는 교차했다. 정상거부율1% 이하에서는 전체160개가,3% 부근에서는 두좌표가 더 유리했다. 같은 수치의 옛 native threshold를 부분집합에 그대로 적용한 오류율은 점수 분포 변화를 보여줄 뿐, 각 점수마다 V에서 재보정한 공정한 배포 비교는 아니다.','',
    '![기존 DF pilot에서 전체160개와 하위2개 좌표의 오류율 곡선](cached_core_20261105/figures_author/03_roc_and_operating_points.png)','',
    '그림은 기존 DF20,000개 부분집합 비교다. 새 gate·21LA·내부 P3-B/P4의 결과를 그린 그림이 아니다. 곡선의 운용점별 장단점을 보여주며,평가 표본에서 배포 threshold를 고른 결과로 읽지 않는다.','',
    '**현재 checkpoint가 만든160차원 표현의 일부 좌표만 읽어도 이 DF 표본의 EER가 낮은 점 추정치를 보였다**는 표현이 적절하다. “성능 동등성 입증”,“158개가 모두 잡음”,“모델의 본질적 차원이2”,“encoder 연산량 감소”로 확대하지 않는다. 전체 encoder와160차원LayerNorm은 여전히 계산했다.','',
    '### 5.2 신규 F1 전체 panel','']
    if n0 and done('F1/paired_effects.json'):
        rows=[[r['basis'],r['dataset'],r['action'],number(r['effect_pp']),ci(r['CI975'])] for r in read('F1/paired_effects.json')]
        lines += [table(['좌표계','평가셋','비교','효과(pp)','97.5% CI'],rows),'',
        'keep효과=random20개 평균EER−topEER,drop효과=topEER−random20개 평균EER다. random mask seed와speaker group을같이재표집했다. 회전좌표에서는 T순위를새로fit했으며, full cosine 불변성과좌표부분집합의의미를구분한다.']
    else:lines += ['169조건의원래좌표와직교회전3개의25%panel·10,000회speaker/mask재표집이진행중이다. 완료된옛DF 실험을인용하되,새전체panel이나회전대조가완료됐다고표시하지않는다.']
    lines += ['', '## 6. F2: 같은 점수·크기의 잔여 방향','',
    '옛source×codec matching106쌍과 새strict matching을 구분한다. 옛106쌍은 같은speaker를 요구하지 않았고caliper도0.01/0.002였다. 새F2는DF source×speaker×codec,21LA speaker×codec×transmission을같게 하고caliper0.02/0.002를사용한다. 표본수 차이를 모순이나 재현 실패로 해석하지 않는다.','']
    f2=read('F2/summary.json');rows=[]
    for r in f2:
        s=read(f"F2/{r['dataset']}_{r['condition']}_direction_statistics.json")
        rows.append([r['dataset'],r['condition'],r['pairs'],r['speaker_groups'],number(s['mean_direction_angle_degrees'],3),number(s['biased_MMD2']),r['status']])
    lines += [table(['평가셋','caliper','쌍','그룹','평균방향 각도(도)','biased MMD²','판정'],rows),'',
    '최소200쌍·10그룹을모든조건이충족하지못했다. 방향moment/MMD의계산은완료됐지만주가설판정은불가다. 예를들어biased MMD²의bootstrap구간이0보다크다고해서조건부방향차이가입증되는것은아니다. biased 추정량은유한표본에서양의기댓값을가질수있고,여기CI는관찰한두표본을재표집한효과변동구간이지동일분포null검정이아니다. 매칭도정확한s/norm일치가아닌근사다.','',
    '정상/스푸프의resultant length,centered covariance의participation ratio와uncentered2차moment의participation ratio도저장했다. 중심화유무가다른수치를같은“내재차원”으로읽지않는다. 작은선택표본의PR을전체표현의최소필요차원으로해석하지않는다.','',
    '## 7. F3/P2: 유클리드·공분산·학습 metric','',
    r'단위 x,c에서는 (\|x-c\|^2=2-2s)다. 따라서 단위공간의단순유클리드거리로바꾸기만하면순위·EER·AUROC는원래cosine과같다. raw z의거리나새중심은norm·center까지바꾸는별도실험이다.','',
    r'F3은T정상unit표현의공분산으로A를정하고 (S_A=1-\tfrac12(x-c)^TA(x-c))를썼다. 중심c는고정했다. diagonal/full과shrinkage후보를V EER로선택했으며,모두V EER0%여서사전에정한큰gamma우선규칙에따라identity(gamma1)가선택됐다.','']
    f3=pd.read_csv(OUT/'F3/condition_effects.csv');f3=f3[f3.condition=='all']
    lines += [table(['F3 후보','DF EER(%)','21LA EER(%)','선택 역할'],[[m,number(f3[(f3.dataset=='DF_pilot')&(f3.model==m)].iloc[0].eer_percent),number(f3[(f3.dataset=='21LA_pilot')&(f3.model==m)].iloc[0].eer_percent),'V선택/native' if m=='identity' else '탐색 후보'] for m in ['identity','diagonal_0.01','diagonal_0.1','diagonal_0.5','full_0.01','full_0.1','full_0.5']]),'',
    'diagonal gamma0.01의DF EER3.05%는관찰된후보값이다. native대비개선0.30pp의개별95%CI는[0.065818,0.391215]이지만8비교보정구간은[0,0.458775]로0을포함한다. V에서고른주모델도아니다. 그러므로“유클리드/마할라노비스가개선됐다”는확정결론으로쓰지않는다.','',
    'P2는정상공분산의비지도추정과달리정답을이용해bounded diagonal A를학습했다. 세seed 모두V에서epoch0이선택돼 최종A=I다. [0,0]개선CI는선택된점수가같다는대수적결과이지,어떤metric도소용없다는정밀한부정증거가아니다.','',
    '## 8. P1: static·dynamic gate와같은크기MLP','',
    '원래EBM/centroid를고정하고unit z뒤에붙였다. static은160파라미터,dynamic과잔차MLP는각5,296파라미터다. gate범위는0.5~1.5이며identity초기화와변환규제를썼다. 학습objective는class-balanced weighted OC이고,V선택은모든head에같은T고정링크를쓴balanced LL이다. 각seed의동일epoch/batch순서를공유했다.','']
    avg=heads[heads.dataset.isin(['DF_pilot','21LA_pilot'])].groupby(['dataset','model'],as_index=False)[['eer_percent','auroc','balanced_logloss']].mean()
    lines += [table(['평가셋','head','seed평균EER(%)','seed평균AUROC','seed평균LL'],[[r.dataset,r.model,number(r.eer_percent),number(r.auroc,9),number(r.balanced_logloss)] for r in avg.itertuples()]),'',
    'static과metric은epoch0,dynamic은epoch1,MLP는seed101/102/103순서로11/9/11epoch가선택됐다. 다음CI는각draw에서seed별paired EER차이를먼저계산한뒤평균한것이다. 예측평균ensemble의성능이아니다.','',
    table(['평가셋','baseline→candidate','평균EER개선(pp)','99% CI'],[[r['dataset'],r['baseline']+'→'+r['candidate'],number(r['mean_delta_EER_pp']),ci(r['CI99'])] for r in contrasts]),'',
    '양수는EER개선이다. dynamic대native의DF0.007407pp는실용효과참고값0.1675pp보다도작고CI가0을포함한다. DF LL은native0.563682보다dynamic평균0.565309로나빠졌다. 21LA EER는같고LL도소폭나빠졌다. 이조건의dynamic우월성은확인하지못했다. 다만동일성/비열등성을따로설계해입증한것은아니다.','',
    'T평균gate·실제gate·T에서정한score/norm구간안의gate재배치개입도저장했다. 이개입은비현실적인중간조합을만들수있으며gate값이나shuffle민감도만으로인과적채널중요도를정하지않는다.','',
    '**부정결론의범위는이bounded posthoc구조와fit/선택규칙에한정된다.** V가매우쉽고,원래표현을고정했으며,long평가캐시만사용했다. 새encoder/classifier와attention을공동학습하거나다른목적함수로학습한모든attention방식까지기각한것은아니다.','',
    '## 9. P3: 방향정보의접근성과전이','',
    '### 9.1 기존DF내부진단','',
    '기존cached_core_v1은DF20,000개에서5×3nested grouped CV를수행했다. spline(s)+lognorm+valid의LL0.098882/EER3.5056%에방향u를더하면LL0.038342/EER1.20%였고,같은차원noise는LL0.123376/EER4.20%였다. 출처·codec·길이·padding까지입력한대조에서도방향추가LL감소0.033961의개별95%CI는[0.021772,0.047048]이었다.','',
    '이결과는정한DF내부모델계열에서추가방향관련신호가held-out speaker에게읽힌다는근거다. u는z의결정적변환이므로새정보를생성한것은아니고,특징변환이probe의접근성·조건수를바꾼영향도가능하다. 정확한conditional mutual information이나모든nuisance제거를증명하지않는다.','',
    '### 9.2 새T/V전이평가','',
    '새P3은s에대해고정norm에서단조인기본logit에선형또는hidden16잔여방향을더했다. norm/valid가다르면전체순위는바뀔수있다. ridge두후보·epoch0포함V선택·seed3개를사용했고,s/n을잔차입력에다시넣지않았다. 기존DF spline진단과모델·fit자료·규제·선택규칙이다르다.','']
    for name in ['DF_pilot','21LA_pilot']:
        df=pd.read_csv(OUT/'P3_P4/A'/name/'metrics.csv').groupby('variant',as_index=False)[['eer_percent','balanced_logloss']].mean()
        lines += [f'**{name} 전이**','',table(['입력/잔차','seed평균EER(%)','seed평균LL'],[[r.variant,number(r.eer_percent),number(r.balanced_logloss,9)] for r in df.itertuples()]),'']
    effects=[dict(dataset=n,**r) for n in ['DF_pilot','21LA_pilot'] for r in read(f'P3_P4/A/{n}/paired_effects.json') if r['baseline']=='scalar' and r['candidate']=='direction_linear']
    lines += [table(['평가셋','LL감소','개별95%LL CI','EER개선(pp)','개별95%EER CI'],[[r['dataset'],number(r['mean_balanced_LL_reduction'],9),ci(r['LL_CI95'],9),number(r['mean_delta_EER_pp']),ci(r['EER_CI95'])] for r in effects]),'',
    '선형방향잔차의LL감소는각각0.002387855와0.000216622로개별95%구간이0보다크다. 따라서작은확률예측개선은기록해야한다. 반면DF EER는0.05pp악화,21LA는동일하다. 이신뢰구간은고정된fit·선택에조건부이고다수의전체탐색을보정한것은아니다.','',
    '세seed 중선형direction의seed101만V에서비영잔차가선택됐고다른두seed는epoch0이었다. scalar/noise/MLP잔차도최종선택은epoch0이다. LL효과가세번독립적으로같이재현됐다고쓰지않는다.','',
    '### 9.3 신규데이터셋내부CV','']
    if b_done and b_verified:
        for name in ['DF_pilot','21LA_pilot']:
            df=pd.read_csv(OUT/'P3_P4/B'/name/'metrics.csv').groupby('variant',as_index=False)[['eer_percent','balanced_logloss']].mean();df=df[df.variant.isin(['scalar','direction_linear','noise_linear','direction_mlp','noise_mlp'])]
            lines += [f'**{name} 내부CV,전이와분리**','',table(['probe','seed평균EER(%)','seed평균LL'],[[r.variant,number(r.eer_percent),number(r.balanced_logloss)] for r in df.itertuples()]),'']
        lines += ['정확한paired효과와LL/EER구간은각셋B/paired_effects.json에있다. 같은fold·같은모델계열의scalar/noise대조에대한증분을판정하고,원래native를별도기준으로확인한다. 내부CV가좋더라도T→평가셋의전이개선과동일시하지않는다.']
    else:lines += ['DF와21LA각2만개에서별도의5outer×3inner speaker-grouped CV를실행중이다. 모든seed·fold·variant와독립검수가끝나기전에는일부완료fold를합쳐전체EER나최종결론을만들지않는다. 이새모델계열의재현여부와21LA내부방향접근성은아직판정하지않는다.']
    lines += ['', '## 10. P4: LayerNorm이정보를없앴는가','',
    '기존DF내부동일L2 logistic비교에서scalar+h는LL0.040671/EER1.4944%,scalar+z는LL0.064016/EER1.7000%였다. LL감소0.023345의개별95%CI는[0.010456,0.042250]이었다. 이수치는원래centroid cosine이아닌새probe점수의결과다.','',
    'h는최종Linear출력,z는그뒤LayerNorm출력이다. h160을z160대신읽었을때정한probe에더유리했다는근거는있다. 그러나h가z에비해얼마나많은정보를가지는지,그차이가기존탐지기의오류원인인지까지이결과로측정하지못한다.','',
    'LayerNorm은발화내공통shift에는정확히불변이다. 양의scale에는epsilon>0일때일반적으로정확히불변하지않다. 현재h분산은epsilon1e-12보다매우커서scale의영향이강하게정규화된다. “평균·분산정보가모두수학적으로완전히없어진다”는표현은피한다.','']
    if b_done and b_verified:
        for name in ['DF_pilot','21LA_pilot']:
            df=pd.read_csv(OUT/'P3_P4/B'/name/'metrics.csv').groupby('variant',as_index=False)[['eer_percent','balanced_logloss']].mean();df=df[df.variant.isin(['direction_linear','direction_hstat','direction_noise2','direction_h','direction_noise160'])]
            lines += [f'**{name}의scalar+u증분비교**','',table(['probe','seed평균EER(%)','seed평균LL'],[[r.variant,number(r.eer_percent),number(r.balanced_logloss)] for r in df.itertuples()]),'']
    else:lines += ['신규P4의scalar+u+h-stat/h160대scalar+u+동일차원noise2/160비교는아직최종결과가없다. 기존h/z비교가이증분대조까지완료한것처럼쓰지않는다.']
    lines += ['', '## 11. F4: 클래스구조와데이터조건','',
    '기존T-fit K=4방향군집에서는DF label NMI0.264455,source0.118216,codec0.004659였다. 정상내부source NMI는0.343788이었다. 새F4는군집번호로축약하지않고u전체의class별벡터분산비율eta²를계산했다. NMI와eta²는정의가달라서두수치의크기를직접비교하지않는다.','']
    eta=pd.read_csv(OUT/'F4/vector_variance.csv');eta=eta[eta.stratum=='all']
    lines += [table(['평가셋','class','조건','vector eta²'],[[r.dataset,'정상' if r.y==1 else 'spoof',r.factor,number(r.eta_squared)] for r in eta.itertuples()]),'',
    'DF정상source eta²0.307754와spoof padding eta²0.163591,21LA정상codec eta²0.168577와spoof padding eta²0.158263이관찰됐다. 이는해당조건별평균차이가관찰된방향분산에서차지하는비율이다. 조건을조작했을때제거되는인과적분산비율이나독립적인원인기여도는아니다.','',
    'T-fit PCA1과spoof padding의Spearman은DF−0.600272,21LA−0.577569였다. duration은각각+0.601706,+0.576446이었다. PCA축부호는임의적이고duration/padding은고정crop규칙으로연결되므로두개의독립발견으로세지않는다.','',
    'transmission별codec분석에서codec범주가하나뿐이면eta²0은“그stratum에서비교할codec변이가없음”을뜻한다. codec이영향없다는음향적증명이아니다. 각cell의정상/spoof수·화자수·20개미만표시와조건별native오류도저장했다.','',
    '기존metadata포함probe에서방향이추가로도움됐다고해도측정하지않은조건·조건간비선형관계를모두제거한것은아니다. 반대로조건연관이있다고해서해당방향이전부불필요한shortcut이라는뜻도아니다. 동일음성개입R3가이관찰을보완한다.','',
    '## 12. R1/R2/R3: 새추론결과의상태','',
    '### 12.1 평가범위확장','']
    rows=[]
    expansion_tables=[]
    for name in ['Q','19LA_eval','DF_full']:
        if done('R1/'+name+'/complete.json'):
            df=pd.read_csv(OUT/'R1'/name/'metrics.csv');n=df[df.model=='native'].iloc[0];rows.append([name,int(n.n),number(n.eer_percent),number(n.auroc,9),'완료·독립metric검수'])
            mean=df.groupby('model',as_index=False)[['eer_percent','balanced_logloss']].mean()
            expansion_tables += [[name,r.model,number(r.eer_percent),number(r.balanced_logloss)] for r in mean.itertuples()]
        else:rows.append([name,'전체준비/추출/검수중','—','—','최종평가미완료'])
    lines += [table(['자료','행수','native EER(%)','native AUROC','상태'],rows),'',
    '완료된 확장 자료에는 같은 V 동결 head를 적용했다. 아래는 seed별 지표의 평균이며 ensemble 성능이 아니다. 확장 자료로 다시 fit하거나 가장 좋은 후보를 고르지 않았다.','',
    table(['확장 자료','head','seed평균 EER(%)','seed평균 balanced LL'],expansion_tables),'',
    '이 표의 미세한 차이는 보조 적용의 점 추정치다. 새 독립 반복이나 확장 자료의 유의성 검정으로 읽지 않는다. EER와 LL의 방향이 다르면 각각 기록하며,주 P1/P2 비교의 구간은 앞의 두 pilot 결과를 따른다.','',
    '각완료자료에는T/V에서이미고른head들을추가fit없이적용했다. Q특징추출은T/Vhead실행과병행했지만Q결과로선택하지않았다. DF전체는기존20,000개와미추출513,928개의전처리/RNG필드를맞춰533,928개로결합한다. 아직추출중인행을성능표본수로세지않는다.','',
    '전체21LA에서dynamic의seed평균EER는1.053539%로native와같고,MLP는1.053289%였다. 이극소차이를새개선근거로선택하지않는다. 전체평가는pilot을포함하는보조적용이며새독립반복이아니다.','',
    '### 12.2 원래attention제거·입력변형','']
    if r23:
        for name in ['DF_pilot','21LA_pilot']:
            df=pd.read_csv(OUT/'R2_R3'/name/'metrics.csv');df=df[(df.subset=='pilot20000')&~df.condition.str.contains('_dynamic_')]
            lines += [f'**{name}**','',table(['조건','EER(%)','AUROC'],[[r.condition,number(r.eer_percent),number(r.auroc,9)] for r in df.itertuples()]),'']
        lines += ['같은발화의paired score/방향각/gate변화와운용오류및구간을R2_R3에저장했다. attention제거의함수민감도와입력perturbation의안정성을보여주며,제거모델을새로학습했을때의성능이나새loss의학습효과를증명하지않는다.']
    else:lines += ['R2의layer/time attention2×2네조건과R3의동일base noise20/10dB·Opus32k재압축은대기/실행단계이며최종분석자료가아직없다. 원래attention을제거해도괜찮다거나,noise/codec에방향이안정적이라고결론내리지않는다. V128개전처리QC와native복제·원래state불변을통과한완료결과만추후반영한다.']
    lines += ['', 'Opus조건은이미압축된음성의추가재압축일수있다. small2,000개는pilot에포함되고개입세조건도같은원발화를공유한다. 이들을독립표본으로합산하지않는다.','',
    '## 13. 연구아이디어별판정과다음해석','',
    table(['가설/아이디어','현재판정','이유와남은질문'],[
      ['원래loss가같은s의spoof방위를별도로제약하지않음','구조적으로확인','embedding불변성;실제음성상의유용성·원인증명은별도'],
      ['현재z에추가방향신호가남음','DF내부지정probe에서지지','noise/metadata대조의LL이득;모든nuisance제거·CMI증명아님'],
      ['dynamic channel gate가전이EER를개선','현재bounded방식에서는미지지','작은DF점추정치·CI0포함,21LA같음'],
      ['방향readout이전이에아무효과없음','그렇게단정불가','P3선형잔차LL은소폭개선;EER는개선안됨'],
      ['단순unit Euclidean이cosine보다좋음','동일순위라별도개선불가','비등방성/새중심/rawnorm은다른가설'],
      ['학습metric으로거리의이점을확인','V선택결과에서는미지지','epoch0선택;비identity후보의가능성전부기각아님'],
      ['LayerNorm이성능저하의원인','미입증','h의probe접근성이좋았음;새증분대조·통제개입필요'],
      ['2차원만으로모델전체가충분','미입증','전체encoder뒤의특정점수좌표실험,ROC·V손해존재'],
      ['새방향loss/공동attention이유리','아직학습효과미시험','동결readout과표현공동학습은다름']]),'',
    '사용자의“원래모델이그정보를활용하도록학습되지않아서동결후처리가불리했을수있다”는설명은가능한가설이다. 그러나현재결과만으로그것을원인으로선택할수없다. domain shift,V선택력,제한된변환폭,objective/regularization,표현의배치·nuisance와다른설명도남는다. 이를구분하려면같은초기화·데이터·seed·optimizer·선택규칙아래에서표현공동학습비교가필요하다.','',
    '새loss를설계할때는여러spoof공격을하나의임의방향으로모으는것이목표인지,정상의국소분포에서멀어지게하는것인지,공격별여러방위를허용하는것인지부터구분해야한다. 현재실험은이선택중하나가좋다는증거를제공하지않았다.','',
    '## 14. 통계·검수·근거파일','',
    'P1/P2주5비교는개별99%CI로이번family의95%범위를보수적으로관리했다. F3의후보8개는별도보정,기존cached_core는2,000회개별95%구간,새F/P는10,000회source×speakerpaired재표집이다. 서로다른구간수준·비교family를혼용하지않는다.','',
    'CI는고정된T순위·선택된head·저장된OOF예측에조건부다. 재표집마다전체fit/CV/모델선택을다시하지않으므로학습표본및선택의전체불확실성은포함하지않는다. 세seed만으로학습변동의모집단을정밀하게추론하지않는다. 이미반복탐색한DF와과거benchmark관찰을독립확인으로되돌릴수도없다.','',
    f"완료범위의새검수는{read('verification.json')['checked_metric_rows']}개metric/{read('verification.json')['checks']}check, P3전이는{read('P3_P4/A/verification.json')['metrics_checked']}개metric/{read('P3_P4/A/verification.json')['candidate_fit_checks']}후보fit을독립확인했다. 독립NumPyhead수식,full tied ROC EER,rank AUC,ID정렬·V선택을사용한다. "+('P3-B/P4의독립검수도완료했고각fold의선택·scaler·refit·OOF예측을대조했다. R1/R2/R3의검수범위는각완료JSON에있다.' if fully else '이수치는현재완료된검수범위이며,진행중F1/P3-B/P4/R2/R3까지모두검증했다는뜻이아니다.'),'',
    '- '+link('새실험검수','verification.json')+'; '+link('P3전이검수','P3_P4/A/verification.json'),
    '- '+link('P1/P2모든seed·데이터지표','head_metrics.csv')+'; '+link('P1/P2주비교CI','head_paired_effects.json'),
    '- '+link('F2표본coverage/완료판정','F2/summary.json')+'; '+link('F3선택기록','F3/metric_registry.json')+'; '+link('F3보정CI','F3/paired_effects.json'),
    '- '+link('F4벡터분산비율','F4/vector_variance.csv')+'; '+link('F4길이/padding','F4/padding_duration.csv')+'; '+link('F4조건별오류','F4/condition_metrics.csv'),
    '- [기존DF내부방향/LayerNorm지표](../../artifacts/geometry_cache/local_analysis/cached_core_v1/probe_metrics.csv); [그림·개별문서목록](cached_core_20261105/README.md)',
    '- [실행명세와완료표](ebm_frozen_model_experiments_20261105.md); [원래공동학습명세](ebm_embedding_geometry_spec_20261105.md)','',
    '해당검토는저장된결과를재해석한것이며실행중인실험을중단·재시작하거나추가EBM학습을하지않았다. 원음·연구실원본소스·checkpoint를새로로컬에복사하지않았다.']
    DOC.write_text(polish(('\n'.join(lines)+'\n').replace('\\\\','\\')),encoding='utf8')
    record={'KST':stamp,'all_experiments_complete':fully,'scope':'completed evidence only; includes reviewed scientific interpretations','path':str(DOC),'sha256':hashlib.sha256(DOC.read_bytes()).hexdigest()}
    (OUT/'analysis_report_render.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return record

if __name__=='__main__':print(json.dumps(render(),ensure_ascii=False))
