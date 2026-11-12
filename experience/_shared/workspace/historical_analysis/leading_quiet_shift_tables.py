from pathlib import Path
import json,hashlib
import numpy as np

assert Path('/.dockerenv').exists()
root=Path('/results/analysis/EBM80_21DF_leading_quiet_paired')
out=root/'shift_comparison'
out.mkdir(exist_ok=False)
d=np.load(root/'paired_scores.npz'); prior=json.loads((root/'summary.json').read_text())
verify=json.loads((root/'verification.json').read_text()); assert verify['all_checks_passed']
s=d['scores'].astype(np.float64); y=d['labels']; b=y==1; src=d['sources']; names=d['names'].tolist()
assert s.shape==(2000,13) and b.sum()==1000
base=Path('/results/analysis/EBM80_21DF_centroid')
assert json.loads((base/'sample_manifest.json').read_text())==json.loads((root/'sample_manifest.json').read_text())
old=np.load(base/'embeddings.npz')['cosine']; tail=b&(old<.9); core=b&~tail
assert tail.sum()==163 and core.sum()==837
threshold=prior['threshold']; eps=prior['meaningful_delta_epsilon']
families={'zero20':(2,3,4),'ownquiet20':(5,6,7),'variable20':(8,9,4),'zero40':(10,11,12)}
def angles(c): return np.degrees(np.arccos(np.clip(c,-1,1)))
def bands(c):
    # Columns/rows run from close to OC center to the opposite decision side.
    return np.where(c>=.9,0,np.where(c>=threshold,1,2))
def description(c,mask):
    a=c[mask]; yy=y[mask]
    return {'n':int(mask.sum()),'cosine_mean':float(a.mean()),'cosine_median':float(np.median(a)),
      'cosine_q10':float(np.quantile(a,.1)),'cosine_q90':float(np.quantile(a,.9)),
      'angle_to_center_mean_deg':float(angles(a).mean()),
      'distance_to_center_mean':float(np.sqrt(np.maximum(0,2-2*a)).mean()),
      'center_region_ge09_n':int((a>=.9).sum()),'outside_center_region_lt09_n':int((a<.9).sum()),
      'bonafide_decision_n':int((a>=threshold).sum()),'spoof_decision_n':int((a<threshold).sum()),
      'wrong_n':int(np.sum((a>=threshold)!=(yy==1))),
      'region_counts':np.bincount(bands(a),minlength=3).tolist()}
def movement(before,after,mask):
    a=before[mask];z=after[mask];delta=z-a;yy=y[mask]
    closer=delta>eps; away=delta<-eps; same=~(closer|away)
    good0=(a>=threshold)==(yy==1); good1=(z>=threshold)==(yy==1)
    transition=np.zeros((3,3),dtype=int)
    np.add.at(transition,(bands(a),bands(z)),1)
    assert transition.sum()==mask.sum()
    assert np.all(transition.sum(1)==np.bincount(bands(a),minlength=3))
    assert np.all(transition.sum(0)==np.bincount(bands(z),minlength=3))
    assert closer.sum()+away.sum()+same.sum()==mask.sum()
    assert (~good1).sum()==(~good0).sum()+(good0&~good1).sum()-(~good0&good1).sum()
    return {'n':int(mask.sum()),'toward_center_n':int(closer.sum()),'away_from_center_n':int(away.sum()),'within_epsilon_n':int(same.sum()),
      'mean_delta_cosine':float(delta.mean()),'median_delta_cosine':float(np.median(delta)),
      'q10_delta_cosine':float(np.quantile(delta,.1)),'q90_delta_cosine':float(np.quantile(delta,.9)),
      'mean_absolute_delta_cosine':float(np.abs(delta).mean()),
      'mean_delta_angle_to_center_deg':float((angles(z)-angles(a)).mean()),
      'mean_delta_distance_to_center':float((np.sqrt(np.maximum(0,2-2*z))-np.sqrt(np.maximum(0,2-2*a))).mean()),
      'toward_center_ge001_n':int((delta>=.01).sum()),'away_from_center_ge001_n':int((delta<=-.01).sum()),
      'toward_center_ge01_n':int((delta>=.1).sum()),'away_from_center_ge01_n':int((delta<=-.1).sum()),
      'crossed_into_center_ge09_n':int(((a<.9)&(z>=.9)).sum()),'crossed_out_center_lt09_n':int(((a>=.9)&(z<.9)).sum()),
      'correct_to_wrong_n':int((good0&~good1).sum()),'wrong_to_correct_n':int((~good0&good1).sum()),
      'before_wrong_n':int((~good0).sum()),'after_wrong_n':int((~good1).sum()),
      'region_transition_rows_before_columns_after':transition.tolist()}
r={'provenance':{'checkpoint':prior['provenance']['checkpoint'],'source_summary_sha256':verify['summary_sha256'],
 'scores_sha256':hashlib.sha256((root/'paired_scores.npz').read_bytes()).hexdigest(),
 'sample_n':2000,'per_class_n':1000,'score':'cosine to learned OC center, +1 closer to center; score is not a probability',
 'threshold':threshold,'direction_epsilon':eps,'identity_repeat_max_delta':prior['identity_repeat_max_delta'],
 'direction_epsilon_rule':'max(1e-4, 10 * maximum identity-repeat difference)',
 'region_order':['cosine >= 0.9','threshold <= cosine < 0.9','cosine < threshold'],
 'effect_size_cutoffs':[.01,.1],'effect_cutoffs_are_descriptive_not_validated_significance_thresholds':True},
 'original':{},'conditions':{},'short_to_long':{},'aligned_to_reversed':{},'original_to_condition':{},'tail_core':{},'by_source':{}}
for label,mask in [('bonafide',b),('spoof',~b)]:r['original'][label]=description(s[:,0],mask)
r['original']['mean_cosine_gap']=float(s[b,0].mean()-s[~b,0].mean())
for fam,(short,mid,long) in families.items():
    cond={'both_short':s[:,short],'both_medium':s[:,mid],'both_long':s[:,long],
          'B_long_S_short':np.where(b,s[:,long],s[:,short]),'B_short_S_long':np.where(b,s[:,short],s[:,long])}
    r['conditions'][fam]={};r['original_to_condition'][fam]={}
    for key,c in cond.items():
        r['conditions'][fam][key]={label:description(c,mask) for label,mask in [('bonafide',b),('spoof',~b)]}
        r['conditions'][fam][key]['mean_cosine_gap']=float(c[b].mean()-c[~b].mean())
        r['original_to_condition'][fam][key]={label:movement(s[:,0],c,mask) for label,mask in [('bonafide',b),('spoof',~b)]}
    r['short_to_long'][fam]={label:movement(s[:,short],s[:,long],mask) for label,mask in [('bonafide',b),('spoof',~b)]}
    r['aligned_to_reversed'][fam]={label:movement(cond['B_long_S_short'],cond['B_short_S_long'],mask) for label,mask in [('bonafide',b),('spoof',~b)]}
    r['tail_core'][fam]={label:movement(s[:,short],s[:,long],mask) for label,mask in [('original_tail',tail),('original_core',core)]}
    r['by_source'][fam]={}
    for source in np.unique(src):
        r['by_source'][fam][source]={label:movement(s[:,short],s[:,long],mask&(src==source)) for label,mask in [('bonafide',b),('spoof',~b)]}
    # Cross-check against already independently verified outcome counts.
    for label,mask in [('bonafide',b),('spoof',~b)]:
        reported=prior['dose'][fam]['all'][label]
        measured=r['short_to_long'][fam][label]
        assert abs(measured['mean_delta_cosine']+reported['mean_delta_long_minus_short'])<1e-6
        assert measured['correct_to_wrong_n']==reported['correct_to_wrong_n']
        assert measured['wrong_to_correct_n']==reported['wrong_to_correct_n']
    gap1=r['conditions'][fam]['B_long_S_short']['mean_cosine_gap']
    gap2=r['conditions'][fam]['B_short_S_long']['mean_cosine_gap']
    delta_b=r['aligned_to_reversed'][fam]['bonafide']['mean_delta_cosine']
    delta_s=r['aligned_to_reversed'][fam]['spoof']['mean_delta_cosine']
    assert abs((gap2-gap1)-(delta_b-delta_s))<1e-9
r['verification']={'all_counts_and_transitions_consistent':True,'agrees_with_prior_verified_summary':True,
 'mean_gap_change_identity_passed':True,'no_new_inference_or_training':True}
r['limitations']=['Direction is relative to the learned OC center. Away from that center does not identify a particular spoof cluster or PCA axis.',
 'Only cosine scores are retained for transformed inputs. Center angle and center distance are computable, but full 160D before-after displacement and transverse motion are not.',
 'Aligned versus reversed conditions use the same body, total duration and filler waveform sample multiset within each fixed-length family; the intervention changes front/back placement.',
 'Original-to-transformed comparisons also include common-body crop and filling. The paired short-to-long contrast is the cleaner placement comparison.',
 '0.9 is a training margin. 0.816285938 is a separately calibrated 19LA-dev decision threshold. Neither marks a newly fitted PCA contour.',
 'Original tail/core comparison is exploratory and affected by source mixture and cosine ceiling effects.',
 'Class mean cosine gap is a scalar score separation, not Euclidean distance between class centroids.']
(out/'tables.json').write_text(json.dumps(r,indent=2,allow_nan=False))

def table(headers,rows):
    return ['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,row))+' |' for row in rows]+['']
lines=['# 저에너지 배치에 따른 이동량 비교','',
'80epoch 실험의 기존 체크포인트, 같은 21DF bonafide 1,000개와 spoof 1,000개. 추가 추론 없이 기존 점수로 계산했다. 여기서는 spoof score의 부호를 뒤집은 **중심 cosine**을 사용한다. 양수 변화는 학습된 bonafide 중심에 가까워짐, 음수는 멀어짐이다. 특정 spoof 군집으로의 이동을 측정한 것은 아니다.','',
f'미세 수치 오차와 구분하기 위해 |Δcosine| ≤ {eps:.7f}는 방향 변화 없음으로 분류했다. Δcosine 0.01·0.1은 설명을 위한 크기 기준이지 통계적 유의성 기준은 아니다.','',
'## 1. 조건별 최종 위치와 오판정','',
'오판정은 별도 19LA dev에서 정한 cosine 임계값 0.816286으로 고정했다. 꼬리 기준 0.9는 학습 마진이므로 오판정과 다르다. 평균 cosine 차이는 한 축의 점수 간격이며 두 클래스 센트로이드 간 유클리드 거리가 아니다.','']
rows=[]
def position_row(label,e):
    B=e['bonafide'];S=e['spoof']
    return [label,f"{B['cosine_mean']:.4f}",f"{S['cosine_mean']:.4f}",f"{e['mean_cosine_gap']:.4f}",B['outside_center_region_lt09_n'],B['wrong_n'],S['wrong_n']]
rows.append(position_row('원본',r['original']))
for fam,title in [('zero20','0'),('ownquiet20','자체 파형')]:
    for cond,label in [('both_short','양쪽 짧게'),('both_long','양쪽 길게'),('B_long_S_short','B 길게 / S 짧게'),('B_short_S_long','B 짧게 / S 길게')]:
        rows.append(position_row(title+' · '+label,r['conditions'][fam][cond]))
lines+=table(['조건','B 평균 cosine','S 평균 cosine','평균 점수 간격','B cosine<0.9','B→S 오판정','S→B 오판정'],rows)
lines+=['## 2. B 길게·S 짧게 → B 짧게·S 길게: 같은 파일 이동','',
'B는 앞부분을 1.02→0.06초로 줄이고, S는 0.06→1.02초로 늘린 비교다. 각 행 n=1,000. 각도는 해당 임베딩과 고정된 학습 중심 사이의 각도이며, 양수 변화는 중심에서 멀어진 것이다.','']
rows=[]
for fam,title in [('zero20','0'),('ownquiet20','자체 파형')]:
    for label,word in [('bonafide','B 길게→짧게'),('spoof','S 짧게→길게')]:
        a=r['aligned_to_reversed'][fam][label]
        rows.append([title+' · '+word,a['toward_center_n'],a['away_from_center_n'],a['within_epsilon_n'],f"{a['mean_delta_cosine']:+.4f}",f"{a['median_delta_cosine']:+.4f}",f"{a['mean_delta_angle_to_center_deg']:+.2f}°",a['toward_center_ge01_n'],a['away_from_center_ge01_n']])
lines+=table(['조건·대상','중심 쪽','중심 반대','미세 변화','평균 Δcos','중앙값 Δcos','평균 중심 각도 변화','중심 쪽 ≥0.1','반대쪽 ≥0.1'],rows)
lines+=['## 3. 같은 비교에서 경계를 넘은 파일 수','']
rows=[]
for fam,title in [('zero20','0'),('ownquiet20','자체 파형')]:
    for label,word in [('bonafide','B'),('spoof','S')]:
        a=r['aligned_to_reversed'][fam][label]
        rows.append([title+' · '+word,a['crossed_into_center_ge09_n'],a['crossed_out_center_lt09_n'],a['correct_to_wrong_n'],a['wrong_to_correct_n'],f"{a['before_wrong_n']}→{a['after_wrong_n']}"])
lines+=table(['대상','cosine 0.9 안으로','cosine 0.9 밖으로','정답→오답','오답→정답','최종 오판정 수'],rows)
lines+=['## 4. 원래 꼬리와 중심부: 모두 짧게→길게 바꾼 반응','',
'원래 꼬리=원본 bonafide cosine<0.9인 163개, 원래 중심부=나머지 837개. 중심부는 이미 cosine이 높아 증가 여지가 적고 출처 구성도 다르므로 변화량 배율을 원인의 기여율로 해석하지 않는다.','']
rows=[]
for fam,title in [('zero20','0'),('ownquiet20','자체 파형')]:
    for label,word in [('original_tail','꼬리'),('original_core','중심부')]:
        a=r['tail_core'][fam][label]
        rows.append([title+' · '+word,a['n'],a['toward_center_n'],a['away_from_center_n'],f"{a['mean_delta_cosine']:+.4f}",f"{a['median_delta_cosine']:+.4f}",a['toward_center_ge01_n'],a['crossed_into_center_ge09_n']])
lines+=table(['대상','N','중심 쪽','반대쪽','평균 Δcos','중앙값 Δcos','중심 쪽 ≥0.1','0.9 안으로 진입'],rows)
lines+=['## 5. 출처별 짧게→길게 반응','']
rows=[]
for fam,title in [('zero20','0'),('ownquiet20','자체 파형')]:
    for source,g in r['by_source'][fam].items():
        for label,word in [('bonafide','B'),('spoof','S')]:
            a=g[label]; rows.append([title,source,word,a['n'],a['toward_center_n'],a['away_from_center_n'],f"{a['mean_delta_cosine']:+.4f}",f"{a['median_delta_cosine']:+.4f}",a['toward_center_ge01_n']])
lines+=table(['여백','출처','정답','N','중심 쪽','반대쪽','평균 Δcos','중앙값 Δcos','중심 쪽 ≥0.1'],rows)
lines+=['## 6. 원본 대비 변화: 편집 효과가 섞인 보조 비교','',
'이 비교는 앞뒤 배치 이외에도 공통 본문 선택·절단과 여백 생성의 효과를 포함한다. 원인 해석은 2번의 동일 본문·동일 총 여백 비교를 우선한다.','']
rows=[]
for fam,title in [('zero20','0'),('ownquiet20','자체 파형')]:
    for cond,word in [('B_long_S_short','B 길게 / S 짧게'),('B_short_S_long','B 짧게 / S 길게')]:
        for label,lab in [('bonafide','B'),('spoof','S')]:
            a=r['original_to_condition'][fam][cond][label]
            rows.append([title,word,lab,a['toward_center_n'],a['away_from_center_n'],f"{a['mean_delta_cosine']:+.4f}",f"{a['median_delta_cosine']:+.4f}",a['correct_to_wrong_n'],a['wrong_to_correct_n']])
lines+=table(['여백','조건','정답','중심 쪽','반대쪽','평균 Δcos','중앙값 Δcos','정답→오답','오답→정답'],rows)
lines+=['## 7. 영역 간 이동 행렬: 자체 파형, 유리한 배치→반대 배치','',
'행은 이전 영역, 열은 이후 영역. ①중심부=cosine≥0.9, ②중심 밖이나 B 판정=0.816286≤cosine<0.9, ③S 판정=cosine<0.816286. 각 행렬은 1,000개를 모두 포함한다.','']
for label,word in [('bonafide','실제 bonafide'),('spoof','실제 spoof')]:
    lines += [word,''];m=r['aligned_to_reversed']['ownquiet20'][label]['region_transition_rows_before_columns_after']
    lines += table(['이전→이후','①중심부','②중심 밖·B 판정','③S 판정'],[[name]+row for name,row in zip(['①중심부','②중심 밖·B 판정','③S 판정'],m)])
lines+=['## 범위와 검증','',
'위치는 학습된 OC 중심을 기준으로 한 160차원 공간의 cosine/각도/중심까지 거리다. 변형본의 전체 임베딩은 저장되지 않아 파일별 160차원 벡터 이동거리, PCA 좌표 이동, 특정 spoof 군집으로 향했는지는 이 계산으로 알 수 없다.','',
'모든 방향별 개수 합계, 영역 전이 행렬의 행·열 합계, 정답↔오답 변화, 기존 검증 결과와의 일치, 클래스 평균 점수 간격 변화 항등식을 확인했다. 추가 학습이나 추론 없이 컨테이너 내부에서 계산했다.','']
(out/'comparison_ko.md').write_text('\n'.join(lines),encoding='utf8')
print(json.dumps({k:r[k] for k in ['original','aligned_to_reversed','tail_core','verification']},indent=2),flush=True)
