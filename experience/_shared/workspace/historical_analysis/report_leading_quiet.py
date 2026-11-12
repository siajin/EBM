import os,json
from pathlib import Path
os.environ['MPLCONFIGDIR']='/home/mentee/.cache/matplotlib'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
assert Path('/.dockerenv').exists()
root=Path('/results/analysis/EBM80_21DF_leading_quiet_paired')
j=json.loads((root/'summary.json').read_text());v=json.loads((root/'verification.json').read_text())
assert v['all_checks_passed']
families=['zero20','ownquiet20','variable20','zero40']
display=['Zeros; fixed length','Own quiet; fixed length','Zeros; variable length','Zeros; strict onset; fixed length']
fig,axes=plt.subplots(1,2,figsize=(14,5.2),layout='constrained')
yy=np.arange(4)
for label,color,shift in [('bonafide','#287DA8',-.10),('spoof','#D07A2B',.10)]:
    entries=[j['dose'][f]['all'][label] for f in families]
    means=np.array([e['mean_delta_long_minus_short'] for e in entries]);ci=np.array([e['speaker_cluster_95ci'] for e in entries])
    axes[0].hlines(yy+shift,ci[:,0],ci[:,1],colors=color,lw=2)
    axes[0].scatter(means,yy+shift,color=color,label=label,zorder=3)
axes[0].axvline(0,c='#777',lw=1);axes[0].set_yticks(yy,display);axes[0].invert_yaxis()
axes[0].set(title='0.06s to 1.02s leading quiet: same-file change',xlabel='Spoof-score change (negative = more bonafide-like)');axes[0].legend(loc='lower left');axes[0].grid(axis='x',alpha=.15)
xx=np.arange(2)
for key,label,offset,color in [('aligned_Blong_Sshort','B long / S short',-.18,'#3D8396'),('reversed_Bshort_Slong','B short / S long',.18,'#C17E45')]:
    values=[v['assignment_experiment'][f]['all'][key]['eer_pct'] for f in families[:2]]
    bars=axes[1].bar(xx+offset,values,width=.34,color=color,label=label)
    axes[1].bar_label(bars,labels=[f'{z:.1f}%' for z in values],padding=4)
axes[1].set_xticks(xx,['Zeros; fixed length','Own quiet; fixed length']);axes[1].set(title='Assign the same short/long views by label',ylabel='Sample EER (%)',ylim=(0,8));axes[1].legend(loc='upper left');axes[1].grid(axis='y',alpha=.15)
fig.suptitle('Frozen epoch80 checkpoint | 1,000 bonafide + 1,000 spoof | no retraining\nFixed-length tests keep body samples, total quiet amount and waveform energy identical across doses',fontsize=12)
fig.savefig(root/'leading_quiet_summary.png',dpi=180);fig.savefig(root/'leading_quiet_summary.pdf');plt.close(fig)
lines=['# 앞부분 저에너지 구간 개입 실험','',
'기존 80epoch 실험의 BestCLS_80_21DF.pt를 고정하고 21DF bona fide 1,000개, spoof 1,000개에 같은 변형을 적용했다. 컨테이너 내 두 GPU에서 처리했으며 학습이나 원본 오디오 수정은 하지 않았다. 추론과 검증 종료코드는 모두 0이다.','',
'## 원래 수치의 의미','',
'19LA train의 1.00초/0.48초, 21DF ASVspoof 출처의 1.02초/0.06초는 우리가 codec·RawBoost를 추가하기 전, 제공된 데이터의 앞부분 저에너지 길이 중앙값이다. 엄밀히는 기존 평가 전처리(앞 64,600샘플, 짧으면 반복) 후 측정했다. 20ms 프레임 RMS가 해당 입력의 최대 프레임 RMS보다 20dB 낮은 기준이며, 사람이 판정한 무음이 아니다. 데이터 제작·편집·합성·기존 코덱 처리의 영향까지 없다는 뜻은 아니다.','',
'앞부분 길이만으로 bonafide를 높은 순위에 놓는 AUC: 19LA train 0.694, 19LA dev 0.726, 21DF ASVspoof 0.837, VCC2018 0.507, VCC2020 0.645. 클래스와의 연관성은 있으나 출처에 따라 다르다.','',
'## 입력 통제','',
'원본 앞부분 저에너지 구간을 찾은 뒤 공통 음성 본문을 최대 3.0175초 선택하고 짧으면 0으로 채웠다. 변형끼리는 이 본문이 비트 단위로 같다. 원본에 비해 뒤쪽 내용이 생략된 파일은 1,692개이므로 원본 대 변형의 차이만을 원인 효과로 해석하지 않는다. 주 비교는 공통 본문을 공유하는 변형끼리다.','',
'고정 길이 조건: [앞 0.06/0.48/1.02초] + [동일 본문 3.0175초] + [뒤 0.96/0.54/0초]. 전체 4.0375초, 총 저에너지 filler 1.02초, 전체 파형 샘플 집합 및 에너지가 같다. 앞뒤 배치를 바꾸는 실험이지 앞길이만 독립적으로 바꾸는 실험은 아니다.','',
'Filler는 ① 완전한 0, ② 해당 파일의 기존 앞부분 저에너지 파형을 반복한 것으로 나눴다. 원래 앞구간이 없는 22개는 0을 사용했다. 추가로 뒤 filler 없이 앞부분을 늘리는 가변 길이 조건과 -40dB 기준으로 더 적게 잘라낸 대조 조건을 계산했다.','',
'## 결과','',
'점수 = -cosine(임베딩, 학습된 OC 중심). 음수 변화는 bonafide 방향이며 확률이나 퍼센트 변화가 아니다. 아래는 앞부분 0.06→1.02초의 같은 파일 내 평균 점수 변화, 괄호는 화자 단위 bootstrap 95% 구간이다.','',
'| 조건 | Bona fide | Spoof |','|---|---:|---:|']
for f,title in zip(families,display):
    items=[]
    for label in ['bonafide','spoof']:
        e=j['dose'][f]['all'][label];lo,hi=e['speaker_cluster_95ci'];items.append(f"{e['mean_delta_long_minus_short']:+.4f} ({lo:+.4f}, {hi:+.4f})")
    lines.append(f'| {title} | {items[0]} | {items[1]} |')
lines+=['','기존 저에너지 파형을 사용한 고정 길이 조건에서 bona fide 96.0%, spoof 83.3%가 bonafide 방향으로 움직였다. 앞구간이 실제로 목표 길이를 충족하고 원래 앞구간이 존재한 표본만 제한해도 평균 변화는 bona fide -0.1521(n=994), spoof -0.0706(n=980)으로 유지됐다.','',
'같은 변형본들을 클래스에 따라 배치한 진단용 비교:','',
'| 고정 길이 filler | B 길게·S 짧게 EER | B 짧게·S 길게 EER |','|---|---:|---:|']
for f,title in zip(families[:2],['0','파일 자체 저에너지 파형']):
    z=v['assignment_experiment'][f]['all'];lines.append(f"| {title} | {z['aligned_Blong_Sshort']['eer_pct']:.1f}% | {z['reversed_Bshort_Slong']['eer_pct']:.1f}% |")
lines+=['','이 표는 같은 본문과 총 filler를 사용하는 가공 조건끼리의 비교다. 원본 EER 2.0%와 직접적인 성능 개선 실험으로 비교하면 안 된다. 전체 공식 평가셋 수치가 아니며, 클래스 정답에 따른 배치는 원인을 진단하기 위한 인위적 조건이다.','',
'19LA dev 1,000+1,000개에서 독립적으로 정한 cosine 임계값 0.816286을 모든 조건에 고정했다. 이 dev 표본은 EER 0%여서 완전 분리 구간의 중간점을 선택했으며, 임계값에는 운용 선택의 여지가 있다. 원본 21DF에서는 bona fide 거절률 11.5%, spoof 통과율 0.4%였다. 임계값의 도메인 이동을 고려해야 하며, 이 오판정률을 EER와 혼동하면 안 된다.','',
'파일 자체 저에너지 파형을 사용하는 고정 길이 조건에서 0.06→1.02초로 바꾸면 bona fide 오판정은 214→70개(145개 회복, 1개 추가 오판정), spoof의 bonafide 오판정은 5→22개(17개 추가)였다. 이 개수는 위 고정 임계값에서의 결과다.','',
'## 해석','',
'학습 데이터에서 길이와 클래스의 연관성이 있고, 같은 파일의 본문·총 길이·총 에너지를 통제해도 앞뒤 저에너지 배치가 점수를 움직인다. 따라서 이 모델이 저에너지 구간 구조를 판별에 이용한다는 근거가 강해졌다. 특히 훈련 데이터의 전형적 배치와 반대로 구성하면 EER가 크게 증가했다.','',
'다만 단순히 앞 무음을 추가하면 항상 bonafide가 된다는 결론은 틀리다. 전체 길이가 함께 늘어나는 조건에서는 평균 효과가 작고 spoof의 평균 변화 신뢰구간은 0을 포함했다. 그 조건에서는 spoof 88.2%가 오히려 작은 spoof 방향 변화를 보였고 소수의 큰 반대 변화가 평균을 끌어내렸다. 앞/뒤 배치, 입력 길이, 내용의 위치가 상호작용한다.','',
'출처별로도 크기가 다르다. 자체 저에너지 파형의 고정 길이 조건에서 VCC2018 bona fide의 평균 변화 -0.3988, ASVspoof spoof의 변화 -0.1865였다. ASVspoof bona fide는 -0.0339에 그쳤다. 하나의 전체 평균만으로 모든 출처의 원인을 설명하면 안 된다.','',
'관찰과 개입이 함께 있어 지름길 의존 가설을 지지하지만, 특정 학습 단계에서 획득했다는 역사적 원인이나 저에너지 질감의 합성 흔적이 모두 부당한 단서라는 사실까지 입증한 것은 아니다. 실제 재학습으로 양 클래스의 앞뒤 저에너지 길이 분포를 같게 만들고, 원본 및 길이 반전 평가에 대한 일반화가 좋아지는지 확인하는 것이 다음 대조 실험이다. 센트로이드를 무음 길이로 바로 나누면 이 편향을 그대로 수용할 가능성도 있다.','',
'## 검증과 파일','',
'13개 조건 × 4가지 원본/변형 조합의 EER·AUC·오판정률 52개를 별도 코드로 검산했다. AUC는 rank-sum, 균형 표본 EER는 정렬된 점수의 중앙 임계값 오류 개수로 교차 확인했다. 체크포인트 해시 동일, 입력 동일성·에너지 보존 검증 통과. 무변형 반복의 최대 점수 차이는 1.295e-5였다. 원본과 체크포인트를 변경하지 않았다.','',
'- summary.json: 모든 조건/출처별 수치와 가정','- verification.json: 검산 및 길이-클래스 배치 실험','- paired_scores.npz: 서버 내부 파일별 점수(외부 반출하지 않음)','- leading_quiet_summary.png / .pdf: 핵심 결과','- leading_quiet_effect.png / .pdf: 세 길이에 따른 반응','',
'관련 연구: [Müller et al., 2021](https://arxiv.org/abs/2106.12914)은 ASVspoof의 무음 길이와 클래스 편향을 분석했다. [Rubio et al., 2026](https://arxiv.org/abs/2607.03150)은 분포 연관성과 통제 개입을 함께 활용하는 진단 틀을 제시했다. 이번 수치는 해당 논문의 결과를 인용한 것이 아니라 우리 체크포인트에서 직접 측정한 결과다.','']
(root/'interpretation_ko.md').write_text('\n'.join(lines),encoding='utf8')
print('Report and compact figure saved.')
