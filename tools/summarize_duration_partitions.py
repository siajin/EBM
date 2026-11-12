from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/duration_partitions_20261109'
NAMES=['2019 LA · train','2019 LA · dev','2019 LA · eval','2021 LA · eval','2021 DF · eval']


def verify_and_select():
    compact=pd.read_csv(OUT/'compact/summary.csv')
    full=pd.read_csv(OUT/'full/summary.csv')
    assert len(compact)==len(full)==5
    selected=[];checks={};allhist=[]
    dest=OUT/'recommended'
    dest.mkdir(exist_ok=True)
    for _,base in compact.iterrows():
        key=base.dataset
        candidates=[('compact',base),('full',full[full.dataset==key].iloc[0])]
        # A transparent presentation choice: retain a class-combined mean piece
        # length >=2s when available, then choose the highest attained overlap.
        # Both raw alternatives remain available for inspection.
        long=[pair for pair in candidates if pair[1].mean_piece_seconds>=2]
        folder,row=max(long or candidates,key=lambda pair:pair[1].overlap_pct)
        plan=pd.read_parquet(OUT/folder/f'{key}_partition_plan.parquet')
        original=pd.read_parquet(OUT/f'{key}_global_optimum_half_second_plan.parquet')
        joined=plan.merge(original[['utterance_id','y','original_samples']],on='utterance_id',
            suffixes=('_chosen','_source'),how='outer',validate='one_to_one',indicator=True)
        assert (joined['_merge']=='both').all()
        assert (joined.y_chosen==joined.y_source).all()
        assert (joined.original_samples_chosen==joined.original_samples_source).all()
        assert (plan.pieces>=1).all() and (plan.piece_samples>0).all()
        assert (plan.pieces*plan.piece_samples+plan.discarded_samples==plan.original_samples).all()
        assert ((plan.discarded_samples>=0)&(plan.discarded_samples<8000)).all()
        small=plan.original_samples<8000
        assert (plan.loc[~small,'piece_samples']>=8000).all()
        assert (plan.loc[small,'pieces']==1).all() and (plan.loc[small,'discarded_samples']==0).all()
        assert (plan.loc[small,'piece_samples']==plan.loc[small,'original_samples']).all()
        hs=plan.groupby(['y','piece_samples']).pieces.sum().unstack(0,fill_value=0)
        assert 1 in hs and 0 in hs
        exact=float(np.minimum(hs[1]/hs[1].sum(),hs[0]/hs[0].sum()).sum())
        assert abs(exact*100-row.overlap_pct)<1e-8
        chosen=row.to_dict();chosen['selected_from']=folder
        chosen['exact_duration_overlap_pct']=exact*100
        selected.append(chosen)
        plan.to_parquet(dest/f'{key}_partition_plan.parquet',index=False)
        hist=pd.read_csv(OUT/folder/'histograms.csv')
        allhist.append(hist[hist.dataset==key])
        checks[key]={'every_original_once':True,'labels_and_lengths_match_sources':True,
            'no_source_reuse_or_overlap':True,'all_tail_losses_strictly_below_0_5s':True,
            'all_new_segments_at_least_0_5s':True,'subminimum_originals_unchanged':True,
            'overlap_recomputed_from_exact_output_sample_lengths_pct':exact*100,
            'manifest_originals':len(plan),'manifest_pieces':int(plan.pieces.sum())}
    table=pd.DataFrame(selected)
    table.to_csv(dest/'summary.csv',index=False)
    pd.concat(allhist,ignore_index=True).to_csv(dest/'histograms.csv',index=False)
    (dest/'validation.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    return table


def plot(table):
    h=pd.read_csv(OUT/'recommended/histograms.csv')
    plt.rcParams.update({'font.family':'Malgun Gothic','font.size':12,'axes.unicode_minus':False,
                         'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    colors=['#2376AB','#D97813']
    for end in [2,10]:
        fig,axes=plt.subplots(2,3,figsize=(18,10),sharex=True,sharey=True)
        ymax=0
        for ax,(_,r),name in zip(axes.flat,table.iterrows(),NAMES):
            for label,color,ls in [(1,colors[0],'-'),(0,colors[1],'--')]:
                d=h[(h.dataset==r.dataset)&(h.stage=='after')&(h.y==label)]
                values=np.zeros(600)
                ix=np.rint(d.bin_left_s.to_numpy()/.05).astype(int)
                values[ix]=d.percent.to_numpy()
                ymax=max(ymax,values[:int(end/.05)].max())
                ax.stairs(values,np.arange(601)*.05,color=color,lw=2,ls=ls,baseline=None)
            ax.set_title(name,loc='left',fontweight='bold',pad=14)
            ax.text(.97,.96,f'겹침 {r.overlap_pct:.3f}%\n조각 수 {r.pieces_per_original:.2f}배',
                    ha='right',va='top',transform=ax.transAxes,fontsize=11,
                    bbox={'facecolor':'white','edgecolor':'none','alpha':.9})
            ax.set_xlim(0,end);ax.tick_params(labelbottom=True)
            ax.set_xlabel('생성된 조각 길이 (초)');ax.set_ylabel('클래스 전체 조각 중 비율 / bin (%)')
            ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        axes.flat[5].set_visible(False)
        for ax in axes.flat[:5]:ax.set_ylim(0,np.ceil(ymax*1.15*2)/2)
        fig.suptitle(f'원본당 꼬리 손실 <0.5초 · 최소 조각 0.5초 · 분할 후 (0–{end}초)',
                     x=.065,y=.975,ha='left',fontsize=22)
        fig.legend(handles=[Line2D([0],[0],color=colors[0],lw=2,label='Bona fide'),
                            Line2D([0],[0],color=colors[1],ls='--',lw=2,label='Spoof')],
                   loc='upper center',bbox_to_anchor=(.5,.92),ncol=2,frameon=False)
        fig.text(.065,.025,'원본마다 분할 방식 1개 · 조각 중복 없음 · 속도 변형 없음 · 0.05초 bin · 겹침률은 전체 길이 범위',fontsize=11,color='#555555')
        fig.subplots_adjust(left=.065,right=.985,bottom=.12,top=.82,wspace=.27,hspace=.52)
        fig.savefig(OUT/f'recommended_0_{end}s.png',dpi=160,facecolor='white')
        fig.savefig(OUT/f'recommended_0_{end}s.svg',facecolor='white')
        plt.close(fig)


def report(table):
    global_=pd.read_csv(OUT/'global_optimum.csv')
    lines=['# 음성 분할로 길이 분포 맞추기 · 2026-11-09','',
        '실제 원본 길이 metadata 803,565개를 사용했다. 원음·서버·GPU·학습·기존 실험은 변경하지 않았다.',
        '', '## 조건','',
        '- 원본마다 한 가지 분할 방식만 사용한다. 대안 분할을 동시에 만들어 중복 노출하지 않는다.',
        '- 동일 원본의 조각들은 인접한 구간이며 겹치지 않는다. 조각은 이어 붙이지 않는다.',
        '- 새로운 조각은 최소 0.5초이고, 원본당 버리는 꼬리는 엄격히 0.5초 미만이다.',
        '- 원래 0.5초 미만인 파일은 원본 그대로 한 번 보존한다. 19LA eval spoof 1개, 21DF eval spoof 9개가 해당한다.',
        '- 속도 조정·패딩·원본 제외가 없다. 양쪽 클래스 모두 분할할 수 있다.',
        '- 분포는 각 클래스의 생성된 조각 수를 분모로 정규화한다. 원본별 생성 조각 수가 달라지므로, 이전의 원본 균등 확률 최적화와 노출 가중치는 다르다.',
        '', '## 알고리즘','',
        '1. 16 kHz 샘플 수로 계산한다. 0.05초 간격의 조각 길이 L마다 k개로 나누었을 때 `0 <= D-kL < 8000 samples`인 후보를 만든다.',
        '2. 예를 들어 12초는 4초×3개, 3초×4개, 6초×2개가 모두 손실 0인 유효 후보다.',
        '3. 비슷한 원본 길이를 묶되 묶음의 모든 원본에서 유효한 후보만 남긴다. 각 원본은 후보 하나를 반드시 선택한다.',
        '4. 클래스별 총 조각 수가 변하는 정규화 문제를 고정 비율별 선형계획으로 풀고, 후보 비율을 탐색한다.',
        '5. 원본당 정수 선택으로 바꾸고 혼합정수 최적화로 보완한다. 최종 실제 조각 개수와 정확한 길이값으로 겹침률을 다시 계산한다.',
        '6. 조각 좌표는 `[jL,(j+1)L)`이다. 원본 전체와 UID를 대조하여 보존·길이·중복·손실 조건을 검산한다.',
        '', '실용 탐색은 동일 길이 분할·0.05초 격자·유한 비율 탐색이며, 일부 정수 최적화는 제한 시간까지의 최선 해다. 실용안의 전역 최적성을 주장하지 않는다. 전체 최적 상한은 아래의 별도 증명과 구성으로 정확히 구했다.',
        '', '## 긴 조각을 유지하는 구체적 분할안','',
        '조각 수를 줄이는 compact 탐색과 일치율 우선 full 탐색을 모두 보존했다. 평균 조각 길이 2초 이상인 후보 중 겹침률이 더 높은 것을 표에 선택했다. 2초는 이 표의 선택 기준이며 사용자가 정한 최소 길이 0.5초를 바꾸지 않는다.',
        '', '| 데이터 | 겹침률 | 조각 수 / 원본 수 | 평균 조각 길이 | 버린 전체 음성 비율 | 최대 꼬리 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in table.itertuples():
        lines.append(f'| {r.dataset} | {r.overlap_pct:.4f}% | {r.pieces_per_original:.3f}배 | {r.mean_piece_seconds:.3f}초 | {r.discarded_audio_pct:.3f}% | {r.max_discard_per_original_s:.7f}초 |')
    lines+=['','겹침은 동일 길이의 클래스별 비율 중 작은 값을 모두 더한 값이다. 새 조각들은 공통 길이 격자에 있으므로 최종 표의 겹침은 정확한 샘플 길이로 재계산해도 같다. 원래 짧은 예외 파일은 그대로 계산한다.',
        '', '## 전체 수학적 최대','',
        '모든 0.5초 이상 원본에서 `k=floor(D/0.5초)`를 선택해 0.5초 조각을 만들면 꼬리는 항상 0.5초 미만이다. 이때 0.5초 미만의 예외 원본을 제외한 모든 조각 길이가 같다.',
        '', '이 자료의 bona fide에는 0.5초 미만 원본이 없으므로 짧은 spoof 원본의 비율은 겹칠 수 없다. 생성 조각당 최소 0.5초 조건에서 spoof 전체 조각 수는 `sum(max(1,floor(D/0.5초)))`를 넘을 수 없다. 따라서 `1 - 짧은 spoof 원본 수 / 최대 spoof 조각 수`가 전역 상한이며, 위 0.5초 분할이 그대로 달성한다. 동일 길이 분할·격자를 벗어나 혼합 길이 분할을 허용해도 이 상한은 변하지 않는다.',
        '', '| 데이터 | 전역 최대 겹침률 | 전부 0.5초 분할 시 조각 수 배율 | 버린 음성 비율 |',
        '|---|---:|---:|---:|']
    for r in global_.itertuples():
        lines.append(f'| {r.dataset} | {r.global_maximum_overlap_pct:.9f}% | {r.baseline_piece_multiplier:.3f}배 | {r.baseline_audio_discard_pct:.3f}% |')
    lines+=['','100% 일치만 목표로 하면 길이 분포가 대부분 0.5초로 모일 수 있다. 긴 문맥을 유지하는 안과 이론 최대를 함께 비교해야 한다.',
        '', '원래 0.5초 미만인 10개 파일을 통째로 버리는 것도 허용하면 각 파일의 손실이 0.5초 미만이므로 사용자의 시간 손실 조건 자체는 만족한다. 이 해석에서는 5개 데이터셋 모두 100%가 가능하다. 위 주 결과와 실용 분할 계획은 질문 때 안내한 대로 이 10개까지 보존한 경우다. 파일 삭제나 실제 음성 처리는 하지 않았다.',
        '', '## 파일','',
        '- `recommended/*_partition_plan.parquet`: 실제 원본별 분할 개수, 조각 샘플 길이, 꼬리 손실.',
        '- `recommended/validation.json`: 원본 전체 1:1 대조, 손실·길이·중복 검산 및 정확한 길이 기반 겹침률.',
        '- `*_global_optimum_half_second_plan.parquet`: 전역 최대를 달성하는 0.5초 분할 계획.',
        '- `compact/`, `full/`: 대안별 결과, 실제 정수 해와 최적화 종료 상태.',
        '- `global_optimum.csv`, `independent_validation.json`: 전역 상한과 독립 검산.',
        '', '이 결과는 데이터셋별 정답 클래스를 사용한 길이 분포 분석이다. 실제 오디오 조각 생성이나 모델 성능 평가는 수행하지 않았다.']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    table=verify_and_select();plot(table);report(table)
    print(table[['dataset','overlap_pct','pieces_per_original','mean_piece_seconds','discarded_audio_pct','selected_from']].to_string(index=False))
