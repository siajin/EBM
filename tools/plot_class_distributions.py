"""Dataset/class distributions from immutable cached measurements, CPU only.

Run from IRLab: python EBM/tools/plot_class_distributions.py
Optional prior acoustic aggregates: collect_prior_acoustic_distributions.py.
No model inference, training, audio transfer, or changes to experiment sources.
"""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

PROJECT = Path(__file__).resolve().parents[2]
CACHE = PROJECT / 'artifacts/geometry_cache'
OUT = PROJECT / 'artifacts/class_distributions_20261108'
FIG = OUT / 'figures'
sys.path.insert(0, str(PROJECT / 'artifacts/geometry_experiments'))
from geometry_cache import resolve_run

SETS = ['19LA_train', '19LA_dev', '19LA_eval', '21LA_eval', '21DF_eval']
TITLE = dict(zip(SETS, ['2019 LA · train', '2019 LA · dev', '2019 LA · eval', '2021 LA · eval', '2021 DF · eval']))
EXPECTED = {'19LA_train': (25380, 2580), '19LA_dev': (24844, 2548), '19LA_eval': (71237, 7355), '21LA_eval': (148176, 14816), '21DF_eval': (533928, 14869)}
COLORS = {1: '#2473aa', 0: '#d47720'}
LABELS = {1: 'Bona fide', 0: 'Spoof'}
LS = {1: '-', 0: '--'}
GALLERY = []
CHECKS = []
SOURCES = []
HISTOGRAMS = []
NUMERIC = []

plt.rcParams.update({'font.family': ['DejaVu Sans', 'Noto Sans KR'], 'font.size': 10, 'axes.unicode_minus': False,
                     'mathtext.fontset': 'dejavusans',
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.titleweight': 'bold',
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'svg.fonttype': 'none'})

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def source(path, expected=None):
    sha = digest(path)
    if expected is not None:
        assert sha == expected, f'Changed cache: {path}'
    SOURCES.append({'path': str(Path(path).relative_to(PROJECT)), 'sha256': sha})

def read_alias(alias):
    folder, entry, split = resolve_run(alias)
    path = folder / 'extract/scores.parquet'
    meta = next(m for m in entry['files'] if m['path'] == 'extract/scores.parquet')
    source(path, meta['sha256'])
    f = pd.read_parquet(path)
    f = f[f.split == split].copy()
    assert f.utterance_id.is_unique
    f['cache_alias'] = alias
    f['source_corpus'] = f.get('source_corpus', 'asvspoof')
    f['codec'] = f.get('codec', 'clean')
    f['transmission'] = f.get('transmission', 'clean')
    return f

def load():
    aliases = {a: read_alias(a) for a in ['T', 'V', 'Q', '19LA_eval', '21LA_eval', 'DF_pilot', 'DF_remaining']}
    frames = {'19LA_train': aliases['T'], '19LA_dev': pd.concat([aliases['V'], aliases['Q']], ignore_index=True),
              '19LA_eval': aliases['19LA_eval'], '21LA_eval': aliases['21LA_eval'],
              '21DF_eval': pd.concat([aliases['DF_pilot'], aliases['DF_remaining']], ignore_index=True)}
    paired_root = CACHE / 'local_analysis/B_final_analysis_20261108_v1'
    manifest = json.loads((paired_root / 'download_manifest.json').read_text())
    for dataset, phase in [('19LA_dev', 'dev'), ('19LA_eval', '19LA'), ('21LA_eval', '21LA'), ('21DF_eval', '21DF')]:
        path = paired_root / f'paired_scores_{phase}.parquet'
        source(path, manifest['files'][path.name]['sha256'])
        p = pd.read_parquet(path)
        f = frames[dataset]
        assert len(p) == len(f) and p.uid.is_unique
        p = p.set_index('uid').loc[f.utterance_id]
        assert np.array_equal(p.label, f.y)
        assert np.array_equal(p.original_samples, np.rint(f.duration * 16000).astype('int64'))
        f['score_B'] = p.score_B.to_numpy()
        f['score_legacy240'] = p.score_legacy240.to_numpy()
    for dataset, f in frames.items():
        n, bona = EXPECTED[dataset]
        assert len(f) == n and f.y.sum() == bona and f.utterance_id.is_unique
        assert set(f.y) == {0, 1} and np.isfinite(f.duration).all() and f.duration.gt(0).all()
        assert np.isfinite(f[['cosine_s', 'raw_norm', 'tangent_norm']]).all().all()
        f['dataset_key'] = dataset
        f['length_band'] = pd.cut(f.duration, [0, 2, 3.0275, 4.0375, np.inf], right=False,
                                  labels=['<2 s', '2–3.0275 s', '3.0275–4.0375 s', '≥4.0375 s'])
        CHECKS.append({'dataset': dataset, 'N': n, 'bonafide': bona, 'spoof': n-bona,
                       'unique_ids': True, 'class_counts_match': True, 'paired_uid_label_duration_match': dataset != '19LA_train'})
    return frames

def legend(fig, y=.91):
    fig.legend(handles=[Line2D([], [], color=COLORS[v], ls=LS[v], lw=2, label=LABELS[v]) for v in [1, 0]],
               loc='upper center', bbox_to_anchor=(.5, y), ncol=2, frameon=False)

def grid(title, note, datasets=SETS):
    rows = 2 if len(datasets) > 3 else 1
    fig, axes = plt.subplots(rows, 3 if len(datasets) > 2 else 2, figsize=(15, 8.5 if rows == 2 else 5.1))
    axes = np.asarray(axes).ravel()
    fig.suptitle(title, x=.065, y=.985, ha='left', fontsize=17, weight='bold')
    fig.text(.065, .025, note, fontsize=10, color='#46525e')
    for ax, dataset in zip(axes, datasets):
        ax.set_title(TITLE[dataset], loc='left', pad=12)
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
    for ax in axes[len(datasets):]:
        ax.set_visible(False)
    fig.subplots_adjust(left=.065, right=.98, top=.83, bottom=.15, hspace=.58, wspace=.28)
    legend(fig, .93)
    return fig, axes[:len(datasets)]

def save(fig, name, title, scope):
    from matplotlib.text import Text
    for text in fig.findobj(match=Text):
        text.set_text(text.get_text().replace('\u2212', '-'))
    for ext in ['png', 'svg']:
        fig.savefig(FIG / f'{name}.{ext}', dpi=180)
    plt.close(fig)
    GALLERY.append({'name': name, 'title': title, 'scope': scope})
    print(f'FIGURE {name}', flush=True)

def record_hist(dataset, label, metric, values, edges):
    counts, bins = np.histogram(values, edges)
    assert counts.sum() == len(values), (metric, counts.sum(), len(values))
    HISTOGRAMS.append({'dataset': dataset, 'label': label, 'metric': metric, 'N': len(values),
                       'edges': bins.tolist(), 'counts': counts.tolist()})
    return counts, bins

def hist(ax, dataset, label, metric, values, edges, scale=1):
    c, b = record_hist(dataset, label, metric, values, edges)
    ax.stairs(100 * c / len(values), b * scale, color=COLORS[label], ls=LS[label], lw=1.6)
    NUMERIC.append({'dataset': dataset, 'class': LABELS[label], 'metric': metric, 'N': len(values),
                    'mean': float(np.mean(values)), 'median': float(np.median(values)),
                    'p05': float(np.quantile(values, .05)), 'p95': float(np.quantile(values, .95)),
                    'min': float(np.min(values)), 'max': float(np.max(values))})

def counts_plot(frames):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    xs = np.arange(5)
    for y in [1, 0]:
        n = np.array([(frames[d].y == y).sum() for d in SETS])
        bars = axes[0].bar(xs + (-.2 if y else .2), n, width=.38, color=COLORS[y], label=LABELS[y])
        axes[0].bar_label(bars, labels=[f'{v:,}' for v in n], padding=4, fontsize=9)
        bottom = np.zeros(5) if y else np.array([100*frames[d].y.mean() for d in SETS])
        pct = 100*n/np.array([len(frames[d]) for d in SETS])
        axes[1].bar(xs, pct, bottom=bottom, width=.65, color=COLORS[y])
        for x, v, b in zip(xs, pct, bottom):
            axes[1].text(x, b+v/2, f'{v:.2f}%', ha='center', va='center', fontsize=10,
                         color='white' if v > 15 else '#162733')
    axes[0].set_yscale('log'); axes[0].set_ylim(1000, 1e6)
    axes[0].set_ylabel('발화 수 · 로그축'); axes[0].set_title('클래스별 실제 표본 수')
    axes[1].set_ylim(0, 100); axes[1].set_ylabel('데이터셋 내 비율 (%)'); axes[1].set_title('클래스 구성비')
    for ax in axes:
        ax.set_xticks(xs, [TITLE[d].replace(' · ', '\n') for d in SETS]); ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
    fig.suptitle('데이터셋별 bona fide · spoof 표본 수', x=.07, ha='left', fontsize=17, weight='bold')
    legend(fig, .94)
    fig.text(.07, .025, '이후 히스토그램은 각 클래스 합계가 100%가 되도록 정규화합니다. 표본 수 불균형과 분포 모양을 구분합니다.', fontsize=10)
    fig.subplots_adjust(top=.78, bottom=.19, left=.07, right=.98, wspace=.28)
    save(fig, '01_class_counts', '클래스별 표본 수와 구성비', '5개 전체 데이터셋')

def durations(frames):
    maximum = max(f.duration.max() for f in frames.values())
    edges = np.arange(0, np.ceil(maximum*4)/4+.26, .25)
    fig, axes = grid('원본 음성 길이 분포', '클래스별 비율 · 0.25초 공통 bin · 전체 범위 표시 · 점선 기준: Short 3.0275초 / Long 4.0375초')
    for ax, dataset in zip(axes, SETS):
        f = frames[dataset]
        for y in [1, 0]:
            v = f.loc[f.y == y, 'duration'].to_numpy()
            hist(ax, dataset, y, 'duration_s', v, edges)
        for t in [3.0275, 4.0375]: ax.axvline(t, color='#69727b', lw=.8, alpha=.7)
        ax.set(xlim=(0, edges[-1]), xlabel='원본 길이 (초)', ylabel='클래스 내 발화 비율 / bin (%)')
    ymax = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes: ax.set_ylim(0, ymax)
    save(fig, '02_duration_histogram', '원본 길이 히스토그램', '5개 전체 데이터셋')
    fig, axes = grid('길이 누적분포: 짧은 음성이 얼마나 많은가', '전체 길이 ECDF · 클래스 내 누적 비율 · 수직선은 2초 / Short 3.0275초 / Long 4.0375초')
    rates = []
    for ax, dataset in zip(axes, SETS):
        f = frames[dataset]
        for y in [1, 0]:
            v = np.sort(f.loc[f.y == y, 'duration'].to_numpy())
            # Keep every distinct value: no sampling or percentile truncation.
            unique, n = np.unique(v, return_counts=True)
            ax.step(np.r_[0, unique], np.r_[0, np.cumsum(n)/len(v)*100], where='post', color=COLORS[y], ls=LS[y], lw=1.7)
            rates.append({'dataset': dataset, 'class': LABELS[y], 'N': len(v), 'median_s': float(np.median(v)),
                          'under2_pct': float(100*np.mean(v<2)), 'under_short_pct': float(100*np.mean(v<3.0275)),
                          'under_long_pct': float(100*np.mean(v<4.0375))})
        for t in [2, 3.0275, 4.0375]: ax.axvline(t, color='#69727b', lw=.7, alpha=.6)
        ax.set(xlim=(0, maximum+.2), ylim=(0, 100), xlabel='원본 길이 (초)', ylabel='누적 비율 (%)')
    pd.DataFrame(rates).to_csv(OUT/'length_summary.csv', index=False, encoding='utf-8-sig')
    save(fig, '03_duration_ecdf', '원본 길이 누적분포', '5개 전체 데이터셋')
    fig, axes = grid('Short · Long 길이 기준으로 나눈 분포', '각 클래스 합계 = 100% · 구간은 왼쪽 경계 포함 / 오른쪽 경계 제외 · 마지막은 4.0375초 이상')
    for ax, dataset in zip(axes, SETS):
        f = frames[dataset]; categories = list(f.length_band.cat.categories); x = np.arange(4)
        for y in [1, 0]:
            g=f[f.y==y]; pct=g.length_band.value_counts(sort=False).reindex(categories, fill_value=0).to_numpy()/len(g)*100
            b=ax.bar(x+(-.19 if y else .19), pct, .36, color=COLORS[y]); ax.bar_label(b, fmt='%.1f', padding=3, fontsize=9)
        ax.set_xticks(x, ['<2', '2–3.0275', '3.0275–4.0375', '≥4.0375'], rotation=18)
        ax.set(ylabel='클래스 내 비율 (%)', xlabel='원본 길이 구간 (초)', ylim=(0, 80))
    save(fig, '04_length_bands', '길이 구간별 클래스 분포', '5개 전체 데이터셋')
    for i, (target, tag) in enumerate([(3.0275, 'Short'), (4.0375, 'Long')], 5):
        fig, axes=grid(f'{tag} 고정 길이로 맞출 때 부족한 길이 비율', f'설명용 계산 max(0, {target} − 원본 길이) / {target} · 0%에는 상한 이상 원본 포함 · 실제 C 배치 패딩 측정값이 아님')
        for ax, dataset in zip(axes, SETS):
            for y in [1, 0]:
                v=np.maximum(0, target-frames[dataset].loc[frames[dataset].y==y, 'duration'].to_numpy())/target
                hist(ax, dataset, y, f'deficit_{tag}', v, np.linspace(0, 1, 26), scale=100)
            ax.set(xlim=(0,100), xlabel='고정 길이 대비 부족분 (%)', ylabel='클래스 내 발화 비율 / bin (%)')
        ymax=max(a.get_ylim()[1] for a in axes)
        for a in axes:a.set_ylim(0,ymax)
        save(fig, f'{i:02d}_{tag.lower()}_deficit', f'{tag} 길이 부족분 비율', '원본 길이에서 계산한 설명용 지표')

def numeric_distributions(frames):
    datasets=SETS[1:]
    for number, column, name, title in [(7,'score_B','B_score','B 모델의 bona fide · spoof 점수 분포'),
                                       (8,'score_legacy240','legacy_score','기존240 모델 점수 분포 · 동일한 B 평가 입력')]:
        fig, axes=grid(title, '클래스별 비율 · 코사인 점수, 클수록 bona fide 방향 · 모든 패널 동일 범위/bin · C 학습 중간점수가 아님', datasets)
        for ax,d in zip(axes,datasets):
            for y in [1,0]:hist(ax,d,y,column,frames[d].loc[frames[d].y==y,column].to_numpy(),np.linspace(-1,1,81))
            ax.set(xlim=(-1,1),xlabel='Bona fide centroid와의 cosine',ylabel='클래스 내 발화 비율 / bin (%)')
        ymax=max(a.get_ylim()[1] for a in axes)
        for a in axes:a.set_ylim(0,ymax)
        save(fig,f'{number:02d}_{name}',title,'dev/eval 전체 · 보존된 fixed-50 B와 기존240의 동일 B 입력 결과')
    fig,axes=grid('기존240 임베딩 크기 분포', '기존 wrap 입력 캐시의 raw norm · 각 클래스 합계 = 100% · B 점수 그림과 입력 방식이 다름')
    low=min(f.raw_norm.min() for f in frames.values());high=max(f.raw_norm.max() for f in frames.values());edges=np.linspace(low-1e-6,high+1e-6,61)
    for ax,d in zip(axes,SETS):
        for y in [1,0]:hist(ax,d,y,'legacy_wrap_raw_norm',frames[d].loc[frames[d].y==y,'raw_norm'].to_numpy(),edges)
        ax.set(xlim=(edges[0],edges[-1]),xlabel='160차원 z의 L2 norm',ylabel='클래스 내 발화 비율 / bin (%)')
        ax.ticklabel_format(axis='x', style='plain', useOffset=False)
    ymax=max(a.get_ylim()[1] for a in axes)
    for a in axes:a.set_ylim(0,ymax)
    save(fig,'09_embedding_norm','임베딩 L2 norm 분포','기존240 wrap 입력 · 5개 전체 데이터셋')
    fig,axes=grid('길이 구간별 B 점수의 중앙값과 분포 폭', '점 = 중앙값 · 선 = 5–95백분위 범위(신뢰구간 아님) · 왼쪽/오른쪽으로 클래스를 약간 이동해 겹침 방지',datasets)
    for ax,d in zip(axes,datasets):
        f=frames[d]
        for y in [1,0]:
            for j,band in enumerate(f.length_band.cat.categories):
                v=f.loc[(f.y==y)&(f.length_band==band),'score_B'].to_numpy()
                if not len(v):continue
                lo,med,hi=np.quantile(v,[.05,.5,.95]);x=j+(-.08 if y else .08)
                ax.errorbar(x,med,yerr=np.array([[med-lo],[hi-med]]),fmt='o' if y else 's',color=COLORS[y],capsize=4,ms=5)
        ax.set_xticks(range(4),['<2','2–3.0275','3.0275–4.0375','≥4.0375'],rotation=18)
        ax.set(ylim=(-1.05,1.05),ylabel='B cosine 점수',xlabel='원본 길이 구간 (초)')
    save(fig,'10_score_by_length','길이 구간별 B 점수 분포','dev/eval 전체 · 분위수 범위')

def category_plot(frames, datasets, column, name, title):
    allcats=sorted(set().union(*(set(frames[d][column].dropna().astype(str)) for d in datasets)))
    max_pct=max(100*frames[d].loc[frames[d].y==y,column].value_counts().max()/(frames[d].y==y).sum() for d in datasets for y in [1,0])
    ymax=min(105,max(20,np.ceil(max_pct/10)*10+5))
    fig,axes=plt.subplots(len(datasets),1,figsize=(13, 4.8*len(datasets)),squeeze=False)
    fig.suptitle(title,x=.09,y=.99,ha='left',fontsize=17,weight='bold');legend(fig,.955)
    for ax,d in zip(axes.ravel(),datasets):
        f=frames[d];x=np.arange(len(allcats))
        for y in [1,0]:
            g=f[f.y==y];counts=g[column].value_counts().reindex(allcats,fill_value=0);pct=counts/len(g)*100
            b=ax.bar(x+(-.19 if y else .19),pct,.36,color=COLORS[y]);ax.bar_label(b,labels=[f'{v:.1f}' if v>=.1 else '' for v in pct],padding=2,fontsize=9)
        ax.set_xticks(x,allcats,rotation=25,ha='right');ax.set_title(TITLE[d],loc='left');ax.set_ylabel('클래스 내 비율 (%)');ax.set_ylim(0,ymax)
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.text(.09,.018,'각 데이터셋·클래스별 합계 = 100% · 범주 구성의 차이는 분포 차이이며 탐지 성능이나 인과 효과를 뜻하지 않습니다.',fontsize=10)
    fig.subplots_adjust(left=.09,right=.98,top=.85 if len(datasets)==1 else .88,bottom=.18 if len(datasets)==1 else .1,hspace=.65)
    save(fig,name,title,'전체 메타데이터 · 클래스 내 정규화')

def categories(frames):
    category_plot(frames,SETS[-2:],'codec','11_codec','코덱별 bona fide · spoof 구성')
    category_plot(frames,['21DF_eval'],'source_corpus','12_source_corpus','2021 DF 원본 코퍼스별 클래스 구성')
    category_plot(frames,['21LA_eval'],'transmission','13_transmission','2021 LA 전송 조건별 클래스 구성')
    # Attack labels are spoof mechanisms, while bona fide has no spoof attack.
    fig,axes=grid('공격 유형별 spoof 표본 분포','분모 = 각 데이터셋의 spoof 전체 · bona fide에는 spoof 공격 ID가 없으므로 이 그림에서 제외 · 21DF는 최대 12개 유형+기타')
    attackrows=[]
    for ax,d in zip(axes,SETS):
        f=frames[d];c=f.loc[f.y==0,'attack_id'].value_counts();total=int(c.sum())
        for attack,n in c.items():attackrows.append({'dataset':d,'attack':attack,'N':int(n),'spoof_share_pct':100*n/total})
        if len(c)>13:c=pd.concat([c.iloc[:12],pd.Series({'기타':c.iloc[12:].sum()})])
        b=ax.bar(np.arange(len(c)),100*c/total,color=COLORS[0]);ax.set_xticks(range(len(c)),c.index,rotation=65,ha='right',fontsize=8)
        ax.set(ylabel='Spoof 내 비율 (%)',ylim=(0,40))
        if (100*c/total).max()>40:ax.set_ylim(0,105)
    # Remove inappropriate two-class legend for this spoof-only chart.
    for l in fig.legends:l.remove()
    pd.DataFrame(attackrows).to_csv(OUT/'attack_counts.csv',index=False,encoding='utf-8-sig')
    save(fig,'14_attack_types','공격 유형별 spoof 표본 분포','전체 spoof · 21DF 긴 꼬리 범주만 기타로 합침')
    fig,axes=grid('화자별 발화 수의 분포','점 1개 단위 = 프로토콜 화자 그룹 · 클래스별 누적 비율 · 실제 생물학적 화자 동일성을 새로 검증한 것은 아님')
    speakerrows=[]
    for ax,d in zip(axes,SETS):
        f=frames[d]
        for y in [1,0]:
            counts=f[f.y==y].groupby('speaker_group').size();v=np.sort(counts.to_numpy())
            ax.step(np.r_[0,v],np.r_[0,np.arange(1,len(v)+1)/len(v)*100],where='post',color=COLORS[y],ls=LS[y],lw=1.7)
            speakerrows.append({'dataset':d,'class':LABELS[y],'speaker_groups':len(v),'utterances':int(v.sum()),'max_utterances_per_group':int(v.max())})
        ax.set_xscale('log')
        ax.set(xlabel='화자 그룹당 발화 수 · 로그축',ylabel='화자 그룹 누적 비율 (%)',ylim=(0,100))
    maxx=max(ax.get_xlim()[1] for ax in axes)
    for ax in axes:ax.set_xlim(1,maxx)
    pd.DataFrame(speakerrows).to_csv(OUT/'speaker_summary.csv',index=False,encoding='utf-8-sig')
    save(fig,'15_speaker_ecdf','프로토콜 화자별 발화 수 분포','전체 · 발화 대신 화자 그룹을 분모로 사용')

def acoustics():
    path=OUT/'prior_acoustic_histograms.json'
    if not path.exists():return
    source(path);data=json.loads(path.read_text())
    metrics=[('20_fraction','저에너지 구간 비율 · −20 dB','시간 비율 (%)',100),
             ('40_fraction','저에너지 구간 비율 · −40 dB','시간 비율 (%)',100),
             ('20_leading_s','앞부분 저에너지 · −20 dB','초',1),
             ('20_trailing_s','뒷부분 저에너지 · −20 dB','초',1),
             ('20_internal_s','중간 저에너지 총 길이 · −20 dB','초',1),
             ('20_longest_internal_s','가장 긴 중간 저에너지 · −20 dB','초',1),
             ('20_exactzero_fraction','정확히 0인 샘플 비율','샘플 비율 (%)',100),
             ('20_internal_runs','중간 저에너지 구간 개수 · −20 dB','구간 개수',1),
             ('20_quiet_energy_fraction','저에너지 부분의 에너지 비율 · −20 dB','전체 에너지 대비 (%)',100)]
    summary=[]
    for i,d in enumerate(SETS,16):
        key='21DF_ASV_eval' if d=='21DF_eval' else d
        group=data['groups'][key];fig,axes=plt.subplots(3,3,figsize=(15,11))
        scope='전체 원본 발화' if d in SETS[:2] else '클래스별 1,000개 기존 무작위 표본'
        if d=='21DF_eval':scope+=' · ASVspoof 출처만, VCC 제외'
        fig.suptitle(TITLE[d]+' · 저에너지·정확한 0값 분포',x=.07,y=.985,ha='left',fontsize=17,weight='bold')
        fig.text(.07,.945,scope,fontsize=11);legend(fig,.925)
        for ax,(metric,title,xlabel,scale) in zip(axes.ravel(),metrics):
            for y in [1,0]:
                m=group[str(y)]['features'][metric];n=group[str(y)]['n'];c=np.array(m['counts']);edges=np.array(m['edges'])*scale
                assert c.sum()==n
                ax.stairs(c/n*100,edges,color=COLORS[y],ls=LS[y],lw=1.5)
                summary.append({'dataset':key,'class':LABELS[y],'metric':metric,'N':n,'mean':m['mean'],**m['quantiles']})
            ax.set_title(title,loc='left',fontsize=11);ax.set(xlabel=xlabel,ylabel='클래스 내 비율 / bin (%)');ax.grid(axis='y',alpha=.15)
            if metric=='20_quiet_energy_fraction':
                ax.set_xscale('symlog',linthresh=1e-8)
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: '0' if x == 0 else f'{x:g}'))
                ax.set_xticks([0,1e-6,1e-3,1,10])
                ax.set_xlabel('전체 에너지 대비 % · symlog (0 부근 선형)')
        fig.text(.07,.035,'20ms 비중첩 RMS가 발화 자체 최대 RMS보다 20/40 dB 낮은 구간 · 사람이 주석한 무음/VAD가 아닙니다.\n원본 전체 음성의 과거 측정치이며, 현재 C 입력·실제 발화 구간·오류 원인을 뜻하지 않습니다. 모든 값은 클래스 내 정규화.',fontsize=10,color='#46525e')
        fig.subplots_adjust(left=.07,right=.98,top=.84,bottom=.13,hspace=.62,wspace=.34)
        save(fig,f'{i:02d}_acoustic_{d}',TITLE[d]+' 저에너지·0값 9개 기준',scope)
    pd.DataFrame(summary).to_csv(OUT/'acoustic_summary.csv',index=False,encoding='utf-8-sig')

def pca_view(frames):
    rng=np.random.default_rng(20261108);samples={};feature_sources=set()
    for d in SETS:
        f=frames[d];parts=[]
        for y in [1,0]:
            g=f[f.y==y];parts.append(g.iloc[rng.choice(len(g),min(1000,len(g)),replace=False)])
        s=pd.concat(parts,ignore_index=True);z=np.empty((len(s),160))
        for alias in s.cache_alias.unique():
            folder,entry,_=resolve_run(alias)
            for rank in [0,1]:
                pos=np.flatnonzero((s.cache_alias==alias)&(s['rank']==rank));p=folder/f'extract/rank{rank}_z.npy'
                if p not in feature_sources:
                    expected=next(m['sha256'] for m in entry['files'] if m['path']==f'extract/rank{rank}_z.npy');source(p,expected);feature_sources.add(p)
                a=np.load(p,mmap_mode='r',allow_pickle=False);z[pos]=a[s.iloc[pos].embedding_row.to_numpy()]
        z/=np.linalg.norm(z,axis=1,keepdims=True);samples[d]=(s,z)
    pca=PCA(2,svd_solver='full').fit(samples['19LA_train'][1]);projected={d:pca.transform(z) for d,(s,z) in samples.items()}
    allxy=np.concatenate(list(projected.values()));lo=allxy.min(axis=0);hi=allxy.max(axis=0);margin=(hi-lo)*.05
    fig,axes=grid('공통 PCA 좌표에서 본 임베딩 분포','기존240 wrap 입력의 단위 z · 각 데이터셋 클래스별 무작위 1,000개 · PCA는 train의 균형 표본 2,000개에만 fit · 모든 패널 공통 축')
    rows=[]
    for ax,d in zip(axes,SETS):
        s,z=samples[d];xy=projected[d]
        for y in [1,0]:
            mask=s.y.to_numpy()==y;ax.scatter(xy[mask,0],xy[mask,1],s=6,alpha=.25,c=COLORS[y],marker='o' if y else 'x',linewidths=.5,rasterized=True)
        ax.set(xlim=(lo[0]-margin[0],hi[0]+margin[0]),ylim=(lo[1]-margin[1],hi[1]+margin[1]),xlabel=f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',ylabel=f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        rows.append(s[['utterance_id','y']].assign(dataset=d,PC1=xy[:,0],PC2=xy[:,1]))
    pd.concat(rows).to_csv(OUT/'pca_plotted_points.csv',index=False,encoding='utf-8-sig')
    np.savez(OUT/'pca_parameters.npz',mean=pca.mean_,components=pca.components_,explained_variance_ratio=pca.explained_variance_ratio_)
    save(fig,'21_common_pca','공통 PCA 임베딩 분포','클래스 균형 표본 · 시각화이며 분리도/성능 검증이 아님')

def duration_zoom(frames):
    fig,axes=grid('원본 길이 분포 · 0–10초 확대','전체 발화를 분모로 유지 · 0.2초 공통 bin · 10초 이후 꼬리는 각 패널에 비율 표시 · 전체 범위는 02번 그림')
    edges=np.arange(0,10.001,.2)
    for ax,d in zip(axes,SETS):
        notes=[]
        for y in [1,0]:
            v=frames[d].loc[frames[d].y==y,'duration'].to_numpy();c,_=np.histogram(v,edges)
            ax.stairs(100*c/len(v),edges,color=COLORS[y],ls=LS[y],lw=1.7)
            notes.append(f'{LABELS[y]}: {100*np.mean(v>10):.2f}%')
        for t in [2,3.0275,4.0375]:ax.axvline(t,color='#69727b',lw=.7,alpha=.6)
        ax.text(.98,.97,'10초 초과\n'+'\n'.join(notes),transform=ax.transAxes,ha='right',va='top',fontsize=9)
        ax.set(xlim=(0,10),xlabel='원본 길이 (초)',ylabel='클래스 내 발화 비율 / bin (%)')
    ymax=max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:ax.set_ylim(0,ymax)
    save(fig,'22_duration_zoom','길이 분포 0–10초 확대','전체 발화를 분모로 유지 · 10초 초과 비율 명시')

def gallery(frames):
    pd.DataFrame(NUMERIC).to_csv(OUT/'numeric_summary.csv',index=False,encoding='utf-8-sig')
    (OUT/'histogram_data.json').write_text(json.dumps(HISTOGRAMS,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    validation={'status':'passed','dataset_checks':CHECKS,'total_utterances':sum(len(f) for f in frames.values()),
                'figure_count':len(GALLERY),'source_files':list({s['path']:s for s in SOURCES}.values()),
                'checks':['Cached score/embedding SHA256 matches catalog or download manifest',
                          'V + Q reconstruct full official 19LA dev with no duplicate IDs',
                          'DF_pilot + DF_remaining reconstruct 533928 official eval utterances without duplicates',
                          'B/legacy paired UID, label and original sample count agree with geometry cache',
                          'All histogram bins contain every input value, with class-specific denominators',
                          'PCA fit only on balanced train sample; same projection and axes for every dataset'],
                'limitations':['Descriptive benchmark distributions; no new performance or causal claim',
                               'Acoustics: eval samples, DF ASVspoof source only',
                               'Prior max_rms_dbfs has positive values up to 55.79; excluded from figures because amplitude normalization is unverified',
                               'C training scores unavailable during training; no C score included']}
    (OUT/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# Bona fide · Spoof 데이터셋별 분포 그래프','',
           '파랑 실선/원 = bona fide, 주황 점선/× = spoof. 히스토그램은 **각 데이터셋·클래스 안에서 합계 100%**이며 bin당 발화 비율입니다.','',
           '| 데이터셋 | Bona fide | Spoof | 합계 |','|---|---:|---:|---:|']
    for d in SETS:
        n,b=EXPECTED[d];lines.append(f'| {TITLE[d]} | {b:,} | {n-b:,} | {n:,} |')
    lines+=['',f'전체 {sum(len(f) for f in frames.values()):,}개 발화. 2019 dev는 기존 V+Q 캐시를 합쳐 전체 24,844개를 복원했습니다. 2021은 공식 eval 필터만 포함하고 progress/hidden은 제외했습니다.','',
            '길이는 원본 전체 길이입니다. Short/Long 부족분은 고정 길이 대비 계산한 값으로 C의 실제 배치 패딩 비율이 아닙니다. B/기존240 점수는 이미 완료된 동일 B 입력 평가이며, 임베딩 norm/PCA는 기존240의 wrap 입력 캐시이므로 입력 방식이 다릅니다.','',
            '음향 그림의 저에너지는 발화 자체 최대 20ms RMS보다 20/40dB 낮은 구간입니다. 사람 주석 무음이나 VAD가 아닙니다. train/dev는 전체이며, eval은 클래스별 1,000개 표본입니다. **21DF 음향 그림만 ASVspoof 출처로 제한되며 VCC를 포함하지 않습니다.** 무음으로 탐지 오류를 설명하는 인과 분석은 하지 않았습니다.','',
            '과거 `max_rms_dbfs`에 +55.79까지의 값이 있어 진폭 정규화가 검증되지 않은 절대 RMS 비교는 제외했습니다. 피치·SNR·스펙트럼 특징은 5개 데이터셋의 두 클래스에 대해 비교 가능한 기존 측정치가 없어 생성하지 않았습니다. 새 원음 처리·학습·GPU 추론은 수행하지 않았습니다.','',
            '[길이 요약 CSV](length_summary.csv) · [수치 요약 CSV](numeric_summary.csv) · [음향 요약 CSV](acoustic_summary.csv) · [범위와 검산 기록](validation.json)','']
    for item in GALLERY:
        lines += [f'## {item["title"]}', '',item['scope'], '',f'![{item["title"]}](figures/{item["name"]}.png)', '',f'[확대 PNG](figures/{item["name"]}.png) · [편집 가능한 SVG](figures/{item["name"]}.svg)', '']
    (OUT/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'status':'complete','figures':len(GALLERY),'N':validation['total_utterances'],'gallery':str(OUT/'README.md')},ensure_ascii=False))

def main():
    FIG.mkdir(parents=True,exist_ok=True)
    frames=load();counts_plot(frames);durations(frames);numeric_distributions(frames);categories(frames);acoustics();pca_view(frames);duration_zoom(frames);gallery(frames)

if __name__=='__main__':
    main()
