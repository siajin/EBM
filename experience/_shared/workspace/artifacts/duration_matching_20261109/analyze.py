"""Metadata-only constrained duration matching. No audio/model/server work.

Each class is a uniform probability measure over all its utterances. In the
maximum-overlap policy an utterance may randomize among feasible output lengths;
there is no deletion, duplication weight, or change to original sampling weight.
Integer LCM masses avoid approximation of the unequal class sizes.
"""
from collections import deque
from math import gcd
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import ks_2samp, wasserstein_distance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCES = [
    '20261105_fast_p0_01/686e593443178bb8',
    '20261105_frozen_all_01/Q/c288fb7a7ab48dd3',
    '20261105_frozen_all_01/19LA_eval/4b8955da49f12316',
    '20261105_21la_full_03/4110118cb92e35c1',
    '20261105_df_fast_01/1b0a7488e72f4ec2',
    '20261105_frozen_all_01/DF_remaining/096bc3b970f33181',
]
EXPECTED = [(2580, 22800), (2548, 22296), (7355, 63882), (14816, 133360), (14869, 519059)]


def load():
    known = json.loads((ROOT/'artifacts/class_distributions_20261108/validation.json').read_text())
    hashes = {str(Path(x['path'])): x['sha256'] for x in known['source_files']}
    frames, provenance = [], []
    for source in SOURCES:
        p = ROOT/'artifacts/geometry_cache/runs'/source/'extract/scores.parquet'
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        assert sha == hashes[str(p.relative_to(ROOT))]
        frames.append(pd.read_parquet(p, columns=['split','utterance_id','y','duration']))
        provenance.append({'path': str(p), 'sha256': sha})
    f = frames[0]
    return [
        ('19LA_train', '2019 LA · train', f[f.split == 'T']),
        ('19LA_dev', '2019 LA · dev', pd.concat([f[f.split == 'V'],frames[1]])),
        ('19LA_eval', '2019 LA · eval', frames[2]),
        ('21LA_eval', '2021 LA · eval', frames[3]),
        ('21DF_eval', '2021 DF · eval', pd.concat(frames[4:])),
    ], provenance


def match(a, b, ratio_num=15, ratio_den=8, plan=False):
    """Maximum flow on ordered interval compatibility graph, greedy sweep."""
    n, m = len(a), len(b)
    g = gcd(n, m)
    ca, cb = m//g, n//g
    total = n*ca
    ra, rb = np.full(n, ca, dtype=np.int64), np.full(m, cb, dtype=np.int64)
    i=j=0
    mass=0
    ai, bi, units = [], [], []
    while i<n and j<m:
        if ratio_num*int(a[i]) < ratio_den*int(b[j]):
            i+=1
        elif ratio_num*int(b[j]) < ratio_den*int(a[i]):
            j+=1
        else:
            z=min(int(ra[i]),int(rb[j]))
            mass+=z
            ra[i]-=z; rb[j]-=z
            if plan:
                ai.append(i); bi.append(j); units.append(z)
            if ra[i]==0: i+=1
            if rb[j]==0: j+=1
    result={'matched_units':mass,'total_units':total,'max_overlap':mass/total,
            'class_capacity_units':[ca,cb]}
    if plan:
        result.update(ai=np.asarray(ai),bi=np.asarray(bi),units=np.asarray(units),ra=ra,rb=rb)
    return result


def mincut_certificate(a,b,p):
    """Independent optimality witness by residual reachability, no dense graph.

    From every unmatched A follow ALL compatible A->B arcs, then positive-flow
    reverse B->A arcs. If unmatched B is unreachable, the resulting cut must
    equal our flow; its capacity is an independently checkable upper bound.
    """
    n,m=len(a),len(b)
    ca,cb=p['class_capacity_units']
    seen_a=p['ra']>0
    seen_b=np.zeros(m,dtype=bool)
    queue=deque(np.flatnonzero(seen_a).tolist())
    reverse=csr_matrix((np.ones(len(p['ai']),dtype=np.int8),(p['bi'],p['ai'])),shape=(m,n))
    nxt=np.arange(m+1,dtype=np.int64)
    def find(k):
        root=k
        while nxt[root]!=root: root=int(nxt[root])
        while nxt[k]!=k:
            q=int(nxt[k]);nxt[k]=root;k=q
        return root
    while queue:
        i=queue.popleft()
        lo=(8*int(a[i])+14)//15
        hi=(15*int(a[i]))//8
        left=int(np.searchsorted(b,lo,side='left'))
        right=int(np.searchsorted(b,hi,side='right'))
        j=find(left)
        while j<right:
            seen_b[j]=True
            for k in reverse.indices[reverse.indptr[j]:reverse.indptr[j+1]]:
                if not seen_a[k]: seen_a[k]=True;queue.append(int(k))
            nxt[j]=find(j+1)
            j=find(j)
    assert not np.any(seen_b & (p['rb']>0)), 'Augmenting path exists'
    cut=int((~seen_a).sum())*ca+int(seen_b.sum())*cb
    assert cut==p['matched_units'], (cut,p['matched_units'])
    return {'flow_units':p['matched_units'],'mincut_units':cut,'total_units':p['total_units'],
            'reachable_bonafide':int(seen_a.sum()),'reachable_spoof':int(seen_b.sum()),
            'unmatched_spoof_reachable':False,'maxflow_equals_mincut':True}


def weighted_median_targets(index,target,units,n,capacity):
    order=np.lexsort((target,index))
    idx=index[order];t=target[order];w=units[order]
    starts=np.r_[0,np.flatnonzero(np.diff(idx))+1]
    ends=np.r_[starts[1:],len(idx)]
    sums=np.add.reduceat(w,starts)
    assert len(starts)==n and np.all(sums==capacity)
    cumulative=np.cumsum(w)
    prior=np.r_[0,cumulative[starts[1:]-1]]
    within=cumulative-np.repeat(prior,ends-starts)
    crossing=2*within>=capacity
    chosen=np.flatnonzero(crossing & np.r_[True,(idx[1:]!=idx[:-1]) | ~crossing[:-1]])
    assert len(chosen)==n and np.array_equal(idx[chosen],np.arange(n))
    return t[chosen]


def operations(original,target):
    rate=np.clip(original/target,.8,1.2)
    crop=np.maximum(0,1-target*rate/original)
    assert rate.min()>=.8-1e-12 and rate.max()<=1.2+1e-12
    assert crop.min()>=0 and crop.max()<=.2+1e-12
    assert np.max(np.abs(original*(1-crop)/rate-target))<1e-10
    return rate,crop


def overlap(a,b,edges,weights_a=None,weights_b=None):
    x=np.histogram(a,bins=edges,weights=weights_a)[0].astype(float)
    y=np.histogram(b,bins=edges,weights=weights_b)[0].astype(float)
    x/=x.sum();y/=y.sum()
    return float(np.minimum(x,y).sum())


def main():
    datasets,provenance=load()
    summary,plans,rows,certificates,oprows,curves=[],[],[],{},[],{}
    for (key,title,df),counts in zip(datasets,EXPECTED):
        assert df.utterance_id.is_unique and df.duration.notna().all() and np.isfinite(df.duration).all()
        classes=[]
        for label in [1,0]:
            f=df[df.y==label].sort_values(['duration','utterance_id']).reset_index(drop=True)
            samples=np.rint(f.duration.to_numpy()*16000).astype('int64')
            assert np.max(np.abs(samples/16000-f.duration))<1e-12 and (samples>0).all()
            classes.append((f,samples))
        fa,a=classes[0];fb,b=classes[1]
        assert (len(a),len(b))==counts
        p=match(a,b,plan=True)
        certificates[key]=mincut_certificate(a,b,p)
        ai,bi,w=p['ai'],p['bi'],p['units']
        # Minimal symmetric log-duration movement in each feasible intersection.
        lo=np.maximum(a[ai],b[bi])*(2/3)
        hi=np.minimum(a[ai],b[bi])*1.25
        assert np.all(lo<=hi+1e-9)
        target=np.clip(np.sqrt(a[ai].astype(float)*b[bi]),lo,hi)/16000
        r_a=np.flatnonzero(p['ra']);r_b=np.flatnonzero(p['rb'])
        ia=np.r_[ai,r_a];ib=np.r_[bi,r_b]
        ta=np.r_[target,a[r_a]/16000];tb=np.r_[target,b[r_b]/16000]
        wa=np.r_[w,p['ra'][r_a]];wb=np.r_[w,p['rb'][r_b]]
        ca,cb=p['class_capacity_units']
        assert wa.sum()==wb.sum()==p['total_units']
        da=weighted_median_targets(ia,ta,wa,len(a),ca)
        db=weighted_median_targets(ib,tb,wb,len(b),cb)
        for f,orig,trans,y in [(fa,a/16000,da,1),(fb,b/16000,db,0)]:
            rate,crop=operations(orig,trans)
            plans.append(pd.DataFrame({'dataset':key,'utterance_id':f.utterance_id,'y':y,
                'original_duration_s':orig,'target_duration_s':trans,'speed_rate':rate,'crop_fraction':crop}))
            oprows.append({'dataset':key,'class':'bonafide' if y else 'spoof','N':len(orig),
                'slowed_pct':float(100*np.mean(rate<1-1e-10)),
                'sped_up_pct':float(100*np.mean(rate>1+1e-10)),
                'cropped_pct':float(100*np.mean(crop>1e-10)),
                'mean_crop_pct':float(100*crop.mean()),'max_crop_pct':float(100*crop.max()),
                'rate_min':float(rate.min()),'rate_max':float(rate.max()),
                'before_under2_pct':float(100*np.mean(orig<2)),
                'after_under2_pct':float(100*np.mean(trans<2))})
        # Randomized policy pieces retain each original file's equal exposure.
        for y,ix,t,u,orig,cap in [(1,ia,ta,wa,a,ca),(0,ib,tb,wb,b,cb)]:
            rate,crop=operations(orig[ix]/16000,t)
            pd.DataFrame({'source_sorted_index':ix,'target_duration_s':t,'mass_units':u,
                'conditional_probability':u/cap,'speed_rate':rate,'crop_fraction':crop}).to_parquet(
                    OUT/f'{key}_{y}_randomized_segments.parquet',index=False)
        row={'dataset':key,'bonafide_N':len(a),'spoof_N':len(b),
            'max_continuous_overlap_pct':100*p['max_overlap'],
            'unmatched_pct':100*(1-p['max_overlap']),
            'speed_only_max_overlap_pct':100*match(a,b,3,2)['max_overlap'],
            'crop_only_max_overlap_pct':100*match(a,b,5,4)['max_overlap'],
            'KS_before':float(ks_2samp(a,b).statistic),
            'KS_single_after':float(ks_2samp(da,db).statistic),
            'W1_before_s':float(wasserstein_distance(a/16000,b/16000)),
            'W1_single_after_s':float(wasserstein_distance(da,db))}
        for width in [.05,.1,.25]:
            edges=np.arange(0,31+width/2,width)
            row[f'overlap_before_bin{width}s_pct']=100*overlap(a/16000,b/16000,edges)
            row[f'overlap_single_after_bin{width}s_pct']=100*overlap(da,db,edges)
            row[f'overlap_random_after_bin{width}s_pct']=100*overlap(ta,tb,edges,wa,wb)
            assert row[f'overlap_random_after_bin{width}s_pct']+1e-7>=100*p['max_overlap']
        summary.append(row)
        edges=np.arange(0,30.05,.05)
        curves[key]=(title,a/16000,b/16000,da,db,ta,tb,wa,wb)
        for mode,xx,yy,wx,wy in [('before',a/16000,b/16000,None,None),
                                 ('single',da,db,None,None),('random',ta,tb,wa,wb)]:
            for cls,vals,weights in [('bonafide',xx,wx),('spoof',yy,wy)]:
                h=np.histogram(vals,bins=edges,weights=weights)[0].astype(float)
                h=h/h.sum()*100
                rows.extend({'dataset':key,'policy':mode,'class':cls,'bin_left_s':lo,
                             'bin_right_s':hi,'percent':v} for lo,hi,v in zip(edges[:-1],edges[1:],h))
        print(key,json.dumps(row),flush=True)
    table=pd.DataFrame(summary)
    table.to_csv(OUT/'summary.csv',index=False)
    pd.concat(plans,ignore_index=True).to_parquet(OUT/'single_transform_plan.parquet',index=False)
    pd.DataFrame(oprows).to_csv(OUT/'operations.csv',index=False)
    pd.DataFrame(rows).to_csv(OUT/'histograms.csv',index=False)
    (OUT/'validation.json').write_text(json.dumps({'sources':provenance,'optimality_certificates':certificates,
        'speed_rate':[.8,1.2],'crop_fraction':[0,.2],'duration_multiplier':[2/3,1.25],
        'maximum_overlap_definition':'1 - total variation; maximal shared probability mass, continuous duration, randomized per-utterance augmentation allowed',
        'constraints':'Every utterance retains uniform within-class sampling weight. Both classes may transform. No removal or reweighting.',
        'single_policy':'Conditional weighted-median target of the maximum-overlap randomized plan, one feasible transform per utterance; not claimed globally optimal.',
        'units':'Lengths optimized as continuous seconds from exact 16 kHz source sample counts; waveform output rounding untested.',
        'scope':'Within each dataset independently, using its class labels. Descriptive oracle feasibility, not a train-only deployment or a classification performance result.'},indent=2),encoding='utf-8')
    plot(curves,table)


def plot(curves,table):
    plt.rcParams.update({'font.family':'Malgun Gothic','font.size':11,'axes.unicode_minus':False,
                         'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    colors=['#2376AB','#D97813']
    for end,width,suffix in [(10,.1,'0_10s'),(2,.05,'0_2s')]:
        fig,axes=plt.subplots(5,3,figsize=(18,19),sharex=True,sharey=True)
        edges=np.arange(0,30+width/2,width)
        ymax=0
        for i,(key,data) in enumerate(curves.items()):
            title,a,b,da,db,ta,tb,wa,wb=data
            info=table.loc[table.dataset==key].iloc[0]
            for j,(xx,yy,wx,wy) in enumerate([(a,b,None,None),(ta,tb,wa,wb),(da,db,None,None)]):
                ax=axes[i,j]
                for k,(v,w) in enumerate([(xx,wx),(yy,wy)]):
                    h=np.histogram(v,bins=edges,weights=w)[0].astype(float)
                    h=h/h.sum()*100
                    ax.stairs(h,edges,color=colors[k],lw=1.8,ls='-' if k==0 else '--',baseline=None)
                    ymax=max(ymax,float(h[edges[:-1]<end].max()))
                if j==0:
                    ax.set_ylabel(title+'\n전체 클래스 중 비율 / bin (%)')
                overlap_val=info[f"overlap_{'before' if j==0 else 'random_after' if j==1 else 'single_after'}_bin{width}s_pct"]
                ax.text(.98,.94,f'전체 범위 겹침 {overlap_val:.2f}%',transform=ax.transAxes,
                        ha='right',va='top',fontsize=10,
                        bbox={'facecolor':'white','edgecolor':'none','alpha':.85})
                ax.set_xlim(0,end)
                ax.grid(axis='y',alpha=.2)
                ax.set_axisbelow(True)
                ax.tick_params(labelbottom=True)
                if i==4:ax.set_xlabel('원본 / 변형 후 길이 (초)')
                if i==0:ax.set_title(['변형 전','최대 공통분포 · 반복 증강','발화당 변형 1개'][j],pad=15,fontweight='bold')
        for ax in axes.flat:ax.set_ylim(0,np.ceil(ymax*1.13))
        fig.suptitle(f'속도 0.8–1.2배 + 최대 20% 크롭 · 길이 분포 비교 (0–{end}초)',
                     fontsize=22,x=.065,ha='left',y=.982)
        fig.legend(handles=[Line2D([0],[0],color=colors[0],lw=2,label='Bona fide'),
                            Line2D([0],[0],color=colors[1],lw=2,ls='--',label='Spoof')],
                   loc='upper center',bbox_to_anchor=(.5,.96),ncol=2,frameon=False)
        fig.text(.065,.02,f'{width}초 공통 bin · 분모는 클래스 전체 발화 · 겹침 수치는 0–30초 전체 범위 · 가운데는 발화별 여러 변형의 확률 분포',fontsize=11)
        fig.subplots_adjust(left=.1,right=.98,top=.92,bottom=.065,hspace=.42,wspace=.2)
        fig.savefig(OUT/f'comparison_{suffix}.png',dpi=150,facecolor='white')
        fig.savefig(OUT/f'comparison_{suffix}.svg',facecolor='white')
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5.8))
    x=np.arange(len(table));w=.24
    for pos,col,label,color in [(-1,'crop_only_max_overlap_pct','크롭만 (최대 20%)','#A9B7C0'),
            (0,'speed_only_max_overlap_pct','속도만 (0.8–1.2배)','#D97813'),
            (1,'max_continuous_overlap_pct','속도 + 크롭','#2376AB')]:
        bars=ax.bar(x+pos*w,table[col],width=w,label=label,color=color)
        for bar,value in zip(bars,table[col]):
            ax.text(bar.get_x()+bar.get_width()/2,value+.6,f'{value:.2f}',ha='center',fontsize=9)
    ax.set_xticks(x,['19LA train','19LA dev','19LA eval','21LA eval','21DF eval'])
    ax.set_ylim(0,110);ax.set_ylabel('최대 공통 분포 비율 (%)')
    ax.set_title('허용 변형별 길이 분포의 최대 일치율',loc='left',fontsize=19,pad=22)
    ax.legend(loc='lower left',ncol=3,frameon=False)
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT/'maximum_overlap.png',dpi=180,facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    main()
