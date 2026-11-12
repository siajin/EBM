"""Sensitivity refinement: bona neighborhoods bounded by observed spoof support.

The initial 5-NN balls can encompass spoof points in a shifted dataset. Keep
those original outputs and report the conservative radius min(r5, D_spoof/2).
Inside such an open ball, the center bona is closer than every observed spoof.
This is a sufficient proximity criterion, never a proof of manifold membership.
"""
from spoof_path_geometry import *

OUT = ROOT / 'conservative_neighborhood'


def load_view(alias):
    data=load_embeddings('21LA_eval' if alias=='21LA_pilot' else alias)
    rows,z=data.rows,data.z
    if alias=='21LA_pilot':
        chosen=pd.read_parquet(DEFAULT_ROOT/'analysis_manifests/frozen_data_sizes_v1/21LA_pilot.parquet')
        ids=chosen.cache_row.to_numpy()
        assert rows.iloc[ids].utterance_id.tolist()==chosen.utterance_id.tolist()
        rows,z=rows.iloc[ids].reset_index(drop=True),z[ids]
    x=unit(z.astype('float64'));s=x@data.centroid
    return rows,x,s,unit(x-s[:,None]*data.centroid)


def fixed_s_ratio(aa,bb,s0,bona_x,bona_s,radius):
    w=np.arccos(np.clip(np.sum(aa*bb,1),-1,1))
    vv=(bb-np.cos(w)[:,None]*aa)/np.maximum(np.sin(w)[:,None],1e-14)
    ya,yv=aa@bona_x.T,vv@bona_x.T
    phase=np.arctan2(yv,ya);lo,hi=.1*w[:,None],.9*w[:,None]
    mx=np.maximum(ya*np.cos(lo)+yv*np.sin(lo),ya*np.cos(hi)+yv*np.sin(hi))
    mx=np.maximum(mx,np.where((phase>=lo)&(phase<=hi),np.hypot(ya,yv),-np.inf))
    score=s0[:,None]*bona_s[None,:]+np.sqrt(1-s0*s0)[:,None]*mx
    return (np.sqrt(np.maximum(2-2*score,0))/radius).min(1)


def main():
    OUT.mkdir(exist_ok=False)
    save_json(OUT/'protocol.json',{'UTC':datetime.now(timezone.utc).isoformat(),
        'reason':'Initial DF 5-NN neighborhoods overlap many spoof endpoints; this post-observation sensitivity pass uses a stricter sufficient neighborhood.',
        'radius':'min(distance to 5th other bona, .5 * distance to nearest observed spoof)',
        'primary_inference':'descriptive; conservative ball hit implies bona nearer than all observed spoof at at least one path point; non-hit does not imply no bona region',
        'zero_radius':'floor 1e-12; any cross-class coincidence reported',
        'centerline':'24 equal-count spoof-PC1 bins; observed medoid nearest normalized bin mean; consecutive medoids are illustrative and need not be nearest neighbors',
        'source_sha256':digest(__file__)})
    base=json.loads((ROOT/'summary.json').read_text(encoding='utf-8'))
    metrics=[];radii_records=[];chain_records=[];alignment=[];reference={};plot_inputs={}
    for alias in ['T','V','DF_pilot','21LA_pilot']:
        print('REFINE',alias,flush=True)
        rows,x,s,u=load_view(alias);y=rows.y.to_numpy()
        full_r=None
        for rep,f in [('unit_z',x),('residual_u',u)]:
            sf,bf=f[y==0],f[y==1]
            _,ev,vec=pca(sf)
            mean=sf.mean(0)
            if alias=='T':reference[rep]=vec.copy()
            alignment.append({'dataset':alias,'representation':rep,
                'T_pc1_alignment_abs':float(abs(reference[rep][0]@vec[0])),
                'T_pc12_subspace_overlap':float(np.sum((reference[rep][:2]@vec[:2].T)**2)/2)})
            bd,_=NearestNeighbors(n_neighbors=6,algorithm='brute',n_jobs=1).fit(bf).kneighbors(bf)
            sd,_=NearestNeighbors(n_neighbors=1,algorithm='brute',n_jobs=1).fit(sf).kneighbors(bf)
            r5=bd[:,5];rstrict=np.maximum(np.minimum(r5,.5*sd[:,0]),1e-12)
            radii_records.append({'dataset':alias,'representation':rep,'bona_n':len(bf),
                'radius5_median':float(np.median(r5)),'conservative_radius_median':float(np.median(rstrict)),
                'normal_radius_shrunk_percent':float(100*np.mean(rstrict<r5)),
                'cross_class_coincidences':int(np.sum(sd[:,0]<=1e-10))})
            if rep=='unit_z':full_r=rstrict.copy()
            edges=pd.read_parquet(ROOT/f'{alias}_{rep}_edges.parquet')
            aa,bb=f[edges.a_row],f[edges.b_row]
            for kind in ['chord','arc']:
                _,ratio,ni=continuous_distances(aa,bb,bf,rstrict,kind)
                edges[kind+'_strict_ratio']=ratio
                edges[kind+'_strict_nearest_bona_row']=np.flatnonzero(y==1)[ni]
            er=endpoint_ratio(aa,bb,bf,rstrict)
            # No endpoint lies in a strict ball, by construction.
            assert np.min(er)>=2-1e-7
            if rep=='residual_u':
                s0=(s[edges.a_row]+s[edges.b_row])/2
                edges['fixed_s_full_arc_strict_ratio']=fixed_s_ratio(aa,bb,s0,x[y==1],s[y==1],full_r)
            edges.to_parquet(OUT/f'{alias}_{rep}_edges.parquet',index=False)
            for kind,g in edges.groupby('edge_type',sort=False):
                rec={'dataset':alias,'representation':rep,'edge_type':kind,'edges':len(g),
                     'strict_arc_hits':int((g.arc_strict_ratio<1).sum()),
                     'strict_arc_percent':float(100*(g.arc_strict_ratio<1).mean()),
                     'strict_chord_hits':int((g.chord_strict_ratio<1).sum()),
                     'strict_chord_percent':float(100*(g.chord_strict_ratio<1).mean()),
                     'strict_arc_min_ratio':float(g.arc_strict_ratio.min()),
                     'strict_arc_median_ratio':float(g.arc_strict_ratio.median()),
                     'broad_arc_percent':float(100*(g.arc_radius5_ratio<1).mean()),
                     'midpoint_closer_bona_balanced_percent':float(100*g.midpoint_closer_bona_balanced.mean())}
                if rep=='residual_u':rec['fixed_s_full_arc_strict_percent']=float(100*(g.fixed_s_full_arc_strict_ratio<1).mean())
                metrics.append(rec)
            for condition in ['same_attack_id','same_source_corpus','same_codec','same_speaker_group']:
                if condition in edges:
                    edges.groupby(['edge_type',condition]).agg(edges=('a','size'),strict_arc_hits=('arc_strict_ratio',lambda a:int((a<1).sum()))).to_csv(OUT/f'{alias}_{rep}_{condition}.csv')
            # A traceable, actual-point chain, rather than a fabricated ellipse.
            pc=(sf-mean)@vec[0]
            order=np.argsort(pc);medoids=[]
            for chunk in np.array_split(order,24):
                center=unit(sf[chunk].mean(0)[None,:])[0]
                medoids.append(chunk[np.argmax(sf[chunk]@center)])
            global_ids=np.flatnonzero(y==0)[medoids]
            ca,cb=f[global_ids[:-1]],f[global_ids[1:]]
            _,cr,ni=continuous_distances(ca,cb,bf,rstrict,'arc')
            for j in range(len(cr)):
                chain_records.append({'dataset':alias,'representation':rep,'segment':j,
                    'a_row':int(global_ids[j]),'b_row':int(global_ids[j+1]),
                    'angle_deg':float(np.degrees(np.arccos(np.clip(ca[j]@cb[j],-1,1)))),
                    'score_delta':float(abs(s[global_ids[j]]-s[global_ids[j+1]])),
                    'strict_arc_ratio':float(cr[j]),'strict_arc_hit':bool(cr[j]<1)})
            # Curves for one closest approach and one median long-axis path.
            long=edges.loc[edges.edge_type=='pc1_extremes'].sort_values('arc_strict_ratio')
            illustrations=[long.iloc[0],long.iloc[len(long)//2]]
            curves=[]
            for illustration in illustrations:
                ai,bi=f[int(illustration.a_row)],f[int(illustration.b_row)]
                w=np.arccos(np.clip(ai@bi,-1,1));t=np.linspace(0,1,101)
                points=np.sin((1-t[:,None])*w)/np.sin(w)*ai+np.sin(t[:,None]*w)/np.sin(w)*bi
                d=np.sqrt(np.maximum(2-2*points@bf.T,0))
                curves.append({'t':t,'strict':(d/rstrict).min(1),'broad':(d/r5).min(1),
                               'a_row':int(illustration.a_row),'b_row':int(illustration.b_row)})
            plot_inputs[(alias,rep)]=((f-mean)@vec[:2].T,y,global_ids,curves,ev/ev.sum())
    pd.DataFrame(metrics).to_csv(OUT/'path_metrics.csv',index=False)
    pd.DataFrame(radii_records).to_csv(OUT/'neighborhood_radii.csv',index=False)
    pd.DataFrame(chain_records).to_csv(OUT/'pc1_ordered_chain.csv',index=False)
    pd.DataFrame(alignment).to_csv(OUT/'axis_alignment.csv',index=False)
    render(metrics,plot_inputs)
    save_json(OUT/'summary.json',{'status':'complete_exploratory_sensitivity','metrics':metrics,'neighborhoods':radii_records,
              'axis_alignment':alignment,'chain':chain_records})
    (OUT/'complete').write_text('0\n')
    print('REFINEMENT COMPLETE',flush=True)


def render(metrics,inputs):
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':10})
    frame=pd.DataFrame(metrics)
    fig,axes=plt.subplots(1,2,figsize=(13,5.7),sharey=True)
    kinds=['local_1','local_20','same_score_local','random','pc1_extremes']
    colors=['#3276A8','#D38B32','#71833C','#B45E8B']
    for ax,rep in zip(axes,['unit_z','residual_u']):
        for j,(ds,color) in enumerate(zip(['T','V','DF_pilot','21LA_pilot'],colors)):
            g=frame[(frame.dataset==ds)&(frame.representation==rep)].set_index('edge_type')
            values=[g.loc[k,'strict_arc_percent'] for k in kinds]
            ax.bar(np.arange(5)+(j-1.5)*.19,values,width=.18,color=color,label=ds)
        ax.set_xticks(np.arange(5),[TYPE_NAMES[k].replace(' ','\n') for k in kinds]);ax.set_ylim(0,100)
        ax.set_title(REP_NAMES[rep],loc='left',fontweight='bold');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('보수적인 정상 영역에 진입한 경로 (%)')
    axes[1].legend(frameon=False,ncol=2)
    fig.suptitle('스푸프 연결이 정상 영역을 통과하는가?',fontsize=16,fontweight='bold')
    fig.text(.02,.015,'정상 반경 = min(정상 5번째 이웃 거리, 가장 가까운 스푸프 거리의 절반) · 경로 양끝 10% 제외\n해당 영역에서는 중심 정상 표본이 모든 관찰 스푸프보다 가까움 · 미진입은 전체 정상 영역 부재의 증명이 아님',fontsize=10)
    fig.tight_layout(rect=[0,.11,1,.94]);fig.savefig(OUT/'conservative_overlap.png',dpi=180,facecolor='white');plt.close(fig)
    for alias in ['T','DF_pilot','21LA_pilot']:
        fig,axes=plt.subplots(2,2,figsize=(13,10))
        rng=np.random.default_rng(SEED)
        for j,rep in enumerate(['unit_z','residual_u']):
            coord,y,chain,curves,frac=inputs[(alias,rep)]
            ax=axes[j,0]
            for lab,col,marker,name in [(0,'#D38B32','.', '스푸프'),(1,'#3276A8','x','정상')]:
                ids=np.flatnonzero(y==lab);ids=rng.choice(ids,min(1800,len(ids)),replace=False)
                ax.scatter(coord[ids,0],coord[ids,1],c=col,marker=marker,s=10,alpha=.35,label=name,rasterized=True)
            ax.plot(coord[chain,0],coord[chain,1],'-o',c='#454B52',lw=1,ms=3,label='PC1 순서 대표점 연결')
            ax.set_title(REP_NAMES[rep]+' · 스푸프 대표점 연결',loc='left',fontweight='bold')
            ax.set_xlabel(f'스푸프 PC1 ({frac[0]*100:.1f}%)');ax.set_ylabel(f'스푸프 PC2 ({frac[1]*100:.1f}%)')
            ax.set_aspect('equal',adjustable='datalim');ax.legend(frameon=False,fontsize=8)
            ax=axes[j,1]
            for curve,col,label in zip(curves,['#B45E8B','#3276A8'],['가장 가까이 접근한 경로','전형적인 경로']):
                ax.plot(curve['t']*100,curve['strict'],c=col,label=label)
            ax.axhline(1,c='#555555',ls='--',lw=1,label='정상 영역 경계')
            ax.axvspan(0,10,color='#eeeeee');ax.axvspan(90,100,color='#eeeeee')
            ax.set_yscale('log');ax.set_xlabel('긴 축 양끝 사이의 구면 경로 진행률 (%)')
            ax.set_ylabel('정상까지 거리 / 보수적 정상 반경')
            ax.set_title('전체 공간에서의 정상 접근 거리',loc='left',fontweight='bold');ax.legend(frameon=False,fontsize=8)
        for ax in axes.ravel():ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.13)
        fig.suptitle(f'{alias} · 연결 방향과 정상 접근',fontsize=17,fontweight='bold')
        fig.text(.02,.01,'왼쪽: 실제 스푸프 24개 대표점을 PC1 순서로 연결한 투영(인접성 보장 없음)\n오른쪽: 별도 긴 축 양끝 경로 600개 중 예시 · 잔여방향은 코사인 성분을 제거한 공간',fontsize=10)
        fig.tight_layout(rect=[0,.055,1,.95]);fig.savefig(OUT/f'{alias}_connected_paths.png',dpi=170,facecolor='white');plt.close(fig)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
