"""Class counts around complete cached spoof arcs, including empty space.

Every reference point contributes an analytically calculated interval on a
short spherical arc. An event sweep integrates class counts and support states
over its interior. Endpoints are excluded from reference counts. No model/GPU.
"""
from refine_spoof_path_neighborhood import *

DENSITY_ROOT = ROOT / 'path_density_v1'
DATASETS = ['T', 'V', 'DF_pilot', '21LA_pilot']
STATES = ['empty','few','bona','spoof','mixed']
STATE_LABELS = {'empty':'둘 다 없음','few':'희박','bona':'보나 우세','spoof':'스푸프 우세','mixed':'혼재'}
STATE_COLORS = {'empty':'#D9DDE1','few':'#A7AEB6','bona':'#3276A8','spoof':'#D38B32','mixed':'#98649C'}


def ball_intervals(A,B,w,radius,exclude):
    """Return each reference's intersection with .1-.9 arc as fractions."""
    threshold=1-radius*radius/2
    assert threshold>0, 'Radius must be smaller than a hemisphere chord'
    amplitude=np.hypot(A,B)
    valid=amplitude>=threshold
    phase=np.arctan2(B,A)
    half=np.arccos(np.clip(threshold/np.maximum(amplitude,1e-15),-1,1))
    lo=np.ones(len(A));hi=np.zeros(len(A))
    for shift in [-2*np.pi,0,2*np.pi]:
        left=np.maximum(.1*w,phase+shift-half)
        right=np.minimum(.9*w,phase+shift+half)
        keep=valid&(right-left>1e-12)
        lo[keep]=left[keep]/w;hi[keep]=right[keep]/w
    lo[exclude]=1;hi[exclude]=0
    return lo,hi


def sweep(lo,hi,y,n_bona,n_spoof):
    keep=hi>lo
    labs=y[keep];left=lo[keep];right=hi[keep]
    events=np.r_[.1,.9,left,right]
    points,inverse=np.unique(events,return_inverse=True)
    n=len(left)
    delta_b=np.bincount(inverse,weights=np.r_[0,0,(labs==1).astype(int),-(labs==1).astype(int)],minlength=len(points))
    delta_s=np.bincount(inverse,weights=np.r_[0,0,(labs==0).astype(int),-(labs==0).astype(int)],minlength=len(points))
    cb=np.rint(np.cumsum(delta_b)[:-1]).astype(int)
    cs=np.rint(np.cumsum(delta_s)[:-1]).astype(int)
    assert cb.min()>=0 and cs.min()>=0
    weight=np.diff(points)/.8
    assert abs(weight.sum()-1)<1e-12
    rb,rs=cb/n_bona,cs/n_spoof
    equivalent=n_bona*(rb+rs)
    out={'tube_unique_bona':int((labs==1).sum()),'tube_unique_spoof':int((labs==0).sum()),
         'mean_neighbor_bona':float(weight@cb),'mean_neighbor_spoof':float(weight@cs),
         'mean_bona_per1000':float(1000*(weight@rb)),
         'mean_spoof_per1000':float(1000*(weight@rs)),
         'any_empty_interval':bool(np.any((cb+cs)==0)),
         'whole_interior_empty':bool(np.all((cb+cs)==0))}
    for minimum in [1,5,10]:
        state=np.full(len(cb),'mixed',dtype='<U5')
        state[rb>=2*rs]='bona'
        state[rs>=2*rb]='spoof'
        state[equivalent<minimum]='few'
        state[cb+cs==0]='empty'
        for name in STATES:out[f't{minimum}_{name}_fraction']=float(weight[state==name].sum())
        assert abs(sum(out[f't{minimum}_{v}_fraction'] for v in STATES)-1)<1e-10
    return out


def direct_verify(features,lo,hi,ai,bi,radius,exclude,y):
    w=np.arccos(np.clip(ai@bi,-1,1));records=[]
    for t in [.137,.333,.517,.731,.887]:
        point=(np.sin((1-t)*w)*ai+np.sin(t*w)*bi)/np.sin(w)
        mask=np.linalg.norm(features-point,axis=1)<=radius
        mask[exclude]=False
        interval=(lo<=t)&(hi>=t)
        assert np.array_equal(mask,interval),int(np.sum(mask!=interval))
        records.append({'t':t,'bona':int(np.sum(mask&(y==1))),'spoof':int(np.sum(mask&(y==0)))})
    return records


def synthetic_verify():
    a=unit(np.array([[1.,-1.,0.]]))[0];b=unit(np.array([[1.,1.,0.]]))[0]
    f=np.array([a,b,[1.,0.,0.],[0.,0.,1.]])
    y=np.array([0,0,0,1]);w=np.arccos(a@b)
    v=(b-(a@b)*a)/np.sin(w)
    lo,hi=ball_intervals(f@a,f@v,w,.1,[0,1])
    out=sweep(lo,hi,y,1,1)
    assert out['t1_empty_fraction']>.8 and out['t1_spoof_fraction']>0 and out['tube_unique_bona']==0
    assert out['tube_unique_spoof']==1
    direct_verify(f,lo,hi,a,b,.1,[0,1],y)
    # All references outside the arc tube must stay empty, with no nearest label.
    lo,hi=ball_intervals(np.array([0.]),np.array([0.]),w,.1,[])
    assert sweep(lo,hi,np.array([1]),1,1)['t5_empty_fraction']==1
    return {'synthetic_gap_and_empty':'pass'}


def main():
    DENSITY_ROOT.mkdir(exist_ok=False)
    protocol={'created_utc':datetime.now(timezone.utc).isoformat(),'datasets':DATASETS,
        'previous_edges':'reuse all five previous edge panels, no new endpoint selection',
        'representations':['unit_z','residual_u'],'interior_arc_fraction':[.1,.9],
        'radius_calibration':'Each dataset/representation: median distance to 5th other point in a fixed balanced subset (up to 2000/class); same radius for both classes and every path location.',
        'radius_multipliers':[.5,1.,2.],
        'reference_counts':'All cached samples of both classes; the two endpoint spoof samples excluded per path.',
        'class_comparison':'counts divided by total eligible count of that class; also report raw counts',
        'few_definition':'nonempty but equivalent balanced reference count N_bona*(count_bona/N_bona + count_spoof/(N_spoof-2)) < 5; sensitivity 1 and 10',
        'dominance_definition':'at least 5 equivalent observations and class-normalized local mass at least twice the other class; otherwise mixed',
        'empty_definition':'zero observed samples of either class inside the same finite-radius ball',
        'integration':'analytic ball-arc intervals and exact event sweep, no grid for path fractions',
        'aggregation':'equal weight per edge; uniform arc-length within each interior, dependent endpoints/panels',
        'scope':'descriptive finite-sample local mass; no population density, posterior or true manifold claim',
        'script_sha256':digest(__file__)}
    save_json(DENSITY_ROOT/'protocol.json',protocol)
    verification=synthetic_verify();verification['cache_checks']=[validate_run(a,verify_hashes=True) for a in ['T','DF_pilot','21LA_eval']]
    records=[];radii=[];verification['direct_checks']=[];curves=[];source_edges=[]
    start=time.time()
    for di,alias in enumerate(DATASETS):
        print('DENSITY',alias,flush=True)
        rows,x,s,u=load_view(alias);y=rows.y.to_numpy();nb=int((y==1).sum());ns=int((y==0).sum())
        for rep,features in [('unit_z',x),('residual_u',u)]:
            edgefile=OUT/'continuous_labels'/f'{alias}_{rep}_edges.parquet'
            e=pd.read_parquet(edgefile)
            source_edges.append({'dataset':alias,'representation':rep,'file':str(edgefile),'sha256':digest(edgefile)})
            rng=np.random.default_rng(SEED+di*17)
            count=min(2000,nb,ns)
            calib=np.r_[rng.choice(np.flatnonzero(y==1),count,replace=False),rng.choice(np.flatnonzero(y==0),count,replace=False)]
            distances,_=NearestNeighbors(n_neighbors=6,algorithm='brute',n_jobs=1).fit(features[calib]).kneighbors(features[calib])
            radius=float(np.median(distances[:,5]));assert radius>1e-8 and radius*2<np.sqrt(2)
            radii.append({'dataset':alias,'representation':rep,'calibration_count_per_class':count,'base_radius':radius,
                          'normal_5nn_median_in_balanced_pool':float(np.median(distances[:count,5])),
                          'spoof_5nn_median_in_balanced_pool':float(np.median(distances[count:,5]))})
            output=[];curve_choices={}
            for typ,g in e.groupby('edge_type',sort=False):
                # Typical angle within each panel, not selected by its density.
                curve_choices[int((g.angle_deg-g.angle_deg.median()).abs().idxmin())]=typ
            print('SPACE',rep,'radius',round(radius,5),'edges',len(e),flush=True)
            for batch in range(0,len(e),48):
                ee=e.iloc[batch:batch+48]
                aa,bb=features[ee.a_row],features[ee.b_row]
                dot=np.clip(np.sum(aa*bb,axis=1),-1,1);w=np.arccos(dot)
                vv=(bb-dot[:,None]*aa)/np.maximum(np.sin(w)[:,None],1e-14)
                AA,BB=aa@features.T,vv@features.T
                for j,(_,er) in enumerate(ee.iterrows()):
                    idx=batch+j
                    for mult in [.5,1.,2.]:
                        rr=radius*mult
                        lo,hi=ball_intervals(AA[j],BB[j],w[j],rr,[int(er.a_row),int(er.b_row)])
                        metrics=sweep(lo,hi,y,nb,ns-2)
                        metrics.update(dataset=alias,representation=rep,edge_type=er.edge_type,
                                       edge_row=idx,radius_multiplier=mult,radius=rr,
                                       a_row=int(er.a_row),b_row=int(er.b_row),angle_deg=float(er.angle_deg),
                                       source_nearest_label_crossing=bool(er.continuous_all_spoof_bona_nearer))
                        output.append(metrics)
                        if idx==0 and mult==1:
                            tests=direct_verify(features,lo,hi,aa[j],bb[j],rr,[int(er.a_row),int(er.b_row)],y)
                            verification['direct_checks'].append({'dataset':alias,'representation':rep,'counts':tests})
                        if mult==1 and idx in curve_choices:
                            for t in np.linspace(.1,.9,81):
                                mask=(lo<=t)&(hi>=t)
                                cb=int(np.sum(mask&(y==1)));cs=int(np.sum(mask&(y==0)))
                                curves.append({'dataset':alias,'representation':rep,'edge_type':er.edge_type,'edge_row':idx,'t':float(t),
                                               'bona_count':cb,'spoof_count':cs,'bona_per1000':1000*cb/nb,'spoof_per1000':1000*cs/(ns-2)})
            frame=pd.DataFrame(output)
            frame.to_parquet(DENSITY_ROOT/f'{alias}_{rep}_edge_density.parquet',index=False)
            records.extend(output)
        save_json(DENSITY_ROOT/'progress.json',{'completed_dataset':alias,'elapsed_seconds':time.time()-start})
    all_results=pd.DataFrame(records)
    measure_cols=[f't{t}_{v}_fraction' for t in [1,5,10] for v in STATES]+['mean_neighbor_bona','mean_neighbor_spoof','mean_bona_per1000','mean_spoof_per1000','tube_unique_bona','tube_unique_spoof','any_empty_interval','whole_interior_empty']
    agg=all_results.groupby(['dataset','representation','edge_type','radius_multiplier'])[measure_cols].mean().reset_index()
    counts=all_results.groupby(['dataset','representation','edge_type','radius_multiplier']).size().reset_index(name='edges')
    agg=agg.merge(counts,validate='one_to_one')
    agg.to_csv(DENSITY_ROOT/'density_metrics.csv',index=False)
    pd.DataFrame(radii).to_csv(DENSITY_ROOT/'radii.csv',index=False)
    pd.DataFrame(curves).to_csv(DENSITY_ROOT/'typical_path_profiles.csv',index=False)
    save_json(DENSITY_ROOT/'input_edges.json',source_edges)
    verification['status']='pass';verification['edge_radius_observations']=len(all_results)
    verification['state_fraction_max_sum_error']=float(np.max(np.abs(all_results[[f't5_{v}_fraction' for v in STATES]].sum(axis=1)-1)))
    save_json(DENSITY_ROOT/'verification.json',verification)
    save_json(DENSITY_ROOT/'summary.json',{'status':'complete_exploratory','elapsed_seconds':time.time()-start,'radii':radii,'metrics':agg.to_dict('records')})
    render_density(agg)
    (DENSITY_ROOT/'complete').write_text('0\n')
    print('DENSITY COMPLETE',round(time.time()-start,1),flush=True)


def render_density(frame):
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':10})
    kinds=['local_1','same_score_local','random','pc1_extremes']
    fig,axes=plt.subplots(2,3,figsize=(15,8),sharex=True,sharey=True)
    for row,ds in enumerate(['DF_pilot','21LA_pilot']):
        for col,mult in enumerate([.5,1.,2.]):
            ax=axes[row,col]
            g=frame[(frame.dataset==ds)&(frame.representation=='residual_u')&(frame.radius_multiplier==mult)].set_index('edge_type')
            left=np.zeros(len(kinds))
            for state in ['spoof','bona','mixed','few','empty']:
                vals=np.array([g.loc[k,f't5_{state}_fraction']*100 for k in kinds])
                ax.barh(np.arange(len(kinds)),vals,left=left,color=STATE_COLORS[state],label=STATE_LABELS[state],height=.63)
                for k,(v,l) in enumerate(zip(vals,left)):
                    if v>=12:ax.text(l+v/2,k,f'{v:.0f}%',ha='center',va='center',fontsize=9,color='white' if state in ['spoof','bona','mixed'] else '#30363D')
                left+=vals
            ax.set_title(f'{ds} · 반경 {mult:g}배',loc='left',fontweight='bold')
            ax.set_yticks(np.arange(len(kinds)),[TYPE_NAMES[k] for k in kinds]);ax.invert_yaxis()
            ax.set_xlim(0,100);ax.spines[['top','right','left']].set_visible(False)
            ax.set_xlabel('경로 내부 길이의 평균 비율 (%)')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=5,bbox_to_anchor=(.5,.055),frameon=False)
    fig.suptitle('스푸프 연결 경로 주변에 실제 표본이 얼마나 있는가?',fontsize=17,fontweight='bold')
    fig.text(.02,.015,'잔여방향 공간 · 경로 양끝 10% 제외 · 끝점 스푸프 두 개는 집계 제외\n표본 수의 클래스 차이 보정 · 희박: 균형 표본 환산 5개 미만 · 빈 구간: 두 클래스 관찰 표본 모두 0개',fontsize=10)
    fig.tight_layout(rect=[0,.13,1,.94]);fig.savefig(DENSITY_ROOT/'density_support.png',dpi=170,facecolor='white');plt.close(fig)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
