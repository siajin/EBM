"""Exact continuous observed-label Voronoi intersection on a short sphere arc.

Center the arc parameter: p(d)=m*cos(d)+v*sin(d), |d|<pi/2. A bona b is
closer than EVERY observed spoof s iff (b-s).m + (b-s).v*tan(d)>0 for all s.
These are scalar linear inequalities; intersect their intervals. No grid is
needed. Chord nearest-label inequalities have the same affine half-plane form.
"""
from refine_spoof_path_neighborhood import *


def intersect(lo,hi,A,B,eps=1e-12):
    pos=B>eps;neg=B < -eps
    roots=np.divide(-A,B,out=np.zeros_like(A),where=np.abs(B)>eps)
    lower=np.max(np.where(pos,roots,-np.inf),axis=1)
    upper=np.min(np.where(neg,roots,np.inf),axis=1)
    valid=~np.any((np.abs(B)<=eps)&(A<=eps),axis=1)
    lower=np.maximum(lo,lower);upper=np.minimum(hi,upper)
    return lower,upper,valid&(lower<upper-1e-12)


def classify(a,b,bona,spoof):
    n=len(a);hit=np.zeros(n,dtype=bool);witnesses=[];candidate_total=0
    for j,(ai,bi) in enumerate(zip(a,b)):
        w=np.arccos(np.clip(ai@bi,-1,1))
        if w<1e-9:
            continue
        m=(ai+bi)/np.linalg.norm(ai+bi);v=(bi-ai)/np.linalg.norm(bi-ai)
        bm,bv=bona@m,bona@v
        half=np.cos(w/2);side=np.sin(w/2)
        A=np.column_stack([bm-half,bm-half])
        B=np.column_stack([bv+side,bv-side])
        qlo,qhi=-np.tan(.4*w),np.tan(.4*w)
        lo,hi,valid=intersect(qlo,qhi,A,B)
        candidates=np.flatnonzero(valid);candidate_total+=len(candidates)
        if not len(candidates):continue
        sm,sv=spoof@m,spoof@v
        for k in candidates:
            ll,hh,ok=intersect(lo[k],hi[k],(bm[k]-sm)[None,:],(bv[k]-sv)[None,:])
            if not ok[0]:continue
            q=(ll[0]+hh[0])/2;d=np.arctan(q)
            point=m*np.cos(d)+v*np.sin(d)
            gap=float(point@bona[k]-np.max(spoof@point))
            if gap>1e-10:
                hit[j]=True
                witnesses.append({'edge_row':j,'bona_local_row':int(k),'arc_fraction':float((d+w/2)/w),
                                  'cosine_gap_bona_over_all_spoof':gap,
                                  'interval_fraction_lo':float((np.arctan(ll[0])+w/2)/w),
                                  'interval_fraction_hi':float((np.arctan(hh[0])+w/2)/w)})
                break
    return hit,witnesses,candidate_total


def main():
    dest=OUT/'continuous_labels';dest.mkdir(exist_ok=False)
    save_json(dest/'protocol.json',{'UTC':datetime.now(timezone.utc).isoformat(),
        'criterion':'exists an interior point (.1,.9 arc fraction) nearer at least one observed bona than every observed spoof',
        'computation':'intersection of scalar half-plane intervals in tan(centered arc angle); exact for finite stored point clouds, numerical witness margin > 1e-10',
        'meaning':'observed nearest-label region, not a learned detector prediction or true audio manifold membership',
        'sha256':digest(__file__)})
    metrics=[];witnesses=[];candidate_counts=[]
    # A literal crossing and a literal non-crossing independently known in 3D.
    test_a=unit(np.array([[1.,-1.,0.]]));test_b=unit(np.array([[1.,1.,0.]]))
    good=unit(np.array([[1.,0.,0.]]));bad=unit(np.array([[0.,0.,1.]]))
    assert classify(test_a,test_b,good,np.concatenate([test_a,test_b]))[0][0]
    assert not classify(test_a,test_b,bad,np.concatenate([test_a,test_b]))[0][0]
    for alias in ['T','V','DF_pilot','21LA_pilot']:
        print('CONTINUOUS',alias,flush=True)
        rows,x,s,u=load_view(alias);y=rows.y.to_numpy()
        for rep,f in [('unit_z',x),('residual_u',u)]:
            e=pd.read_parquet(OUT/'independent_support'/f'{alias}_{rep}_all_spoof_edges.parquet')
            hit,ww,nc=classify(f[e.a_row],f[e.b_row],f[y==1],f[y==0])
            assert np.all(hit[e.all_spoof_grid_any_bona_nearer.to_numpy()]),'A grid witness must also be a continuous witness'
            e['continuous_all_spoof_bona_nearer']=hit
            e.to_parquet(dest/f'{alias}_{rep}_edges.parquet',index=False)
            candidate_counts.append({'dataset':alias,'representation':rep,'bona_edge_candidates':nc})
            for kind,g in e.groupby('edge_type',sort=False):
                metrics.append({'dataset':alias,'representation':rep,'edge_type':kind,'edges':len(g),
                    'continuous_bona_nearer_hits':int(g.continuous_all_spoof_bona_nearer.sum()),
                    'continuous_bona_nearer_percent':float(100*g.continuous_all_spoof_bona_nearer.mean())})
            for rec in ww:
                er=e.iloc[rec['edge_row']]
                rec.update(dataset=alias,representation=rep,edge_type=er.edge_type,a_row=int(er.a_row),b_row=int(er.b_row),
                           same_attack=bool(er.same_attack_id),score_delta=float(er.score_delta),angle_deg=float(er.angle_deg),
                           bona_row=int(np.flatnonzero(y==1)[rec['bona_local_row']]))
                witnesses.append(rec)
    pd.DataFrame(metrics).to_csv(dest/'metrics.csv',index=False)
    pd.DataFrame(witnesses).to_csv(dest/'witnesses.csv',index=False)
    save_json(dest/'summary.json',{'status':'complete_exploratory','metrics':metrics,'candidate_counts':candidate_counts,
                                 'witness_count':len(witnesses),'synthetic_crossing_and_noncrossing':'pass','all_grid_witnesses_recovered':True})
    (dest/'complete').write_text('0\n')
    print(pd.DataFrame(metrics).to_string(index=False));print('CONTINUOUS COMPLETE',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
