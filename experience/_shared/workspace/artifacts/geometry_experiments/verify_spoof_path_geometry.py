"""Independent numerical checks and all-observed-spoof sensitivity scan."""
from scan_spoof_path_support import *
from scipy.optimize import minimize_scalar


def main():
    dest=OUT/'independent_support'
    checks=[];metrics=[];all_hits=[]
    for i,alias in enumerate(['T','V','DF_pilot','21LA_pilot']):
        print('VERIFY',alias,flush=True)
        rows,x,s,u=load_view(alias);y=rows.y.to_numpy()
        for rep,f in [('unit_z',x),('residual_u',u)]:
            e=pd.read_parquet(dest/f'{alias}_{rep}_edges.parquet')
            sf,bf=f[y==0],f[y==1]
            # Superset pruning is exact: if bona loses to the fixed subset of
            # spoof support, it also loses to the full observed spoof set.
            candidates=np.flatnonzero(e.grid_any_bona_nearer.to_numpy())
            e['all_spoof_grid_any_bona_nearer']=False
            e['all_spoof_grid_bona_nearer_fraction']=0.
            if len(candidates):
                ee=e.iloc[candidates];aa,bb=f[ee.a_row],f[ee.b_row]
                w=np.arccos(np.clip(np.sum(aa*bb,1),-1,1));t=np.linspace(.1,.9,9)
                pts=(np.sin((1-t[None,:,None])*w[:,None,None])*aa[:,None,:]
                     +np.sin(t[None,:,None]*w[:,None,None])*bb[:,None,:])/np.sin(w)[:,None,None]
                scores=nearest_scores(pts.reshape(-1,160),bf,sf).reshape(len(ee),9,2)
                hit=scores[:,:,0]>scores[:,:,1]
                e.loc[candidates,'all_spoof_grid_any_bona_nearer']=hit.any(1)
                e.loc[candidates,'all_spoof_grid_bona_nearer_fraction']=hit.mean(1)
            e.to_parquet(dest/f'{alias}_{rep}_all_spoof_edges.parquet',index=False)
            for kind,g in e.groupby('edge_type',sort=False):
                metrics.append({'dataset':alias,'representation':rep,'edge_type':kind,'edges':len(g),
                    'all_spoof_grid_bona_nearer_hits':int(g.all_spoof_grid_any_bona_nearer.sum()),
                    'all_spoof_grid_bona_nearer_percent':float(100*g.all_spoof_grid_any_bona_nearer.mean()),
                    'balanced_grid_any_bona_nearer_percent':float(100*g.grid_any_bona_nearer.mean()),
                    'balanced_grid_new_entry_percent':float(100*g.grid_new_bona_nearer.mean())})
            for _,row in e[e.all_spoof_grid_any_bona_nearer].iterrows():
                all_hits.append({'dataset':alias,'representation':rep,'edge_type':row.edge_type,'a_row':int(row.a_row),'b_row':int(row.b_row),
                                 'score_delta':float(row.score_delta),'same_attack':bool(row.same_attack_id),
                                 'angle_deg':float(row.angle_deg)})
            # Direct Euclidean distances and 1-D optimization do not reuse the
            # matrix-based closed-form path-distance implementation.
            bd,_=NearestNeighbors(n_neighbors=6,algorithm='brute',n_jobs=1).fit(bf).kneighbors(bf)
            sd,_=NearestNeighbors(n_neighbors=1,algorithm='brute',n_jobs=1).fit(sf).kneighbors(bf)
            radius=np.maximum(np.minimum(bd[:,5],.5*sd[:,0]),1e-12)
            for ix in [int(e.arc_strict_ratio.idxmin()),len(e)//2]:
                rr=e.iloc[ix];ai,bi=f[int(rr.a_row)],f[int(rr.b_row)]
                bn=f[int(rr.arc_strict_nearest_bona_row)]
                bona_global=np.flatnonzero(y==1)
                k=int(np.flatnonzero(bona_global==int(rr.arc_strict_nearest_bona_row))[0])
                w=np.arccos(np.clip(ai@bi,-1,1))
                def objective(t):
                    p=(np.sin((1-t)*w)*ai+np.sin(t*w)*bi)/np.sin(w)
                    return np.linalg.norm(p-bn)/radius[k]
                opt=minimize_scalar(objective,bounds=(.1,.9),method='bounded',options={'xatol':1e-12})
                alt=min(objective(.1),objective(.9),opt.fun)
                error=abs(alt-float(rr.arc_strict_ratio))
                assert error<1e-6,(alias,rep,error)
                direct_spoof=np.min(np.linalg.norm(sf-bn,axis=1))
                assert abs(direct_spoof-sd[k,0])<1e-8
                checks.append({'dataset':alias,'representation':rep,'edge_row':ix,'arc_ratio_error':float(error)})
            # Reconcile exact count denominators, and the stricter/superset nesting.
            assert e.all_spoof_grid_any_bona_nearer.sum()<=e.grid_any_bona_nearer.sum()
    frame=pd.DataFrame(metrics)
    frame.to_csv(dest/'all_observed_spoof_metrics.csv',index=False)
    pd.DataFrame(all_hits).to_csv(dest/'all_observed_spoof_hit_examples.csv',index=False)
    save_json(ROOT/'final_verification.json',{'status':'pass','direct_checks':checks,
        'max_arc_ratio_error':max(c['arc_ratio_error'] for c in checks),'edge_observations':int(frame.edges.sum()),
        'all_observed_spoof_grid_hits':int(frame.all_spoof_grid_bona_nearer_hits.sum()),
        'scope':'9-point all-observed nearest-label scan; conservative arc intersection exact on the continuous interior; shared endpoints and multiple edge panels are dependent'})
    print(frame[['dataset','representation','edge_type','all_spoof_grid_bona_nearer_hits','all_spoof_grid_bona_nearer_percent']].to_string(index=False))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
