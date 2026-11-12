"""Independent equal-count nearest-support check along stored spoof arcs.

This complements conservative balls; it does not turn either criterion into a
population density, posterior, or guarantee about a generated audio waveform.
"""
from refine_spoof_path_neighborhood import *


def nearest_scores(points,bona,spoof):
    out=[]
    for start in range(0,len(points),1500):
        xx=points[start:start+1500]
        out.append(np.column_stack([(xx@bona.T).max(1),(xx@spoof.T).max(1)]))
    return np.concatenate(out)


def main():
    dest=OUT/'independent_support'
    dest.mkdir(exist_ok=False)
    save_json(dest/'protocol.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'grid':list(np.linspace(.1,.9,9)),'support':'all bona versus equal-size random spoof reference excluded from all endpoints; same fixed split as initial analysis',
        'new_entry':'both endpoints nearer spoof reference, at least one sampled interior point nearer bona',
        'scope':'descriptive nearest-support geometry, not posterior or true manifold crossing; finite grid may miss narrow crossings',
        'script_sha256':digest(__file__)})
    records=[]
    for i,alias in enumerate(['T','V','DF_pilot','21LA_pilot']):
        print('SUPPORT',alias,flush=True)
        rows,x,s,u=load_view(alias);y=rows.y.to_numpy()
        for rep,f in [('unit_z',x),('residual_u',u)]:
            e=pd.read_parquet(OUT/f'{alias}_{rep}_edges.parquet')
            spoof_ids=np.flatnonzero(y==0);bf=f[y==1]
            rng=np.random.default_rng(SEED+i*17)
            support_ids=spoof_ids[rng.permutation(len(spoof_ids))[:len(bf)]]
            assert not set(support_ids)&set(e.a_row) and not set(support_ids)&set(e.b_row)
            sf=f[support_ids]
            aa,bb=f[e.a_row],f[e.b_row]
            w=np.arccos(np.clip(np.sum(aa*bb,1),-1,1))
            t=np.linspace(.1,.9,9)
            points=(np.sin((1-t[None,:,None])*w[:,None,None])*aa[:,None,:]
                    +np.sin(t[None,:,None]*w[:,None,None])*bb[:,None,:])/np.sin(w)[:,None,None]
            sc=nearest_scores(points.reshape(-1,160),bf,sf).reshape(len(e),9,2)
            hit=sc[:,:,0]>sc[:,:,1]
            a_sc,b_sc=nearest_scores(aa,bf,sf),nearest_scores(bb,bf,sf)
            endpoints_spoof=(a_sc[:,0]<a_sc[:,1])&(b_sc[:,0]<b_sc[:,1])
            e['grid_any_bona_nearer']=hit.any(1)
            e['grid_bona_nearer_fraction']=hit.mean(1)
            e['endpoints_both_spoof_nearer']=endpoints_spoof
            e['grid_new_bona_nearer']=endpoints_spoof&hit.any(1)
            e.to_parquet(dest/f'{alias}_{rep}_edges.parquet',index=False)
            for kind,g in e.groupby('edge_type',sort=False):
                records.append({'dataset':alias,'representation':rep,'edge_type':kind,'edges':len(g),
                    'any_grid_bona_nearer_hits':int(g.grid_any_bona_nearer.sum()),
                    'any_grid_bona_nearer_percent':float(100*g.grid_any_bona_nearer.mean()),
                    'new_entry_hits':int(g.grid_new_bona_nearer.sum()),
                    'new_entry_percent_of_all':float(100*g.grid_new_bona_nearer.mean()),
                    'endpoints_both_spoof_nearer_edges':int(g.endpoints_both_spoof_nearer.sum()),
                    'mean_grid_bona_nearer_percent':float(100*g.grid_bona_nearer_fraction.mean())})
    pd.DataFrame(records).to_csv(dest/'metrics.csv',index=False)
    save_json(dest/'summary.json',{'status':'complete_exploratory','metrics':records})
    (dest/'complete').write_text('0\n')
    print('SUPPORT COMPLETE',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
