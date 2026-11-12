"""Validate and summarize the locked diagnostic cohort, without causal overclaims."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from analyze_B_scores import bootstrap_mean_delta,save

def main(folder):
    folder=Path(folder);root=folder/'mechanism_2x2';out=folder/'analysis'
    selected=pd.read_parquet(root/'selected.parquet').sort_values('selection_index')
    protocol=json.loads((root/'protocol.json').read_text());complete=json.loads((root/'complete.json').read_text())
    assert hashlib.sha256((root/'selected.parquet').read_bytes()).hexdigest()==protocol['selected_sha256']
    assert hashlib.sha256((root/'paired_embeddings.npz').read_bytes()).hexdigest()==complete['embeddings_sha256']
    d=np.load(root/'paired_embeddings.npz');z=d['z'].astype('float64');c=d['centroids'].astype('float64')
    assert z.shape==(len(selected),2,2,160) and c.shape==(2,160)
    u=z/np.linalg.norm(z,axis=-1,keepdims=True);c=c/np.linalg.norm(c,axis=-1,keepdims=True)
    scores=np.einsum('nmvd,md->nmv',u,c);angles=np.degrees(np.arccos(np.clip(scores,-1,1)))
    assert np.max(np.abs(scores-d['scores']))<1e-6
    assert np.max(np.abs(angles-d['angles']))<.01
    long=selected.original_samples.to_numpy()>=64600
    assert np.array_equal(z[long,:,0],z[long,:,1])
    statistics=[];contrasts=[];basis=[];projection=[];directions=[]
    comparison=json.loads((out/'comparison.json').read_text())
    for m,model in enumerate(['B','legacy240']):
        tangent=u[:,m]-scores[:,m,:,None]*c[m]
        _,singular,vectors=np.linalg.svd(tangent.reshape(-1,160),full_matrices=False)
        v=vectors[:2]
        assert np.max(np.abs(v@c[m]))<1e-6
        xy=np.einsum('nvd,kd->nvk',u[:,m],v)
        coords=np.concatenate([xy,scores[:,m,:,None]],axis=-1)
        basis.append({'model':model,'tangent_top2_energy_percent':float(100*np.sum(singular[:2]**2)/np.sum(singular**2)),
                      'total_projected_unit_energy_percent':float(100*np.mean(np.sum(coords**2,axis=-1))),
                      'fit':'uncentered SVD of within-model tangent residuals, all768 selected cases x2 inputs; descriptive, not training PCA'})
        projection.append(coords)
        for view,name in enumerate(['native','wrap']):
            residual=tangent[:,view]/np.linalg.norm(tangent[:,view],axis=1,keepdims=True)
            in_group=(selected.phase=='21DF')&(selected.length_bin=='under2')
            normal=in_group&(selected.label==1);spoof=in_group&(selected.label==0)
            bad=normal&(scores[:,m,view]<comparison['dev'][model]['dev_threshold'])
            b=residual[normal].mean(0);s=residual[spoof].mean(0)
            row={'model':model,'input':name,'normal_N':int(normal.sum()),'spoof_N':int(spoof.sum()),
                 'normal_error_N':int(bad.sum()),'residual_mean_normal_spoof_cosine':float(b@s/(np.linalg.norm(b)*np.linalg.norm(s))),
                 'normal_R':float(np.linalg.norm(b)),'spoof_R':float(np.linalg.norm(s))}
            if bad.any():
                q=residual[bad].mean(0)
                row['wrong_normal_vs_spoof_residual_mean_cosine']=float(q@s/(np.linalg.norm(q)*np.linalg.norm(s)))
            directions.append(row)
        for (phase,label,binname),g in selected.groupby(['phase','label','length_bin'],sort=True):
            ids=g.selection_index.to_numpy();delta=angles[ids,m,0]-angles[ids,m,1]
            seed=int.from_bytes(hashlib.sha256(f'Bdiag|{model}|{phase}|{label}|{binname}'.encode()).digest()[:8],'big')
            ci=bootstrap_mean_delta(g,delta,seed,multiplier=1)
            row={'model':model,'phase':phase,'label':int(label),'length_bin':binname,'N':len(g),
                 'native_mean_angle':float(angles[ids,m,0].mean()),'wrap_mean_angle':float(angles[ids,m,1].mean()),
                 'native_mean_score':float(scores[ids,m,0].mean()),'wrap_mean_score':float(scores[ids,m,1].mean()),
                 'native_minus_wrap_mean_angle':float(delta.mean()),'mean_absolute_score_change':float(np.abs(scores[ids,m,0]-scores[ids,m,1]).mean()),
                 'angle_delta_CI95':ci['CI95'],'speakers':ci['speakers']}
            for view,name in enumerate(['native','wrap']):
                t=tangent[ids,view];norm=np.linalg.norm(t,axis=1);valid=norm>1e-5
                row[name+'_residual_R']=float(np.linalg.norm((t[valid]/norm[valid,None]).mean(0))) if valid.any() else None
                row[name+'_residual_N']=int(valid.sum())
                mean=u[ids,m,view].mean(0)
                row[name+'_normalised_mean_direction_angle']=float(np.degrees(np.arccos(np.clip((mean/np.linalg.norm(mean))@c[m],-1,1))))
            statistics.append(row)
    for (phase,label,binname),g in selected.groupby(['phase','label','length_bin'],sort=True):
        ids=g.selection_index.to_numpy()
        change=np.abs(scores[ids,0,0]-scores[ids,0,1])-np.abs(scores[ids,1,0]-scores[ids,1,1])
        seed=int.from_bytes(hashlib.sha256(f'Bdiaginteraction|{phase}|{label}|{binname}'.encode()).digest()[:8],'big')
        ci=bootstrap_mean_delta(g,change,seed,multiplier=1)
        contrasts.append({'phase':phase,'label':int(label),'length_bin':binname,'N':len(g),
                          'B_minus_legacy_absolute_sensitivity':float(change.mean()),'CI95':ci['CI95'],'speakers':ci['speakers']})
    np.savez_compressed(out/'diagnostic_projections.npz',coordinates=np.stack(projection,axis=1))
    save(out/'geometry_statistics.json',statistics);save(out/'geometry_interactions.json',contrasts);save(out/'projection_basis.json',basis)
    save(out/'geometry_directions.json',directions)
    save(out/'geometry_validation.json',{'status':'passed','N':len(selected),'metadata_only_locked_selection':True,
         'selected_sha256':protocol['selected_sha256'],'long_control_identical_embeddings':True,'noise_std':0.,
         'score_identity_max_error':float(np.max(np.abs(scores-d['scores']))),
         'angle_identity_max_error_degrees':float(np.max(np.abs(angles-d['angles']))),
         'interpretation':'diagnostic cohort; same-content policy intervention changes context length too; axes within each model only'})
    print(json.dumps({'N':len(selected),'DF_under2_normal':[r for r in statistics if r['phase']=='21DF' and r['label']==1 and r['length_bin']=='under2'],
        'DF_under2_interactions':[r for r in contrasts if r['phase']=='21DF' and r['length_bin']=='under2']}),flush=True)

if __name__=='__main__':main(sys.argv[1])
