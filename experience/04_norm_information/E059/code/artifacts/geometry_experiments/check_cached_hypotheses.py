"""Cache-only follow-up diagnostics. Never fits a detector or runs audio inference.

Hypothesis numbers refer to the Korean chat on 2026-11-06. Original experiment
code, caches, folds, models and the declined CV4 follow-up remain unchanged.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
from frozen_common import panel, DEFAULT_ROOT, CHECKPOINT, CAL_A, CAL_B, save, metric, interval
from frozen_heads import Head, oc
from frozen_probes import Probe, inputs, predict
from frozen_r23_cache import load as load_r23, CATALOG
from frozen_fast_bootstrap import fast_eers
from audit_cached_core_results import independent_eer, independent_auc

ROOT=Path(__file__).resolve().parents[2]
ORIGINAL=DEFAULT_ROOT/'local_analysis/frozen_all_v1'
OUT=DEFAULT_ROOT/'local_analysis/cache_hypotheses_20261106_v2'
DATASETS=['DF_pilot','21LA_pilot']
CONDITIONS=['R2-00','R3-noise20','R3-noise10','R3-Opus32k_recompression']
VARIANTS=['scalar','direction_linear','direction_mlp','direction_h']
SEEDS=[101,102,103]
DRAWS=2000
START=time.monotonic()
torch.set_num_threads(2)
torch.set_default_dtype(torch.float64)

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8388608), b''): h.update(b)
    return h.hexdigest()

def progress(stage,**kw):
    save(OUT/'progress.json',dict(stage=stage,elapsed_seconds=time.monotonic()-START,**kw))
    print(stage,kw,flush=True)

def stats(a):
    a=np.asarray(a);a=a[np.isfinite(a)]
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),
                p05=float(np.quantile(a,.05)),p95=float(np.quantile(a,.95)),max=float(a.max()))

def softplus_slope(a):
    # Match PyTorch's default threshold=20 linear branch, not just log(1+exp(a)).
    return np.where(a>20,1.,expit(a))

def align(d,reference):
    pos=pd.Index(d.rows.utterance_id).get_indexer(reference.rows.utterance_id)
    assert (pos>=0).all() and len(set(pos))==len(reference.rows)
    rows=d.rows.iloc[pos].reset_index(drop=True)
    assert rows.utterance_id.tolist()==reference.rows.utterance_id.tolist()
    assert np.array_equal(rows.y,reference.rows.y)
    assert np.allclose(d.centroid,reference.centroid,rtol=0,atol=1e-12)
    return type(d)(d.z[pos],d.h[pos],rows,d.centroid,d.provenance)

def features(d,g=None):
    g=d.geometry() if g is None else g; h=d.h.astype(float)
    return dict(s=g['cosine'],n=np.log(g['norm']),valid=g['valid'].astype(float),
                u=g['direction'],h=h,hstat=np.c_[h.mean(1),np.log(np.sqrt(h.var(1))+1e-12)])

def cluster_draws(groups):
    ug,codes=np.unique(groups,return_inverse=True)
    counts=np.random.default_rng(20261106).multinomial(len(ug),np.full(len(ug),1/len(ug)),size=DRAWS)
    return ug,codes,counts

def loss_rows(y,logit):
    p=np.clip(expit(logit),1e-7,1-1e-7)
    return -y*np.log(p)-(1-y)*np.log1p(-p)

def boot_ll(y,logit,codes,counts):
    losses=loss_rows(y,logit);result=np.zeros(len(counts))
    for label in [0,1]:
        mask=y==label;n=np.bincount(codes[mask],minlength=counts.shape[1])
        a=np.bincount(codes[mask],weights=losses[mask],minlength=counts.shape[1])
        result+=.5*(counts@a)/(counts@n)
    return result

def self_checks():
    rng=np.random.default_rng(812);c=rng.normal(size=160);c/=np.linalg.norm(c)
    z=rng.normal(size=(24,160));y=np.tile([0,1],12);zt=torch.tensor(z,requires_grad=True)
    s=torch.nn.functional.normalize(zt,dim=-1)@torch.tensor(c)
    val=oc(s,torch.tensor(y));val.backward()
    x=z/np.linalg.norm(z,axis=1,keepdims=True);ss=x@c
    ds=np.where(y==1,-20*softplus_slope(20*(.9-ss))*5,20*softplus_slope(20*(ss-.2))*(5/9))
    analytic=ds[:,None]*(c-ss[:,None]*x)/np.linalg.norm(z,axis=1,keepdims=True)/len(y)
    error=float(np.max(np.abs(analytic-zt.grad.numpy())));assert error<1e-12
    x1=rng.normal(size=(24,160));x1/=np.linalg.norm(x1,axis=1,keepdims=True)
    r0=x-ss[:,None]*c;r1=x1-(x1@c)[:,None]*c
    a=np.linalg.norm(r0,axis=1);b=np.linalg.norm(r1,axis=1)
    identity=(a-b)**2+a*b*np.sum((r0/a[:,None]-r1/b[:,None])**2,axis=1)
    geometric_error=float(np.max(np.abs(identity-np.sum((r1-r0)**2,axis=1))))
    assert geometric_error<1e-12
    save(OUT/'numerical_self_checks.json',dict(status='pass',analytic_OC_gradient_max_error=error,
         residual_chord_identity_max_error=geometric_error,actual_audio_experiment=False))

def prepare():
    OUT.mkdir(parents=True,exist_ok=False)
    decision=json.loads((ORIGINAL/'CV4_coverage_candidate_v1/execution_decision.json').read_text(encoding='utf8'))
    assert decision['status']=='excluded_by_user' and not decision['fits_launched']
    protocol=dict(version='cache-hypotheses-fast-v2',registered_KST=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
        checkpoint=CHECKPOINT,datasets=['T','V']+DATASETS,conditions=CONDITIONS,readout_variants=VARIANTS,
        seeds=SEEDS,bootstrap_draws=DRAWS,bootstrap_seed=20261106,
        new_detector_or_head_fits=0,new_audio_forwards=0,CV4_excluded=True,
        H3='Class-balanced loss and gradients at clean cached vectors; gradients of existing heads; OC azimuth invariance.',
        H4='Fixed radius bins; log chord versus log native radius and log full-vector change with class/source strata; speaker-cluster intervals.',
        H5='Recover fixed LayerNorm affine coefficients from T cache only; verify reconstruction on V and both pilots; decompose saved h readouts without refitting.',
        H1='Saved native-trained held-out-speaker scalar/direction probes applied to existing paired perturbation caches; compare advantage retention.',
        radius_bin_edges=[0,.05,.1,.2,.4,.6,.8,1.000001],
        confidence_scope='Exploratory individual 95% intervals, conditional on saved fits; no multiple-comparison or refit uncertainty claim.',
        limitations=['Existing DF5x3 source-class coverage limitation retained.','DF/LA shared speakers and unresolved source families.',
                     'No gain from a new loss, new gate, or nuisance-removal model is tested.',
                     'Fixed-layer coefficient recovery is a numerical identity measurement, not a supervised classifier fit.',
                     'Large residual-angle change can be geometric amplification without float precision error.'])
    save(OUT/'protocol.json',protocol)
    sources={Path(__file__).resolve(),CATALOG,DEFAULT_ROOT/'catalog.json',
             ORIGINAL/'CV4_coverage_candidate_v1/execution_decision.json'}
    for name in ['frozen_common.py','frozen_heads.py','frozen_probes.py','geometry_cache.py','frozen_r23_cache.py','frozen_fast_bootstrap.py']:
        sources.add(Path(__file__).parent/name)
    loaded={n:panel(n) for n in ['T','V']+DATASETS}
    for d in loaded.values():
        entry=d.provenance
        for f in entry['files']:
            p=DEFAULT_ROOT/entry['local_folder']/f['path']
            assert p.stat().st_size==f['bytes'] and digest(p)==f['sha256']
            sources.add(p)
        if 'analysis_manifest_sha256' in entry:
            sources.add(DEFAULT_ROOT/'analysis_manifests/frozen_data_sizes_v1'/f"{entry['analysis_alias']}.parquet")
        g=d.geometry();assert np.isfinite(d.z).all() and np.isfinite(d.h).all()
        assert np.max(np.abs(g['cosine']-d.rows.cosine_s))<1e-10
    rc=json.loads(CATALOG.read_text(encoding='utf8'))['entries']
    for ds in DATASETS:
        for cond in CONDITIONS:
            entry=rc[ds+'::'+cond]
            for f in entry['files']:
                p=DEFAULT_ROOT/entry['local_folder']/f['path']
                if p not in sources:
                    assert p.stat().st_size==f['bytes'] and digest(p)==f['sha256']
                    sources.add(p)
        folder=ORIGINAL/'P3_P4/B'/ds
        for name in ['fold_manifest.json','predictions.parquet','paired_effects.json','complete.json']:sources.add(folder/name)
        for variant in VARIANTS:
            for seed in SEEDS:
                for outer in range(5):sources.add(folder/'models'/variant/str(seed)/f'outer{outer}/refit/model.pt')
    for kind in ['static','metric','dynamic']:
        for seed in (SEEDS if kind=='dynamic' else [101]):
            sources.add(ORIGINAL/('P2' if kind=='metric' else 'P1')/str(seed)/kind/'head_state.pt')
    records=[dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(sources)]
    save(OUT/'inputs.json',records)
    self_checks();progress('inputs_verified',files=len(records))
    return loaded,records

def gradients(data):
    summaries=[];head_records=[];maximum_azimuth_error=0.
    for name,d in data.items():
        g=d.geometry();s=g['cosine'];y=d.rows.y.to_numpy();x=g['unit_z'];c=d.centroid
        arg=20*np.where(y==1,.9-s,s-.2)
        ds=np.where(y==1,-20*softplus_slope(arg),20*softplus_slope(arg))
        losses=np.where(arg>20,arg,np.logaddexp(0,arg))
        grad_unit=np.abs(ds)*np.sqrt(np.maximum(0,1-s*s))
        grad_raw=grad_unit/g['norm']
        # At fixed x/c, every direction perpendicular to both has zero OC derivative.
        rng=np.random.default_rng(20261106);ix=np.arange(min(256,len(y)));a=rng.normal(size=(len(ix),160))
        cc=c[None,:];rr=x[ix]-s[ix,None]*c;a-=np.sum(a*cc,axis=1,keepdims=True)*cc
        a-=np.sum(a*rr,axis=1,keepdims=True)/np.maximum(np.sum(rr*rr,axis=1,keepdims=True),1e-24)*rr
        a/=np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-24)
        gg=ds[ix,None]*(c-s[ix,None]*x[ix])/g['norm'][ix,None]
        ae=float(np.max(np.abs(np.sum(a*gg,axis=1))));maximum_azimuth_error=max(maximum_azimuth_error,ae)
        means={label:float(np.mean(grad_raw[y==label])) for label in [0,1]}
        lossmean={label:float(np.mean(losses[y==label])) for label in [0,1]}
        for label in [1,0]:
            m=y==label
            summaries.append(dict(dataset=name,y=label,n=int(m.sum()),loss_mean=float(losses[m].mean()),
                abs_score_gradient=stats(np.abs(ds[m])),unit_tangent_gradient=stats(grad_unit[m]),raw_z_gradient=stats(grad_raw[m]),
                fraction_abs_score_gradient_below_1e3=float(np.mean(np.abs(ds[m])<1e-3)),
                above_spoof_margin=int(np.sum(s[m]>.2)) if label==0 else None,
                spoof_fraction_of_balanced_loss=lossmean[0]/sum(lossmean.values()),
                spoof_fraction_of_balanced_raw_gradient_mass=means[0]/max(sum(means.values()),1e-300),
                OC_azimuth_derivative_max_abs=ae))
    d=data['T'];g=d.geometry();x=torch.as_tensor(g['unit_z']);y=d.rows.y.to_numpy()
    for kind in ['static','metric','dynamic']:
        for seed in (SEEDS if kind=='dynamic' else [101]):
            for state_name in (['identity','selected'] if kind=='dynamic' else ['identity']):
                head=Head(kind,d.centroid,seed)
                if state_name=='selected':
                    obj=torch.load(ORIGINAL/'P1'/str(seed)/kind/'head_state.pt',weights_only=True);head.load_state_dict(obj['state_dict'])
                vectors={};record=dict(kind=kind,seed=seed,state=state_name,at_clean_T_only=True)
                for label in [1,0]:
                    head.zero_grad(set_to_none=True);indices=np.flatnonzero(y==label)
                    for chunk in np.array_split(indices,max(1,int(np.ceil(len(indices)/2048)))):
                        score=head(x[chunk]);loss=(torch.nn.functional.softplus(20*(.9-score)) if label else torch.nn.functional.softplus(20*(score-.2))).sum()/len(indices)*.5
                        loss.backward()
                    vec=np.concatenate([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().numpy().ravel() for p in head.parameters()]);vectors[label]=vec
                    record['bona_gradient_L2' if label else 'spoof_gradient_L2']=float(np.linalg.norm(vec))
                gb,gs=vectors[1],vectors[0]
                record.update(spoof_fraction_sum_class_gradient_norms=float(np.linalg.norm(gs)/(np.linalg.norm(gb)+np.linalg.norm(gs)+1e-300)),
                    class_gradient_cosine=float(np.dot(gb,gs)/(np.linalg.norm(gb)*np.linalg.norm(gs)+1e-300)),combined_gradient_L2=float(np.linalg.norm(gb+gs)))
                head_records.append(record)
    save(OUT/'H3_OC_gradients.json',dict(status='complete',class_summaries=summaries,head_parameter_gradients=head_records,
         maximum_azimuth_derivative_error=maximum_azimuth_error,
         scope='Clean cached long views, no BPL or augmentation, no inference about the entire original training trajectory.'))
    progress('H3_complete')

def recover_ln(data):
    h=data['T'].h.astype(float);z=data['T'].z.astype(float);q=(h-h.mean(1,keepdims=True))/np.sqrt(h.var(1,keepdims=True)+1e-12)
    qm=q.mean(0);zm=z.mean(0);gamma=np.sum((q-qm)*(z-zm),axis=0)/np.sum((q-qm)**2,axis=0);beta=zm-gamma*qm
    assert np.min(np.abs(gamma))>1e-8
    checks=[]
    for name,d in data.items():
        hh=d.h.astype(float);zz=d.z.astype(float);mu=hh.mean(1);scale=np.sqrt(hh.var(1)+1e-12)
        qq=(hh-mu[:,None])/scale[:,None];predz=qq*gamma+beta
        reconstructed=mu[:,None]+scale[:,None]*(zz-beta)/gamma
        error=float(np.max(np.abs(reconstructed-hh)));zerr=float(np.max(np.abs(predz-zz)))
        assert error<1e-5 and zerr<1e-5,(name,error,zerr)
        checks.append(dict(dataset=name,rows=len(hh),z_affine_max_abs_error=zerr,h_reconstruction_max_abs_error=error,
                           h_reconstruction_relative_RMS=float(np.linalg.norm(reconstructed-hh)/np.linalg.norm(hh))))
    np.savez_compressed(OUT/'H5_T_recovered_LN_coefficients.npz',gamma=gamma,beta=beta)
    save(OUT/'H5_LN_reconstruction.json',dict(status='complete',estimated_from='T h/z pairs only, no class labels',
         affine_parameters=320,minimum_abs_gamma=float(np.min(np.abs(gamma))),checks=checks,
         identity='h_j=mean(h)+sqrt(var(h)+epsilon)*(z_j-beta_j)/gamma_j',
         interpretation='Reconstruction availability and multiplicative interaction; does not establish why a restricted probe improved.'))
    progress('H5_reconstruction_complete')
    return gamma,beta

def geometry_diagnostics(ds,datasets):
    native=datasets['R2-00'];g0=native.geometry();y=native.rows.y.to_numpy();rho=g0['tangent_norm'];x0=g0['unit_z'];c=native.centroid
    source=native.rows.source_corpus.fillna('unknown').astype(str).to_numpy() if 'source_corpus' in native.rows else np.full(len(y),'asvspoof')
    ug,codes,counts=cluster_draws(native.rows.dependency_group.to_numpy());table=[];regressions=[];max_identity=0.
    edges=np.array([0,.05,.1,.2,.4,.6,.8,1.000001]);bids=np.digitize(rho,edges[1:-1])
    strata=np.char.add(np.char.add(source.astype(str),'::'),y.astype(str));names=np.unique(strata)
    for cond,d in datasets.items():
        if cond=='R2-00':continue
        g=d.geometry();valid=g['valid']&g0['valid'];delta=g['unit_z']-x0;dx=np.linalg.norm(delta,axis=1)
        r0=x0-g0['cosine'][:,None]*c;r1=g['unit_z']-g['cosine'][:,None]*c
        dr=np.linalg.norm(r1-r0,axis=1);dot=np.sum(g0['direction']*g['direction'],axis=1)
        angle=np.degrees(np.arccos(np.clip(dot,-1,1)));chord=np.linalg.norm(g['direction']-g0['direction'],axis=1)
        identity=(g['tangent_norm']-rho)**2+rho*g['tangent_norm']*chord**2
        err=float(np.max(np.abs(identity[valid]-dr[valid]**2)));max_identity=max(max_identity,err);assert err<1e-10
        frame=native.rows[['utterance_id','y','dependency_group']].copy();frame['source_corpus']=source
        for key,val in dict(native_score=g0['cosine'],native_radius=rho,changed_radius=g['tangent_norm'],unit_z_shift=dx,
                            residual_shift=dr,direction_chord=chord,direction_angle=angle,score_change=g['cosine']-g0['cosine'],valid=valid).items():frame[key]=val
        frame.to_parquet(OUT/f'H4_{ds}_{cond}_rows.parquet',index=False)
        for label in [-1,0,1]:
            lm=np.ones(len(y),bool) if label<0 else y==label
            for bin_id in range(len(edges)-1):
                m=lm&valid&(bids==bin_id)
                if not m.any():continue
                table.append(dict(dataset=ds,condition=cond,y=label,radius_bin=bin_id,radius_low=edges[bin_id],radius_high=edges[bin_id+1],
                    n=int(m.sum()),groups=len(np.unique(codes[m])),mean_angle=float(angle[m].mean()),median_angle=float(np.median(angle[m])),
                    mean_unit_shift=float(dx[m].mean()),mean_residual_shift=float(dr[m].mean()),mean_abs_score_change=float(np.abs(g['cosine'][m]-g0['cosine'][m]).mean()),
                    fraction_angle_gt30_and_abs_score_change_lt001=float(np.mean((angle[m]>30)&(np.abs(g['cosine'][m]-g0['cosine'][m])<.01)))))
        m=valid&(dx>1e-10)&(chord>1e-10)&(rho>1e-6)
        X=np.c_[np.log(rho[m]),np.log(dx[m]),np.column_stack([strata[m]==a for a in names])];target=np.log(chord[m]);code=codes[m]
        coef=np.linalg.lstsq(X,target,rcond=None)[0];fitted=X@coef
        xx=np.stack([X[code==i].T@X[code==i] for i in range(len(ug))]);xy=np.stack([X[code==i].T@target[code==i] for i in range(len(ug))])
        bxx=np.einsum('bg,gij->bij',counts,xx);bxy=counts@xy
        bc=(np.linalg.pinv(bxx,rcond=1e-10)@bxy[:,:,None])[:,:,0]
        reg=dict(dataset=ds,condition=cond,n=int(m.sum()),groups=len(ug),radius_log_coefficient=float(coef[0]),radius_coefficient_CI95=interval(bc[:,0]),
            shift_log_coefficient=float(coef[1]),R_squared=float(1-np.sum((target-fitted)**2)/np.sum((target-target.mean())**2)),
            source_class_strata=names.tolist(),median_angle=float(np.median(angle[valid])),
            fraction_angle_gt30_and_abs_score_change_lt001=float(np.mean((angle[valid]>30)&(np.abs(g['cosine'][valid]-g0['cosine'][valid])<.01))))
        regressions.append(reg)
    # Roundoff diagnostic is separate from geometric amplification of real input changes.
    zz=native.z.astype(np.float32);xx=zz/np.linalg.norm(zz,axis=1,keepdims=True);ss=xx@c.astype(np.float32)
    rr=xx-ss[:,None]*c.astype(np.float32);rr/=np.maximum(np.linalg.norm(rr,axis=1,keepdims=True),1e-30)
    numeric_error=np.linalg.norm(rr-g0['direction'],axis=1)[g0['valid']]
    pd.DataFrame(table).to_csv(OUT/f'H4_{ds}_radius_bins.csv',index=False)
    save(OUT/f'H4_{ds}_summary.json',dict(regressions=regressions,native_radius=stats(rho),
        fraction_radius_below01=float(np.mean(rho<.1)),max_geometric_identity_error=max_identity,
        float32_vs_float64_direction_chord=stats(numeric_error),
        distinction='Log coefficient is descriptive geometry, not proof that normalization causes classification errors or that a reliability gate helps.'))
    progress('H4_dataset_complete',dataset=ds)

def readouts(ds,datasets,gamma,beta):
    base=datasets['R2-00'];rows=base.rows;y=rows.y.to_numpy();folder=ORIGINAL/'P3_P4/B'/ds
    lock=json.loads((folder/'fold_manifest.json').read_text(encoding='utf8'))
    assert lock['UID_order']==rows.utterance_id.tolist()
    cached=pd.read_parquet(folder/'predictions.parquet');assert cached.utterance_id.tolist()==rows.utterance_id.tolist()
    raw={name:features(d) for name,d in datasets.items()}
    scores={(v,s,cond):np.full(len(rows),np.nan) for v in VARIANTS for s in SEEDS for cond in CONDITIONS}
    hhat=np.zeros(len(rows));decomposed=np.zeros(len(rows));hvar=np.zeros(len(rows));hmu=np.zeros(len(rows));baseline_errors=[]
    hh=base.h.astype(float);zz=base.z.astype(float);mu=hh.mean(1);scale=np.sqrt(hh.var(1)+1e-12);q=(zz-beta)/gamma
    reconstructed=mu[:,None]+scale[:,None]*q
    # The output here reuses seed101 only to verify the exact algebraic decomposition.
    for outer in lock['folds']:
        train=np.asarray(outer['train_rows']);test=np.asarray(outer['test_rows'])
        assert not set(rows.iloc[train].dependency_group)&set(rows.iloc[test].dependency_group)
        for variant in VARIANTS:
            for seed in SEEDS:
                p=folder/'models'/variant/str(seed)/f"outer{outer['fold']}/refit/model.pt"
                state=torch.load(p,weights_only=False);assert state['variant']==variant and state['seed']==seed
                model=Probe(state['width'],variant,seed);model.load_state_dict(state['state_dict']);model.eval()
                scaler=state['scaler']
                for cond in CONDITIONS:scores[(variant,seed,cond)][test]=predict(model,inputs(raw[cond],test,variant,scaler))
                if variant=='direction_h' and seed==101:
                    xin=inputs(raw['R2-00'],test,variant,scaler);w=model.w.detach().numpy();base_log=predict(model,xin)-xin[:,163:]@w[160:]
                    hs=scaler['h'];hw=w[160:]/hs['std'];offset=float(np.dot(hs['mean'],hw))
                    term_mu=mu[test]*hw.sum();term_interaction=scale[test]*(q[test]@hw)
                    decomposed[test]=base_log+term_mu+term_interaction-offset
                    new_raw=dict(raw['R2-00'],h=reconstructed)
                    hhat[test]=predict(model,inputs(new_raw,test,variant,scaler))
                    # Functional intervention, not refitting or a causal layer ablation.
                    hvar[test]=base_log+term_mu+scale[train].mean()*(q[test]@hw)-offset
                    hmu[test]=base_log+mu[train].mean()*hw.sum()+term_interaction-offset
        progress('saved_readouts_fold',dataset=ds,fold=outer['fold'])
    records=[];maximum=0.
    for (variant,seed,cond),s in scores.items():
        assert np.isfinite(s).all()
        if cond=='R2-00':
            error=float(np.max(np.abs(s-cached[f'{variant}_{seed}'].to_numpy())));maximum=max(maximum,error);assert error<1e-8,(ds,variant,seed,error)
        rec=dict(dataset=ds,variant=variant,seed=seed,condition=cond,**metric(y,s,logit=True))
        assert abs(independent_eer(y,s)-rec['eer_percent'])<1e-9
        assert abs(independent_auc(y,s)-rec['auroc'])<1e-12;records.append(rec)
    exact=scores[('direction_h',101,'R2-00')]
    de=float(np.max(np.abs(decomposed-exact)));re=float(np.max(np.abs(hhat-exact)))
    assert de<1e-5 and re<1e-5,(ds,de,re)
    h5=dict(dataset=ds,seed=101,maximum_saved_native_logit_error=maximum,h_reconstructed_logit_max_error=re,
            h_decomposition_logit_max_error=de,
            decomposition='existing scalar+direction terms + mean(h)*sum(w_h/std_h) + scale(h)*q(z) dot (w_h/std_h) - scaler offset',
            metrics=[dict(case=case,**metric(y,s,logit=True)) for case,s in [('saved_h',exact),('reconstructed_h',hhat),
                ('scale_fixed_to_outer_train_mean',hvar),('mean_fixed_to_outer_train_mean',hmu)]],
            limitation='Input interventions of a fixed trained readout; their metric changes are not performance of refitted alternatives or proof of LayerNorm causality.')
    save(OUT/f'H5_{ds}_saved_readout_decomposition.json',h5)
    pd.DataFrame(records).to_csv(OUT/f'H1_{ds}_readout_metrics.csv',index=False)
    preds=rows[['utterance_id','y','dependency_group']].copy()
    for (v,s,c),a in scores.items():preds[f'{v}__{s}__{c}']=a
    preds.to_parquet(OUT/f'H1_{ds}_saved_readout_predictions.parquet',index=False)
    ug,codes,counts=cluster_draws(rows.dependency_group.to_numpy());bs={};bll={}
    for k,s in scores.items():
        bs[k]=fast_eers(y,s,codes,counts);bll[k]=boot_ll(y,s,codes,counts)
    effects=[]
    for variant in ['direction_linear','direction_mlp']:
        native_gain=np.mean([bs[('scalar',s,'R2-00')]-bs[(variant,s,'R2-00')] for s in SEEDS],axis=0)
        native_ll_gain=np.mean([bll[('scalar',s,'R2-00')]-bll[(variant,s,'R2-00')] for s in SEEDS],axis=0)
        for cond in CONDITIONS:
            gain=np.mean([bs[('scalar',s,cond)]-bs[(variant,s,cond)] for s in SEEDS],axis=0)
            llgain=np.mean([bll[('scalar',s,cond)]-bll[(variant,s,cond)] for s in SEEDS],axis=0)
            point_gain=float(np.mean([metric(y,scores[('scalar',s,cond)],logit=True)['eer_percent']-metric(y,scores[(variant,s,cond)],logit=True)['eer_percent'] for s in SEEDS]))
            point_ll=float(np.mean([metric(y,scores[('scalar',s,cond)],logit=True)['balanced_logloss']-metric(y,scores[(variant,s,cond)],logit=True)['balanced_logloss'] for s in SEEDS]))
            effects.append(dict(dataset=ds,variant=variant,condition=cond,EER_advantage_pp=point_gain,EER_advantage_CI95=interval(gain),
                change_of_EER_advantage_CI95=interval(gain-native_gain),LL_advantage=point_ll,LL_advantage_CI95=interval(llgain),
                change_of_LL_advantage_CI95=interval(llgain-native_ll_gain),draws=DRAWS,groups=len(ug)))
    save(OUT/f'H1_{ds}_advantage_retention.json',effects)
    progress('H1_H5_dataset_complete',dataset=ds,max_native_saved_logit_error=maximum)

def run():
    data,records=prepare();gradients(data);gamma,beta=recover_ln(data)
    for ds in DATASETS:
        datasets={c:align(load_r23(ds,c),data[ds]) for c in CONDITIONS}
        assert np.max(np.abs(datasets['R2-00'].z-data[ds].z))<1e-6
        assert np.max(np.abs(datasets['R2-00'].h-data[ds].h))<1e-6
        geometry_diagnostics(ds,datasets);readouts(ds,datasets,gamma,beta)
    for r in records:
        p=Path(r['path']);assert p.stat().st_size==r['bytes'] and digest(p)==r['sha256'],p
    save(OUT/'verification.json',dict(status='pass',input_files_rehashed_unchanged=len(records),numerical_self_checks_passed=True,
         readout_metric_rows_per_dataset=len(VARIANTS)*len(SEEDS)*len(CONDITIONS),
         EER_and_AUROC_independently_checked_for_every_readout_metric_row=True,
         all_saved_native_predictions_reproduced=True,original_models_and_folds_unchanged=True,
         original_EBM_retraining=False,new_head_fits=0,new_audio_inference=False,CV4_not_launched=True,
         limits='Individual exploratory intervals conditional on saved fits; no source-family/refit uncertainty or full new-method validation.'))
    save(OUT/'complete.json',dict(status='complete',finished_KST=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
         seconds=time.monotonic()-START,hypotheses_checked=['H3','H4','H5','H1_partial'],new_head_fits=0,new_audio_inference=0))
    progress('complete')

if __name__=='__main__':
    with threadpool_limits(2):run()
