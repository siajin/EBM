"""Fit Train-only activity probabilities with original source priors retained."""
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.optimize import linprog,minimize
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import arguments
from positive_bpl import MatchedTrainSet
from dataset.ASVspoof2019_la_train import load_train_items_only
from activity_policy import activity_bins


def totals(code,x,n):return np.bincount(code,weights=x,minlength=n)


def errors(x,row,target,gl,gt,labels,length,bins,voice,q,k):
    r=totals(row,x,len(target));active=target>1e-14
    err_row=float(np.max(np.abs(r[active]-target[active])/target[active]))
    err_gl=0.
    if gl is not None:err_gl=float(np.max(np.abs(totals(gl,x,len(gt))-gt)/np.maximum(gt,1e-14)))
    c=totals(labels*k*6+length*6+bins,x,2*k*6).reshape(2,k,6)
    m=totals(labels*k+length,x*voice,2*k).reshape(2,k)
    return {'original_normalization':err_row,'group_duration':err_gl,
            'activity_bins':float(np.max(np.abs(c[0]-c[1])/q[:,None])),
            'activity_means':float(np.max(np.abs(m[0]-m[1])/q))}


def fit_flow(base,row,target,gl,gt,labels,length,bins,voice,q,k,maxiter=2500,progress_path=None):
    keep=base>0
    support=totals(labels*k*6+length*6+bins,(base>0).astype(float),2*k*6).reshape(2,k*6)>0
    shared=support[0]&support[1]
    keep&=shared[length*6+bins]
    keep_ids=np.flatnonzero(keep)
    if not len(keep_ids):raise ValueError('No common activity support')
    x=base[keep].copy();r=row[keep];g=gl[keep] if gl is not None else None
    y,l,b,v=labels[keep],length[keep],bins[keep],voice[keep]
    coefficient=(2*y-1)*v
    for it in range(1,maxiter+1):
        sums=totals(r,x,len(target))
        if np.any((target>1e-14)&(sums==0)):raise ValueError('An original/context has no shared support')
        factor=np.divide(target,sums,out=np.ones_like(target),where=sums>0);x*=factor[r]
        if g is not None:
            sums=totals(g,x,len(gt))
            if np.any((gt>1e-14)&(sums==0)):raise ValueError('An attack/length has no shared support')
            x*=np.divide(gt,sums,out=np.ones_like(gt),where=sums>0)[g]
        sums=totals(y*k*6+l*6+b,x,2*k*6).reshape(2,k*6)
        common=np.sqrt(sums[0]*sums[1])
        factors=np.divide(common[None,:],sums,out=np.zeros_like(sums),where=sums>0)
        x*=factors[y,l*6+b]
        lam=np.zeros(k)
        for newton in range(30):
            z=x*np.exp(np.clip(-lam[l]*coefficient,-80,80))
            f=totals(l,z*coefficient,k)
            if np.max(np.abs(f)/q)<1e-12:break
            d=totals(l,z*coefficient**2,k)
            lam+=np.clip(np.divide(f,d,out=np.zeros_like(f),where=d>0),-4,4)
        x=z
        if it%20==0:
            err=errors(x,r,target,g,gt,y,l,b,v,q,k)
            if progress_path is not None and it%100==0:
                Path(progress_path).write_text(json.dumps({'stage':'global probability fitting','iterations':it,'errors':err}))
            if max(err.values())<2e-8:
                result=np.zeros_like(base);result[keep_ids]=x
                return result,{'method':'KL_IPF','iterations':it,'errors':err}
    # Handle nearly-boundary feasible solutions without switching to sparse LP crops.
    m=len(x);cols=np.arange(m);rows=[r];values=[1/target[r]];rhs=[np.where(target>1e-14,1.,0.)];offset=len(target)
    if g is not None:
        rows.append(offset+g);values.append(1/gt[g]);rhs.append(np.ones(len(gt)));offset+=len(gt)
    rows.append(offset+l*6+b);values.append((2*y-1)/q[l]);rhs.append(np.zeros(k*6));offset+=k*6
    rows.append(offset+l);values.append(coefficient/q[l]);rhs.append(np.zeros(k));offset+=k
    A=sparse.coo_matrix((np.concatenate(values),(np.concatenate(rows),np.tile(cols,len(rows)))),shape=(offset,m)).tocsr()
    rhs=np.concatenate(rhs)
    if progress_path is not None:
        Path(progress_path).write_text(json.dumps({'stage':'feasibility check','iterations':maxiter,'errors':err}))
    feasible=linprog(np.zeros(m),A_eq=A,b_eq=rhs,bounds=(0,None),method='highs',
                     options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9,'time_limit':90})
    if not feasible.success:
        if feasible.status==2:raise ValueError('Joint constraints infeasible: '+feasible.message)
        raise RuntimeError('Feasibility check did not complete: '+feasible.message)
    reference=np.maximum(x,base[keep]*1e-100)
    def dual(z):
        flow=reference*np.exp(np.clip(-A.T@z,-700,100))
        return float(z@rhs+flow.sum()),rhs-A@flow
    fitted=minimize(dual,np.zeros(offset),jac=True,method='L-BFGS-B',
                    options={'maxiter':12000,'ftol':1e-15,'gtol':1e-10,'maxls':60,'maxcor':80})
    x=reference*np.exp(np.clip(-A.T@fitted.x,-700,100))
    # The sampler conditions on the selected original/context. Evaluate the
    # same normalized probabilities here rather than an approximate dual mass.
    sums=totals(r,x,len(target))
    x*=np.divide(target,sums,out=np.ones_like(target),where=sums>0)[r]
    err=errors(x,r,target,g,gt,y,l,b,v,q,k)
    if max(err.values())>1e-6:raise ValueError('Feasible but probability fit did not converge: '+str(err))
    result=np.zeros_like(base);result[keep_ids]=x
    return result,{'method':'KL_IPF_and_dual_refinement','iterations':maxiter+int(fitted.nit),'errors':err}


def fixed_mean_ranges(lp,orig,length,voice,label,nclass,k,q):
    n=len(lp);ctx=orig*k+length
    low=np.full(n*k,np.inf);high=np.full(n*k,-np.inf)
    np.minimum.at(low,ctx,voice);np.maximum.at(high,ctx,voice)
    low=low.reshape(n,k);high=high.reshape(n,k)
    lower=[];upper=[]
    for y in [0,1]:
        ids=np.flatnonzero(label==y);mass=lp[ids]/nclass[y]
        lower.append(np.where(mass>0,mass*np.where(np.isfinite(low[ids]),low[ids],0),0).sum(0)/q)
        upper.append(np.where(mass>0,mass*np.where(np.isfinite(high[ids]),high[ids],0),0).sum(0)/q)
    gap=np.maximum(lower[0],lower[1])-np.minimum(upper[0],upper[1])
    return float(gap.max()),{'lower':np.asarray(lower).tolist(),'upper':np.asarray(upper).tolist()}


def main(root):
    root=Path(root);os.nice(10);start=time.monotonic()
    args,_,_=arguments.get_args();args['duration_policy_path']=str(root/'duration_policy.json')
    args['activity_policy_path']=None
    ds=MatchedTrainSet(args,load_train_items_only('/data/ASVspoof2019'))
    rawmeta=json.loads((root/'candidate_metadata.json').read_text())
    assert rawmeta['source_fingerprint']==ds.source_fingerprint and rawmeta['duration_policy_sha256']==ds.policy_sha256
    cache=root/'raw_candidates.npz';assert hashlib.sha256(cache.read_bytes()).hexdigest()==rawmeta['candidate_npz_sha256']
    with np.load(cache,allow_pickle=False) as f:data={key:f[key] for key in f.files}
    k=len(ds.targets);n=len(ds);offsets=data['context_offsets']
    context=np.repeat(np.arange(n*k),np.diff(offsets));all_orig=context//k;all_length=context%k
    frames=ds.targets[all_length]//320
    all_voice=data['speech_frames']/frames
    all_bins=activity_bins(data['speech_frames'],frames)
    multiplicity=data['multiplicity'].astype(float)
    pos_base=multiplicity/totals(context,multiplicity,n*k)[context]
    old_lp=np.stack([ds.tables[ds.keys[i]][ds.buckets[i]-1] for i in range(n)])
    source_labels=np.array([v.label for v in ds.items])
    diagnostics=[]
    for speaker in sorted({v.speaker for v in ds.items}):
        ids=np.array([i for i,v in enumerate(ds.items) if v.speaker==speaker])
        mapping=np.full(n,-1);mapping[ids]=np.arange(len(ids))
        selected=np.flatnonzero(mapping[all_orig]>=0);orig=mapping[all_orig[selected]]
        label=source_labels[ids];nclass={y:int((label==y).sum()) for y in [0,1]}
        gap,ranges=fixed_mean_ranges(old_lp[ids],orig,all_length[selected],all_voice[selected],label,nclass,k,ds.q)
        diagnostics.append({'speaker':speaker,'fixed_per_original_length_mean_gap':gap,'mean_ranges':ranges})
    # Match class-global activity while retaining every original, speaker/attack
    # prior and each speaker/attack duration target. Individual speaker activity
    # equality is not asserted when the raw candidate supports do not overlap.
    source_prior=ds.source_probability
    valid=ds.lengths[:,None]>=ds.targets[None,:]
    flexible=np.where(valid,np.maximum(old_lp,1e-12),0);flexible/=flexible.sum(1)[:,None]
    base=source_prior[all_orig]*flexible[all_orig,all_length]*pos_base
    groups=sorted(ds.profiles);gm={key:i for i,key in enumerate(groups)}
    group=np.array([gm[key] for key in ds.keys]);gl=group[all_orig]*k+all_length
    group_mass=np.bincount(group,weights=source_prior,minlength=len(groups))
    gt=(group_mass[:,None]*ds.q[None,:]).ravel()
    (root/'fit_progress.json').write_text(json.dumps({'stage':'global probability fitting','originals':n,'candidates':len(context)}))
    flow,report=fit_flow(base,all_orig,source_prior,gl,gt,source_labels[all_orig],all_length,
        all_bins,all_voice,ds.q,k,maxiter=6500,progress_path=root/'fit_progress.json')
    output_mass=flow/source_prior[all_orig]
    reports=[report]
    original_sum=totals(all_orig,output_mass,n)
    assert np.all(original_sum>0)
    conditional=output_mass/original_sum[all_orig]
    length_probability=totals(context,conditional,n*k).reshape(n,k)
    position_probability=np.divide(conditional,length_probability[all_orig,all_length],out=np.zeros_like(conditional),where=length_probability[all_orig,all_length]>0)
    arrays_path=root/'activity_sampling.npz'
    np.savez_compressed(arrays_path,context_offsets=offsets,starts=data['starts'],speech_frames=data['speech_frames'],
        position_probabilities=position_probability,length_probabilities=length_probability,
        source_uids=data['source_uids'],source_sizes=data['source_sizes'],source_mtime_ns=data['source_mtime_ns'],source_sha256=data['source_sha256'])
    meta={'schema_version':1,'reference':'raw_before_RawBoost','source_fingerprint':ds.source_fingerprint,
        'duration_policy_sha256':ds.policy_sha256,'arrays_file':arrays_path.name,
        'activity_matching_scope':'class_global_with_source_marginals',
        'arrays_sha256':hashlib.sha256(arrays_path.read_bytes()).hexdigest(),
        'activity_bins':['zero','(0,.2]','(.2,.4]','(.4,.6]','(.6,.8]','(.8,1]'],
        'validation_probability_tolerance':2e-6,
        'vad':rawmeta['vad'],'constraints':['original source probabilities unchanged','speaker/attack duration targets unchanged',
            'same class-global activity bins and mean within each length','no original excluded'],
        'fit':reports,'fixed_context_diagnostics':diagnostics,'within_speaker_activity_equality_guaranteed':False,'dev_used':False,'eval_used':False,'model_training':False,
        'candidates':len(context),'originals':n,'elapsed_seconds':time.monotonic()-start}
    path=root/'activity_policy.json';path.write_text(json.dumps(meta,indent=2))
    from activity_policy import ActivityPolicy
    checked=ActivityPolicy(path,ds)
    meta['validated_statistics']=checked.statistics
    path.write_text(json.dumps(meta,indent=2))
    (root/'fit_progress.json').write_text(json.dumps({'stage':'complete','elapsed_seconds':time.monotonic()-start,'statistics':checked.statistics}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();main(a.root)
