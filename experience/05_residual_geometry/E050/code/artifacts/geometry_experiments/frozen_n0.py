"""Complete frozen-cache F1-F4 panels with fixed samples and controls."""
import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, cdist
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
from frozen_common import *

def fisher(z,rows):
    b=z[rows.y.to_numpy()==1];attacks=rows.attack_id.to_numpy()
    return np.mean([(b.mean(0)-z[attacks==a].mean(0))**2/(b.var(0,ddof=1)+z[attacks==a].var(0,ddof=1)+1e-8) for a in sorted(set(attacks[rows.y.to_numpy()==0]))],axis=0)

def f2(data,geo,out):
    out.mkdir(exist_ok=False)
    tg=geo['T'];tr=data['T'].rows
    ix=np.r_[stable_indices(tr[tr.y==1].reset_index().rename(columns={'index':'original'}),'F2-bandwidth|1701|',500)]
    chosen=[]
    for label in [1,0]:
        sub=tr[tr.y==label].reset_index();chosen.extend(sub.loc[stable_indices(sub,'F2-bandwidth|1701|',500),'index'])
    distances=pdist(tg['direction'][chosen]);band=float(np.median(distances[distances>0]));assert band>0
    save(out/'bandwidth.json',dict(bandwidth=band,fit_split='T',selected_UIDs=tr.iloc[chosen].utterance_id.tolist()))
    reports=[];coverage=[];balances=[]
    for name in ['DF_pilot','21LA_pilot']:
        rows=data[name].rows;g=geo[name];s=g['cosine'];n=np.log(g['norm']);valid=g['valid']
        columns=['source_corpus','speaker_group','codec'] if name=='DF_pilot' else ['speaker_group','codec','transmission']
        for ds,dn,tag in [(.02,.002,'primary'),(.05,.005,'wide_sensitivity')]:
            pairs=[]
            for key,part in rows.groupby(columns,dropna=False,sort=True):
                bi=part.index[(part.y==1)&valid[part.index]].to_numpy();si=part.index[(part.y==0)&valid[part.index]].to_numpy()
                order=sorted(bi,key=lambda j:hashlib.sha256(('F2|17|'+rows.iloc[j].utterance_id).encode()).hexdigest())
                remaining=set(si)
                for j in order:
                    candidates=np.array(sorted(remaining),int)
                    if not len(candidates):break
                    candidates=candidates[(np.abs(s[candidates]-s[j])<=ds)&(np.abs(n[candidates]-n[j])<=dn)]
                    if not len(candidates):continue
                    k=min(candidates,key=lambda k:(((s[k]-s[j])/ds)**2+((n[k]-n[j])/dn)**2,rows.iloc[k].utterance_id))
                    remaining.remove(k);pairs.append((j,k))
            pairs=sorted(pairs,key=lambda p:hashlib.sha256(('F2|17|'+rows.iloc[p[0]].utterance_id).encode()).hexdigest())[:2000]
            bi=np.asarray([p[0] for p in pairs],int);si=np.asarray([p[1] for p in pairs],int)
            match=pd.DataFrame({'bona_index':bi,'spoof_index':si,'bona_uid':rows.iloc[bi].utterance_id.to_numpy(),'spoof_uid':rows.iloc[si].utterance_id.to_numpy(),'dependency_group':rows.iloc[bi].dependency_group.to_numpy(),'delta_s':s[bi]-s[si],'delta_lognorm':n[bi]-n[si]})
            match.to_parquet(out/(name+'_'+tag+'_matches.parquet'),index=False)
            ng=match.dependency_group.nunique();status='complete_descriptive' if len(pairs)>=200 and ng>=10 else 'inconclusive_low_overlap'
            reports.append(dict(dataset=name,condition=tag,pairs=len(pairs),speaker_groups=ng,status=status))
            for label,inds in [(1,bi),(0,si)]:
                full=rows[rows.y==label]
                for column in ['source_corpus','codec']:
                    for value,part in full.groupby(column):
                        coverage.append(dict(dataset=name,matching=tag,y=label,condition=column,value=value,full_count=len(part),matched_count=int(np.isin(inds,part.index).sum()),inclusion_fraction=float(np.isin(inds,part.index).sum()/len(part))))
            if len(pairs):
                for measure,vals in [('cosine',s),('lognorm',n)]:
                    balances.append(dict(dataset=name,matching=tag,measure=measure,full_mean_bona=float(vals[rows.y==1].mean()),full_mean_spoof=float(vals[rows.y==0].mean()),matched_mean_bona=float(vals[bi].mean()),matched_mean_spoof=float(vals[si].mean()),max_absolute_paired_difference=float(np.max(np.abs(vals[bi]-vals[si])))))
            if not len(pairs):continue
            ub,us=g['direction'][bi],g['direction'][si]
            def moments(u):
                mean=u.mean(0);center=u-mean;cov=center.T@center/len(u);moment=u.T@u/len(u)
                def pr(a):
                    den=float(np.sum(a*a));return float(np.trace(a)**2/den) if den>1e-12 else None
                return dict(resultant_length=float(np.linalg.norm(mean)),centered_covariance_PR=pr(cov),uncentered_second_moment_PR=pr(moment)),mean
            mb,bm=moments(ub);ms,sm=moments(us)
            angle=float(np.degrees(np.arccos(np.clip(bm@sm/(np.linalg.norm(bm)*np.linalg.norm(sm)),-1,1)))) if min(np.linalg.norm(bm),np.linalg.norm(sm))>1e-12 else None
            kbb=np.exp(-cdist(ub,ub,'sqeuclidean')/(2*band**2));kss=np.exp(-cdist(us,us,'sqeuclidean')/(2*band**2));kbs=np.exp(-cdist(ub,us,'sqeuclidean')/(2*band**2))
            mmd=float(kbb.mean()+kss.mean()-2*kbs.mean())
            groups,codes=np.unique(match.dependency_group,return_inverse=True);indicator=np.eye(len(groups))[codes];sizes=indicator.sum(0)
            kernel=indicator.T@(kbb+kss-kbs-kbs.T)@indicator
            counts=np.random.default_rng(7301).multinomial(len(groups),np.full(len(groups),1/len(groups)),size=10000)
            denom=counts@sizes;boot=np.einsum('bi,ij,bj->b',counts,kernel,counts)/(denom**2)
            save(out/(name+'_'+tag+'_direction_statistics.json'),dict(status=status,bona=mb,spoof=ms,mean_direction_angle_degrees=angle,biased_MMD2=mmd,MMD_CI95=interval(boot),speaker_bootstrap_draws=10000,bandwidth=band,approximate_matching_not_conditional_independence_test=True))
    pd.DataFrame(coverage).to_csv(out/'coverage.csv',index=False);pd.DataFrame(balances).to_csv(out/'matching_balance.csv',index=False);save(out/'summary.json',reports)

def f3(data,geo,out):
    out.mkdir(exist_ok=False);x=geo['T']['unit_z'][data['T'].rows.y.to_numpy()==1];cov=np.cov(x,rowvar=False);scale=np.trace(cov)/160;c=data['T'].centroid
    np.savez(out/'covariance.npz',covariance=cov,centroid=c)
    matrices={'identity':np.eye(160)};registry=[]
    for mode in ['diagonal','full']:
        for gamma in [.01,.1,.5,1.]:
            shrink=(1-gamma)*cov+gamma*scale*np.eye(160)
            if mode=='diagonal':a=np.diag(1/np.maximum(np.diag(shrink),1e-6*scale))
            else:
                eig,vec=np.linalg.eigh(shrink);a=(vec*(1/np.maximum(eig,1e-6*scale)))@vec.T
            a*=160/np.trace(a);matrices[f'{mode}_{gamma}']=a
    scores={}
    for name,g in geo.items():
        if name=='T':continue
        r=g['unit_z']-c;scores[name]={n:1-.5*np.einsum('bi,ij,bj->b',r,a,r) for n,a in matrices.items()}
        assert np.max(np.abs(scores[name]['identity']-g['cosine']))<1e-10
    for n,a in matrices.items():
        s=scores['V'][n];registry.append(dict(model=n,V_EER=eer(data['V'].rows.y.to_numpy(),s),threshold=threshold(data['V'].rows.y.to_numpy(),s),trace=np.trace(a)))
    selected={}
    for mode in ['diagonal','full']:
        opts=[r for r in registry if r['model'].startswith(mode)];minimum=min(r['V_EER'] for r in opts)
        selected[mode]=max([r for r in opts if r['V_EER']<=minimum+1e-12],key=lambda r:float(r['model'].split('_')[-1]))['model']
    save(out/'metric_registry.json',dict(selected=selected,candidates=registry,selection='V EER; tied prefer larger gamma',center_fixed=True))
    records=[];effects=[]
    for name in ['DF_pilot','21LA_pilot']:
        d=data[name];frame=d.rows[['utterance_id','y','dependency_group']].copy()
        for n,score in scores[name].items():
            frame[n]=score;th=next(r['threshold']['threshold'] for r in registry if r['model']==n)
            records.append(dict(dataset=name,model=n,condition='all',**metric(d.rows.y.to_numpy(),score,th)))
            for column in ['source_corpus','codec']:
                for value,idx in d.rows.groupby(column).groups.items():
                    ii=np.asarray(idx);records.append(dict(dataset=name,model=n,condition=column+':'+str(value),**metric(d.rows.y.to_numpy()[ii],score[ii],th)))
        frame.to_parquet(out/(name+'_scores.parquet'),index=False)
        bs,meta=bootstrap_scores(d.rows.y.to_numpy(),scores[name],d.rows.dependency_group.to_numpy())
        for n in scores[name]:
            if n=='identity':continue
            delta=bs['identity']-bs[n];effects.append(dict(dataset=name,model=n,delta_EER_pp=eer(d.rows.y.to_numpy(),scores[name]['identity'])-eer(d.rows.y.to_numpy(),scores[name][n]),CI95=interval(delta),CI_adjusted=interval(delta,1-.05/8),**meta))
    pd.DataFrame(records).to_csv(out/'condition_effects.csv',index=False);save(out/'paired_effects.json',effects)

def eta(values,labels):
    mean=values.mean(0);den=float(np.sum((values-mean)**2))
    if den<=1e-12:return None
    return float(sum(np.sum(labels==v)*np.sum((values[labels==v].mean(0)-mean)**2) for v in np.unique(labels))/den)

def f4(data,geo,out):
    out.mkdir(exist_ok=False);t=geo['T']['direction'];mean=t.mean(0);_,_,vt=np.linalg.svd(t-mean,full_matrices=False);pca=vt[:3];np.savez(out/'T_fit_PCA.npz',mean=mean,components=pca)
    cover=[];variances=[];padding=[];conditions=[];th=threshold(data['V'].rows.y.to_numpy(),geo['V']['cosine'])['threshold']
    for name in ['DF_pilot','21LA_pilot']:
        rows=data[name].rows.copy();g=geo[name];padding_bin=np.digitize(rows.padding_fraction.to_numpy(),[0,.25,.5],right=True);rows['padding_bin']=padding_bin
        main='source_corpus' if name=='DF_pilot' else 'transmission'
        for y in [0,1]:
            idx=np.flatnonzero((rows.y.to_numpy()==y)&g['valid']);u=g['direction'][idx]
            for column in [main,'codec','padding_bin']:
                labs=rows.iloc[idx][column].astype(str).to_numpy();variances.append(dict(dataset=name,y=y,stratum='all',factor=column,eta_squared=eta(u,labs),n=len(idx)))
            for val,part in rows.iloc[idx].groupby(main):
                ii=part.index.to_numpy();variances.append(dict(dataset=name,y=y,stratum=main+':'+str(val),factor='codec',eta_squared=eta(g['direction'][ii],part.codec.astype(str).to_numpy()),n=len(ii)))
            coordinates=(u-mean)@pca.T
            for measure in ['padding_fraction','duration']:
                vals=rows.iloc[idx][measure].to_numpy()
                for pc in range(3):
                    rho,p=spearmanr(vals,coordinates[:,pc]);padding.append(dict(dataset=name,y=y,measure=measure,PC=pc+1,Spearman_rho=float(rho) if np.isfinite(rho) else None,n=len(idx)))
        for cols in [[main],['codec'],['padding_bin'],['attack_id'],[main,'codec']]:
            for key,part in rows.groupby(cols,dropna=False):
                ii=part.index.to_numpy();conditions.append(dict(dataset=name,factor='*'.join(cols),value=str(key),speakers=part.dependency_group.nunique(),**metric(part.y.to_numpy(),g['cosine'][ii],th)))
                for y,cell in part.groupby('y'):
                    cover.append(dict(dataset=name,factor='*'.join(cols),value=str(key),y=y,n=len(cell),speakers=cell.dependency_group.nunique(),n_lt20=len(cell)<20,speakers_lt5=cell.dependency_group.nunique()<5,direction_resultant_length=float(np.linalg.norm(g['direction'][cell.index].mean(0))),mean_cosine=float(g['cosine'][cell.index].mean()),std_cosine=float(g['cosine'][cell.index].std())))
    pd.DataFrame(cover).to_csv(out/'coverage.csv',index=False);pd.DataFrame(variances).to_csv(out/'vector_variance.csv',index=False);pd.DataFrame(padding).to_csv(out/'padding_duration.csv',index=False);pd.DataFrame(conditions).to_csv(out/'condition_metrics.csv',index=False)

def f1(data,geo,out):
    out.mkdir(exist_ok=False);rank=np.argsort(-fisher(data['T'].z.astype(float),data['T'].rows),kind='stable');records=[];effects=[];mask_record={}
    for basis in ['original',511,512,513]:
        if basis=='original':q=np.eye(160);ks=[8,16,40,80]
        else:
            mat=np.random.default_rng(basis).normal(size=(160,160));q,r=np.linalg.qr(mat);q=q*np.where(np.diag(r)<0,-1,1);ks=[40]
        zt=data['T'].z.astype(float)@q;c=data['T'].centroid@q
        order=np.argsort(-fisher(zt,data['T'].rows),kind='stable')
        masks={'native':np.arange(160)}
        for k in ks:
            masks[f'top_keep_{k}']=order[:k];masks[f'top_drop_{k}']=order[k:]
            for seed in range(200,220):
                ix=np.random.default_rng(seed).permutation(160);masks[f'random_keep_{k}_{seed}']=ix[:k];masks[f'random_drop_{k}_{seed}']=ix[k:]
        assert len(masks)==(169 if basis=='original' else 43)
        mask_record[str(basis)]={n:ix.tolist() for n,ix in masks.items()}
        np.save(out/(str(basis)+'_orthogonal.npy'),q)
        result={}
        thresholds={}
        for name in ['V','DF_pilot','21LA_pilot']:
            z=data[name].z.astype(float)@q;native=z/np.linalg.norm(z,axis=1,keepdims=True)@c;assert np.max(np.abs(native-geo[name]['cosine']))<1e-10
            frame=data[name].rows[['utterance_id','y','dependency_group']].copy();local={}
            for n,ix in masks.items():
                if n=='native':score=native
                else:
                    cc=c[ix];xx=z[:,ix];assert np.linalg.norm(cc)>1e-12
                    score=(xx@cc)/(np.linalg.norm(xx,axis=1)*np.linalg.norm(cc))
                local[n]=score;frame[n]=score
                if name=='V':thresholds[n]=threshold(data[name].rows.y.to_numpy(),score)
                records.append(dict(basis=str(basis),dataset=name,mask=n,**metric(data[name].rows.y.to_numpy(),score,thresholds[n]['threshold'])))
            frame.to_parquet(out/(str(basis)+'_'+name+'_scores.parquet'),index=False);result[name]=local
        save(out/(str(basis)+'_V_thresholds.json'),thresholds)
        for name in ['DF_pilot','21LA_pilot']:
            d=data[name];chosen={n:s for n,s in result[name].items() if '_40' in n}
            print('F1 bootstrap',basis,name,len(chosen),flush=True)
            bs,meta=bootstrap_scores(d.rows.y.to_numpy(),chosen,d.rows.dependency_group.to_numpy())
            # Re-sample the 20 mask designs inside each shared group draw.
            mask_draws=np.random.default_rng(7303).integers(0,20,size=(10000,20));rr=np.arange(10000)[:,None]
            for action in ['keep','drop']:
                random_names=[f'random_{action}_40_{s}' for s in range(200,220)]
                rand=np.stack([bs[n] for n in random_names],axis=1);randmean=rand[rr,mask_draws].mean(1);top=bs[f'top_{action}_40']
                delta=randmean-top if action=='keep' else top-randmean
                re=np.mean([eer(d.rows.y.to_numpy(),chosen[n]) for n in random_names]);te=eer(d.rows.y.to_numpy(),chosen[f'top_{action}_40'])
                effects.append(dict(basis=str(basis),dataset=name,action=action,top_EER=te,mean_random_EER=re,effect_pp=re-te if action=='keep' else te-re,CI95=interval(delta),CI975=interval(delta,.975),mask_seed_resampling=True,**meta))
            np.savez_compressed(out/(str(basis)+'_'+name+'_bootstrap.npz'),**bs)
    save(out/'masks.json',mask_record);pd.DataFrame(records).to_csv(out/'mask_curves.csv',index=False);save(out/'paired_effects.json',effects)

def run(out):
    out=Path(out);start=time.monotonic();data={name:panel(name) for name in ['T','V','DF_pilot','21LA_pilot']};geo={n:d.geometry() for n,d in data.items()}
    save(out/'N0_protocol_lock.json',dict(script_sha256=digest(__file__),spec_version='frozen-model-spec-v1.2-size-matched-21LA',manifest_sha256={n:d.provenance.get('analysis_manifest_sha256') for n,d in data.items()},new_EBM_training=False))
    for func in [f2,f4,f3,f1]:
        print('START',func.__name__,flush=True);s=time.monotonic();func(data,geo,out/func.__name__.upper());save(out/(func.__name__+'_complete.json'),dict(status='complete',seconds=time.monotonic()-s));print('DONE',func.__name__,flush=True)
    save(out/'N0_complete.json',dict(status='complete',seconds=time.monotonic()-start))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(DEFAULT_ROOT/'local_analysis/frozen_all_v1'));o=p.parse_args()
    with threadpool_limits(2):run(o.output)
