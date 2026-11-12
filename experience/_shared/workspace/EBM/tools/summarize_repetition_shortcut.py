"""Analyze new controlled frozen-model scores; no model/audio access."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from analyze_repetition_shortcut import eer,cluster_ci

def main(root):
    p=json.loads((root/'protocol.json').read_text());parts=[]
    for rank in range(2):
        file=root/f'full_rank{rank}.csv';done=json.loads((root/f'full_done_rank{rank}.json').read_text())
        assert hashlib.sha256(file.read_bytes()).hexdigest()==done['sha256'];parts.append(pd.read_csv(file))
    f=pd.concat(parts,ignore_index=True)
    f['bin']=pd.cut(f.original_samples,[0,16000,24000,32000,40000,48000,56000,64600,np.inf],labels=False)
    meta=pd.read_csv(root/'trial_metadata_21LA.txt',sep=r'\s+',header=None,names=['speaker_meta','uid','codec','transmission','attack_meta','class_meta','trim','subset'])
    assert meta.uid.is_unique
    f=f.merge(meta[['uid','codec']],how='left',on='uid',validate='many_to_one');f['codec']=f.codec.fillna('dev')
    metrics=[];errors=[];paired=[];thresholds={}
    for model in ['legacy240','B']:
        col=f'score_{model}'
        for cond in ['native','wrap4']:
            d=f[(f.phase=='dev')&(f.condition==cond)];fp,tp,thr=roc_curve(d.label,d[col],drop_intermediate=False)
            ix=int(np.argmin(abs(fp-(1-tp))));thresholds[model+'_'+cond]=float(thr[ix])
        for (phase,cond),g in f.groupby(['phase','condition']):
            metrics.append(dict(model=model,phase=phase,condition=cond,weighting='observed',eer=eer(g.label,g[col]),n=len(g)))
            counts=pd.crosstab(g['bin'],g.label).reindex(columns=[0,1],fill_value=0);common=counts[(counts[0]>0)&(counts[1]>0)]
            for name,target in [('overlap_min',common.min(axis=1)),('bona_marginal',common[1]),('uniform_common',pd.Series(1.,index=common.index))]:
                a=g[g.bin.isin(common.index)];target=target/target.sum();lookup=np.zeros((8,2))
                for b in common.index:lookup[b,:]=target.loc[b]/counts.loc[b].to_numpy()
                w=lookup[a.bin.to_numpy(int),a.label.to_numpy(int)]
                metrics.append(dict(model=model,phase=phase,condition=cond,weighting=name,eer=eer(a.label,a[col],w),n=len(a)))
            for grouping in [['label','bin'],['label','bin','attack','codec']]:
                for key,a in g.groupby(grouping):
                    y=int(key[0]);t=thresholds[model+'_native'];e=(a[col]<t) if y else (a[col]>=t)
                    errors.append(dict(model=model,phase=phase,condition=cond,label=y,bin=int(key[1]),attack=key[2] if len(key)>2 else 'ALL',codec=key[3] if len(key)>2 else 'ALL',n=len(a),errors=int(e.sum()),error_percent=float(e.mean()*100),threshold=t))
        a=f[f.condition=='native'].drop(columns=['condition']).set_index(['phase','uid'])
        b=f[f.condition=='wrap4'].set_index(['phase','uid']);assert a.index.equals(b.index)
        a=a.copy();a['wrap_minus_native']=b[col]-a[col]
        for (phase,y,binname),g in a.reset_index().groupby(['phase','label','bin']):
            ci,n=cluster_ci(g,'wrap_minus_native');t=thresholds[model+'_native']
            s0=g[col];s1=s0+g.wrap_minus_native;e0=s0<t if y else s0>=t;e1=s1<t if y else s1>=t
            paired.append(dict(model=model,phase=phase,label=int(y),bin=int(binname),n=len(g),speakers=n,delta=float(g.wrap_minus_native.mean()),ci_lo=ci[0],ci_hi=ci[1],native_errors=int(e0.sum()),wrap_errors=int(e1.sum()),correct_to_error=int((~e0&e1).sum()),error_to_correct=int((e0&~e1).sum())))
    pd.DataFrame(metrics).to_csv(root/'new_full_metrics.csv',index=False);pd.DataFrame(errors).to_csv(root/'new_length_attack_codec_errors.csv',index=False);pd.DataFrame(paired).to_csv(root/'new_paired_effects.csv',index=False)
    # Dose comparisons average head/tail per original utterance BEFORE cluster resampling.
    probe=pd.concat([pd.read_csv(root/f'probe_rank{r}.csv') for r in range(2)],ignore_index=True)
    dose=[]
    for model in ['legacy240','B']:
        col=f'score_{model}';wide=probe.pivot(index=['uid','speaker','label','attack'],columns='condition',values=col)
        for seconds in [1,2]:
            for treatment,control in [('repeat2','native'),('repeat3','native'),('repeat4','native'),('smooth_seams4','repeat4'),('smooth_seams4','smooth_sham4'),('reflect4','repeat4')]:
                delta=sum(wide[f'{end}_{seconds}s_{treatment}']-wide[f'{end}_{seconds}s_{control}'] for end in ['head','tail'])/2
                d=delta.rename('wrap_minus_native').reset_index()
                for y,g in d.groupby('label'):
                    ci,n=cluster_ci(g,'wrap_minus_native');dose.append(dict(model=model,label=int(y),crop_seconds=seconds,treatment=treatment,control=control,n=len(g),speakers=n,mean_delta=float(g.wrap_minus_native.mean()),ci_lo=ci[0],ci_hi=ci[1]))
    pd.DataFrame(dose).to_csv(root/'new_dose_controls.csv',index=False)
    result={'status':'analysis_complete','thresholds':thresholds,'full_metrics':metrics,'paired_effects':paired,'dose_controls':dose,'protocol_sha256':hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest(),'limits':p['limits'],'noise_std':0,'optimizer_updates':0,'interpretation_rule':'Report observed signs, class differences and shared-support sensitivities; no automatic claim that the origin of learned response was identified.'}
    (root/'analysis_complete.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
