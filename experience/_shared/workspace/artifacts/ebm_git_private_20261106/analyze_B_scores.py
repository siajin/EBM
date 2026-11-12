"""Independent score/coverage checks and paired speaker-bootstrap diagnostics."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest

EXPECTED={'dev':(24844,2548),'19LA':(71237,7355),'21LA':(148176,14816),'21DF':(533928,14869)}
BINS=['under2','2to3.0275','3.0275to4.0375','atleast4.0375']

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def independent_roc(y,s):
    order=np.argsort(-s,kind='stable');scores=s[order];labels=y[order]
    ends=np.r_[np.flatnonzero(scores[:-1]!=scores[1:]),len(scores)-1]
    pos=labels.sum();neg=len(labels)-pos;assert pos and neg
    tp=np.cumsum(labels)[ends];fp=(ends+1)-tp
    fpr=np.r_[0.,fp/neg];fnr=np.r_[1.,1-tp/pos]
    thresholds=np.r_[np.inf,scores[ends]];d=fpr-fnr;k=np.flatnonzero(d>=0)[0]
    fraction=-d[k-1]/(d[k]-d[k-1]);eer=fpr[k-1]+fraction*(fpr[k]-fpr[k-1])
    j=min(range(len(d)),key=lambda i:(abs(d[i]),fpr[i],-thresholds[i]))
    return float(eer*100),float(thresholds[j])

def bootstrap_mean_delta(frame,delta,seed,multiplier=100,primary=False):
    groups=frame.assign(_delta=delta).groupby('speaker',sort=True)._delta.agg(['sum','size'])
    if len(groups)<2:return {'speakers':len(groups),'CI95':None,'family_CI95':None,'bootstrap_replicates':0}
    rng=np.random.default_rng(seed);n=len(groups);values=[]
    for start in range(0,5000,256):
        weights=rng.multinomial(n,np.full(n,1/n),size=min(256,5000-start))
        values.extend((multiplier*(weights@groups['sum'].to_numpy())/(weights@groups['size'].to_numpy())).tolist())
    return {'speakers':n,'CI95':np.quantile(values,[.025,.975]).tolist(),
            'family_CI95':np.quantile(values,[.05/12,1-.05/12]).tolist() if primary else None,
            'bootstrap_replicates':len(values),'bootstrap_unit':'protocol speaker, paired model outcomes',
            'family':'six full-target class error differences; Bonferroni95%' if primary else 'exploratory marginal95%, not family corrected'}

def analyze(folder):
    folder=Path(folder);metrics=json.loads((folder/'metrics.json').read_text())
    manifest=json.loads((folder/'download_manifest.json').read_text())
    out=folder/'analysis';out.mkdir(exist_ok=True)
    frames={};comparison={};rate_rows=[];checks=[];thresholds={}
    for phase,(N,n_bona) in EXPECTED.items():
        path=folder/f'paired_scores_{phase}.parquet'
        assert digest(path)==manifest['files'][path.name]['sha256']
        f=pd.read_parquet(path).sort_values('index').reset_index(drop=True)
        assert f['index'].tolist()==list(range(N)) and f.uid.is_unique
        assert len(f)==N and f.label.sum()==n_bona and set(f.label)=={0,1}
        assert not f.isna().any().any()
        assert (f.used_samples==np.minimum(f.original_samples,64600)).all() and f.used_samples.min()>=720
        f['length_bin']=pd.cut(f.original_samples,[0,32000,48440,64600,np.inf],right=False,labels=BINS).astype(str)
        y=f.label.to_numpy();comparison[phase]={}
        for model,column in [('B','score_B'),('legacy240','score_legacy240')]:
            s=f[column].to_numpy();assert np.isfinite(s).all() and np.max(np.abs(s))<=1.000001
            eer,threshold=independent_roc(y,s)
            assert abs(eer-metrics[phase][model]['EER_percent'])<1e-9
            if phase=='dev':
                thresholds[model]=threshold
                assert abs(threshold-metrics[phase][model]['dev_threshold'])<1e-12
            accept=s>=thresholds[model]
            error=np.where(y==1,~accept,accept)
            f['error_'+model]=error.astype(int)
            f['angle_'+model]=np.degrees(np.arccos(np.clip(s,-1,1)))
            fnr=float(100*error[y==1].mean());fpr=float(100*error[y==0].mean())
            assert abs(fnr-metrics[phase][model]['normal_reject_percent'])<1e-9
            assert abs(fpr-metrics[phase][model]['spoof_accept_percent'])<1e-9
            comparison[phase][model]={'EER_percent':eer,'dev_threshold':thresholds[model],
                'normal_reject_percent':fnr,'spoof_accept_percent':fpr,'N':N,'bonafide':n_bona,'spoof':N-n_bona}
        if phase!='dev':
            for group in ['all',*BINS]:
                for label in [1,0]:
                    g=f[(f.label==label)&(True if group=='all' else f.length_bin==group)].copy()
                    row={'phase':phase,'length_bin':group,'label':label,'error_kind':'normal_reject' if label else 'spoof_accept','N':len(g)}
                    if not len(g):
                        row.update(B_error_count=0,legacy_error_count=0,B_error_percent=None,legacy_error_percent=None,delta_pp=None,CI95=None,family_CI95=None,speakers=0)
                    else:
                        b=g.error_B.to_numpy();old=g.error_legacy240.to_numpy();delta=b-old
                        worse=int(((b==1)&(old==0)).sum());fixed=int(((b==0)&(old==1)).sum())
                        primary=group=='all';key=f'EBM-B-final-rates-v1|{phase}|{label}|{group}'
                        seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'big')
                        ci=bootstrap_mean_delta(g,delta,seed,primary=primary)
                        row.update(B_error_count=int(b.sum()),legacy_error_count=int(old.sum()),
                            B_error_percent=float(100*b.mean()),legacy_error_percent=float(100*old.mean()),
                            delta_pp=float(100*delta.mean()),B_only_errors=worse,legacy_only_errors=fixed,
                            mcnemar_p_assuming_independent_utterances=float(binomtest(min(worse,fixed),worse+fixed,.5).pvalue) if worse+fixed else 1.,
                            B_mean_angle=float(g.angle_B.mean()),legacy_mean_angle=float(g.angle_legacy240.mean()),**ci)
                    rate_rows.append(row)
        frames[phase]=f
        checks.append({'phase':phase,'N':N,'unique_UID':True,'full_indices':True,'labels_match_expected_counts':True,'scores_finite':True,'caps_valid':True,'independent_EER_matches_server':True,'dev_threshold_only':True})
    save(out/'comparison.json',comparison)
    save(out/'paired_error_rates.json',rate_rows)
    pd.DataFrame(rate_rows).to_csv(out/'paired_error_rates.csv',index=False)
    save(out/'validation.json',{'status':'passed','checks':checks,'thresholds':thresholds,
        'uncertainty':'paired protocol-speaker cluster bootstrap5000; primary six global class-error differences family corrected; bins exploratory',
        'EER_uncertainty':'point estimates only; class error-difference intervals are not EER intervals',
        'limits':['one trained seed','legacy/B training differences confounded','finite benchmark and protocol speakers, not all unseen sources']})
    result={'status':'paired_score_analysis_ready','comparison':comparison,'primary_error_differences':[r for r in rate_rows if r['length_bin']=='all']}
    save(out/'summary.json',result)
    print(json.dumps(result),flush=True)

if __name__=='__main__':analyze(sys.argv[1])
