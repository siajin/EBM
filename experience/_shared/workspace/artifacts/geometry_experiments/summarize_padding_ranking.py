"""Secondary ranking check on the deliberately stratified471 selected cohort."""
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.metrics import roc_auc_score
from frozen_common import weighted_eers
from run_padding_causal import OUT


def independent_eer(y,s):
    order=np.argsort(-s,kind='stable');yy=y[order];ss=s[order]
    last=np.r_[np.flatnonzero(ss[:-1]!=ss[1:]),len(ss)-1]
    fpr=np.r_[0,np.cumsum(1-yy)[last]/sum(1-yy)]
    fnr=np.r_[1,1-np.cumsum(yy)[last]/sum(yy)]
    diff=fpr-fnr;i=np.flatnonzero(diff>=0)[0]
    if i==0:return float(100*fpr[0])
    t=-diff[i-1]/(diff[i]-diff[i-1]);return float(100*((1-t)*fpr[i-1]+t*fpr[i]))


def main():
    f=pd.read_parquet(OUT/'scores.parquet');ctx=pd.read_parquet(OUT/'frontend_context/results.parquet').set_index('utterance_id')
    base=f[f.condition=='native_wrap'].sort_values('utterance_id').reset_index(drop=True);y=base.y.to_numpy(int)
    scores={}
    for name in ['native_wrap','native_original','native_reflect']:
        v=f[f.condition==name].set_index('utterance_id').loc[base.utterance_id];scores[name]=v.cosine_s.to_numpy()
    for name in ['mask2_score','mask7_score']:scores[name]=ctx.loc[base.utterance_id,name].to_numpy()
    names,codes=np.unique(base.source_corpus+'|'+base.speaker_group,return_inverse=True)
    rng=np.random.default_rng(2026110617);weights=np.zeros((5000,len(names)),int)
    for source in sorted(base.source_corpus.unique()):
        ix=np.flatnonzero(np.char.startswith(names.astype(str),source+'|'));weights[:,ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),5000)
    draws={name:weighted_eers(y,s,codes,weights) for name,s in scores.items()}
    rows=[]
    for name,s in scores.items():
        point=independent_eer(y,s)
        assert abs(point-weighted_eers(y,s,codes,np.ones((1,len(names)),int))[0])<1e-10
        d=draws[name]-draws['native_wrap']
        rows.append({'condition':name,'N':len(y),'EER_percent':point,'AUROC':float(roc_auc_score(y,s)),
            'EER_worsening_vs_native_pp':float(point-independent_eer(y,scores['native_wrap'])),
            'exploratory_pointwise95_delta_CI':np.nanquantile(d,[.025,.975]).tolist()})
    out={'scope':'Exploratory posthoc ranking check; same stratified471 only, not fullDF or a population-weighted benchmark',
        'N_normals':int(y.sum()),'N_spoofs':int((1-y).sum()),'source_speaker_clusters':len(names),'bootstrap_draws':5000,
        'CI_scope':'source-stratified paired speaker cluster bootstrap, pointwise95, no original-family or model-fit uncertainty',
        'rows':rows,'native_threshold_errors_are_a_different_metric':True,'independent_DET_and_weighted_ROC_agree':True}
    (OUT/'selected_cohort_ranking.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
