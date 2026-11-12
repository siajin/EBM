"""Full-cache common-support and intervention-coverage audit, offline CPU.

Maximum bipartite matching replaces the earlier greedy lower bound. An
alternating-path minimum vertex cover certifies each maximum cardinality.
No model is fitted, no SSH/audio/GPU operations are performed.
"""
from pathlib import Path
from collections import deque
import hashlib
import json
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, diags
from scipy.sparse.csgraph import maximum_bipartite_matching, min_weight_full_bipartite_matching
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]/'geometry_cache'
OUT=ROOT/'local_analysis/full_score_overlap_20261106_v2'
CHECKPOINT='104873703a57cad98b686d770e0860e031a2cf9abef79b72668780c1660e0b24'
HASHES={}
CALIPERS=[.02,.01,.005]
CONDITIONS=['R2-10','R2-01','R2-11','R3-noise20','R3-noise10','R3-Opus32k_recompression']


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    p=Path(p);HASHES[str(p.resolve())]=digest(p)
    return pd.read_parquet(p) if p.suffix=='.parquet' else json.loads(p.read_text(encoding='utf-8'))


def save(name,obj):
    def conv(x):
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,np.ndarray):return x.tolist()
        raise TypeError(type(x).__name__)
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=conv,allow_nan=False)+'\n',encoding='utf-8')


def prepare(cat,key):
    entry=cat['runs'][key];assert entry['status']=='ready' and entry['identity']['checkpoint_sha256']==CHECKPOINT
    p=ROOT/entry['local_folder']/'extract/scores.parquet'
    f=read(p)
    expected=next(x['sha256'] for x in entry['files'] if x['path']=='extract/scores.parquet')
    assert HASHES[str(p.resolve())]==expected
    return f


def edges(frame,strict):
    # Native metadata only. Both classes retain all eligible score ranges.
    f=frame.copy()
    f['_bin']=pd.cut(f.duration,[0,2,3,4.0375,6,np.inf],right=False).astype(str)
    cols=['source_corpus','codec','_bin']
    if 'transmission' in f:cols+=['transmission']
    if strict:cols+=['speaker_group']
    assert f[cols].notna().all().all()
    f['_key']=f[cols].astype(str).agg('|'.join,axis=1)
    # UID hash order is independent of class response and the intervention.
    f['_order']=[hashlib.sha256(str(x).encode()).hexdigest() for x in f.utterance_id]
    b=f[f.y==1].sort_values('_order').reset_index(drop=True)
    s=f[f.y==0].sort_values('_order').reset_index(drop=True)
    grouped={}
    for k,g in s.groupby('_key',sort=False):
        ix=g.index.to_numpy();order=np.argsort(g.cosine_s.to_numpy(),kind='stable');ix=ix[order]
        grouped[k]=(ix,s.loc[ix,'cosine_s'].to_numpy(),s.loc[ix,'duration'].to_numpy())
    bs=b.cosine_s.to_numpy();bt=b.duration.to_numpy();parts=[];indptr=[0]
    for i,k in enumerate(b._key):
        pool=grouped.get(k)
        if pool is None:out=np.array([],dtype=np.int32)
        else:
            ix,score,dur=pool
            lo=np.searchsorted(score,bs[i]-.02,side='left');hi=np.searchsorted(score,bs[i]+.02,side='right')
            out=ix[lo:hi][abs(dur[lo:hi]-bt[i])<=.25].astype(np.int32)
            out=np.sort(out)
        parts.append(out);indptr.append(indptr[-1]+len(out))
    indices=np.concatenate(parts) if parts else np.array([],dtype=np.int32)
    graph=csr_matrix((np.ones(len(indices),dtype=np.int8),indices,np.asarray(indptr,dtype=np.int64)),shape=(len(b),len(s)))
    return b,s,graph


def certified_match(g,bscore,sscore):
    cardinal_match=maximum_bipartite_matching(g,perm_type='column')
    # Dummy edges allow unmatched bona. Their cost dominates the largest
    # possible real-edge total, so this first maximizes pair count, then
    # minimizes total absolute native-score gap among maximum matchings.
    cost=g.astype(float).copy()
    rr=np.repeat(np.arange(g.shape[0]),np.diff(g.indptr))
    cost.data=1+abs(bscore[rr]-sscore[g.indices])
    dummy=diags(np.full(g.shape[0],2*(g.shape[0]+1)),format='csr')
    left,right=min_weight_full_bipartite_matching(hstack([cost,dummy],format='csr'))
    match=np.full(g.shape[0],-1,dtype=int);real=right<g.shape[1];match[left[real]]=right[real]
    assert np.sum(match>=0)==np.sum(cardinal_match>=0)
    assert len(match)==g.shape[0]
    right_to_left=np.full(g.shape[1],-1,dtype=int)
    rows=np.flatnonzero(match>=0);right_to_left[match[rows]]=rows
    assert len(np.unique(match[rows]))==len(rows)
    # Alternating reachability from unmatched left vertices. Traversing a
    # matched edge L->R adds no extra vertices because its R was already reached.
    reach_l=match<0;reach_r=np.zeros(g.shape[1],dtype=bool);queue=deque(np.flatnonzero(reach_l).tolist())
    while queue:
        i=queue.popleft();neigh=g.indices[g.indptr[i]:g.indptr[i+1]];new=neigh[~reach_r[neigh]]
        reach_r[new]=True;left=right_to_left[new];left=left[left>=0];left=left[~reach_l[left]]
        reach_l[left]=True;queue.extend(left.tolist())
    cover_size=int((~reach_l).sum()+reach_r.sum())
    assert cover_size==len(rows)
    for i in np.flatnonzero(reach_l):assert reach_r[g.indices[g.indptr[i]:g.indptr[i+1]]].all()
    return rows,match[rows],dict(maximum_matching_size=len(rows),minimum_vertex_cover_size=cover_size,certificate_pass=True,
                               minimum_total_native_score_gap_among_maximum_matchings=True)


def summarise(b,s,g,ds,strict,cal,availability):
    ib,js,cert=certified_match(g,b.cosine_s.to_numpy(),s.cosine_s.to_numpy())
    bb=b.iloc[ib].reset_index(drop=True);ss=s.iloc[js].reset_index(drop=True)
    pairs=pd.DataFrame(dict(bona_uid=bb.utterance_id,spoof_uid=ss.utterance_id,
        bona_s=bb.cosine_s,spoof_s=ss.cosine_s,bona_duration=bb.duration,spoof_duration=ss.duration,
        bona_speaker=bb.speaker_group,spoof_speaker=ss.speaker_group,source=bb.source_corpus,codec=bb.codec))
    if 'transmission' in bb:pairs['transmission']=bb.transmission
    label='strict' if strict else 'relaxed'
    fname=f'{ds}_{label}_cal{cal:g}_{availability}_pairs.parquet';pairs.to_parquet(OUT/fname,index=False)
    supported_b=np.flatnonzero(np.diff(g.indptr)>0);supported_s=np.unique(g.indices)
    groups_b=bb[['source_corpus','speaker_group']].drop_duplicates();groups_s=ss[['source_corpus','speaker_group']].drop_duplicates()
    ans=dict(dataset=ds,matching=label,caliper=cal,availability=availability,n_bona=len(b),n_spoof=len(s),edges=int(g.nnz),
        eligible_bona_with_any_spoof=len(supported_b),eligible_spoof_with_any_bona=len(supported_s),n_pairs=len(pairs),
        matched_bona_speakers=len(groups_b),matched_spoof_speakers=len(groups_s),
        mean_score_gap=float(np.mean(abs(bb.cosine_s-ss.cosine_s))) if len(bb) else None,
        max_score_gap=float(np.max(abs(bb.cosine_s-ss.cosine_s))) if len(bb) else None,
        mean_duration_gap=float(np.mean(abs(bb.duration-ss.duration))) if len(bb) else None,
        pair_file=fname,**cert)
    for source,gp in pairs.groupby('source'):
        SOURCES.append(dict(dataset=ds,matching=label,caliper=cal,availability=availability,source=source,n_pairs=len(gp)))
    return ans,pairs


SOURCES=[]


def main():
    start=time.monotonic();OUT.mkdir(parents=True,exist_ok=False)
    save('scope_lock.json',dict(mode='offline_existing_metadata_and_interventions',no_SSH=True,no_new_forward=True,no_new_model_fit=True,
        primary_question='same native score, comparable observed conditions, differing intervention response?',
        matching='maximum-cardinality, then minimum total absolute native-score gap; min-vertex-cover certificate',
        strict='exact source,codec,length_bin,speaker,and transmission if present;duration_gap<=.25',
        relaxed='same except speaker may differ',calipers=CALIPERS,
        native_selection_only=True,existing_target_inspection='exploratory; no new blind confirmation',
        unresolved=['original-family dependencies','unobserved cross-speaker confounding','no model-fit uncertainty']))
    cat=read(ROOT/'catalog.json');r23=read(ROOT/'r23_catalog.json')
    df=pd.concat([prepare(cat,'21df_pilot'),prepare(cat,'frozen_DF_remaining')],ignore_index=True)
    la=prepare(cat,'21la_eval')
    assert df.y.value_counts().to_dict()=={0:519059,1:14869} and la.y.value_counts().to_dict()=={0:133360,1:14816}
    all_results=[];responses=[];inventory=[];full_sets={}
    for ds,full in [('DF_full',df),('21LA_full',la)]:
        assert not full.utterance_id.duplicated().any() and np.isfinite(full[['cosine_s','duration']]).all().all()
        full_sets[ds]=full
        rname='DF_pilot' if ds=='DF_full' else '21LA_pilot'
        transformed={}
        for cond in ['R2-00']+CONDITIONS:
            entry=r23['entries'][rname+'::'+cond]
            p=ROOT/entry['local_folder']/rname/cond/'scores.parquet';raw=read(p)
            rel=rname+'/'+cond+'/scores.parquet';expected=next(x['sha256'] for x in entry['files'] if x['path']==rel)
            assert HASHES[str(p.resolve())]==expected
            assert len(raw)==20000 and raw.utterance_id.is_unique
            check=raw.merge(full[['utterance_id','y','cosine_s']],on='utterance_id',validate='one_to_one',suffixes=('_raw','_native'))
            assert len(check)==len(raw) and np.array_equal(check.y_raw,check.y_native)
            if cond=='R2-00':assert np.max(abs(check.cosine_s_raw-check.cosine_s_native))<1e-10
            transformed[cond]=raw.set_index('utterance_id')
            inventory.append(dict(dataset=ds,condition=cond,n=len(raw),n_bona=int((raw.y==1).sum()),n_spoof=int((raw.y==0).sum())))
        available=set(transformed['R2-00'].index)
        assert all(set(g.index)==available for g in transformed.values())
        for strict in [True,False]:
            b,s,graph=edges(full,strict)
            print(ds,'strict' if strict else 'relaxed','graph_edges',graph.nnz,flush=True)
            # Tighten score caliper on the fixed graph; never inspect responses.
            erow=np.repeat(np.arange(len(b)),np.diff(graph.indptr));es=np.abs(b.cosine_s.to_numpy()[erow]-s.cosine_s.to_numpy()[graph.indices])
            for cal in CALIPERS:
                g=graph.copy();g.data[es>cal]=0;g.eliminate_zeros()
                ans,pairs=summarise(b,s,g,ds,strict,cal,'all_native');all_results.append(ans)
                # Keep pair list for selective future inference, but do not run it.
                existing_b=pairs.bona_uid.isin(available);existing_s=pairs.spoof_uid.isin(available)
                ans['both_existing_in_this_full_matching']=int((existing_b&existing_s).sum())
                ans['missing_endpoint_forward_count_for_this_matching']=int((~existing_b).sum()+(~existing_s).sum())
                bi=np.flatnonzero(b.utterance_id.isin(available));si=np.flatnonzero(s.utterance_id.isin(available))
                ready=g[bi][:,si].tocsr();br=b.iloc[bi].reset_index(drop=True);sr=s.iloc[si].reset_index(drop=True)
                ansr,pr=summarise(br,sr,ready,ds,strict,cal,'existing_R23');all_results.append(ansr)
                for cond in CONDITIONS:
                    if len(pr)==0:continue
                    raw=transformed[cond];bn=transformed['R2-00']
                    bd=raw.loc[pr.bona_uid,'cosine_s'].to_numpy()-bn.loc[pr.bona_uid,'cosine_s'].to_numpy()
                    sd=raw.loc[pr.spoof_uid,'cosine_s'].to_numpy()-bn.loc[pr.spoof_uid,'cosine_s'].to_numpy()
                    result=dict(dataset=ds,matching='strict' if strict else 'relaxed',caliper=cal,condition=cond,n_pairs=len(pr),
                        mean_bona_delta=float(bd.mean()),mean_abs_bona_delta=float(abs(bd).mean()),mean_spoof_delta=float(sd.mean()),
                        mean_D=float(np.mean(sd-bd)),median_D=float(np.median(sd-bd)),positive_D_fraction=float(np.mean(sd-bd>0)),
                        no_inferential_CI='small support; cross-speaker/family dependencies unresolved; no absence-of-effect claim')
                    responses.append(result)
            del graph,es,erow
        # Score-only neighbor coverage is an upper-support diagnostic, not a
        # controlled comparison; no same-source or duration claims follow.
        bs=np.sort(full.loc[full.y==1,'cosine_s'].to_numpy());ss=np.sort(full.loc[full.y==0,'cosine_s'].to_numpy())
        for cal in CALIPERS:
            eligible=(np.searchsorted(ss,bs+cal,side='right')-np.searchsorted(ss,bs-cal,side='left'))>0
            i=j=n=0
            while i<len(bs) and j<len(ss):
                if bs[i]<ss[j]-cal:i+=1
                elif ss[j]<bs[i]-cal:j+=1
                else:n+=1;i+=1;j+=1
            all_results.append(dict(dataset=ds,matching='score_only_uncontrolled',caliper=cal,availability='all_native',n_bona=len(bs),n_spoof=len(ss),eligible_bona_with_any_spoof=int(eligible.sum()),n_pairs=n))
    # Inventory other existing paired interventions; cannot substitute their
    # scores for R2/R3 scores on a different UID or different condition.
    for rel in ['paired_derived.parquet','frontend_context/results.parquet','pooling_mechanism/results.parquet']:
        f=read(ROOT/'local_analysis/padding_causal_20261106_v1'/rel)
        assert set(f.utterance_id).issubset(set(df.utterance_id))
        inventory.append(dict(dataset='DF_selected471',condition=rel,n_unique=f.utterance_id.nunique(),rows=len(f)))
    pd.DataFrame(all_results).to_csv(OUT/'overlap_and_coverage.csv',index=False)
    pd.DataFrame(responses).to_csv(OUT/'existing_matched_responses.csv',index=False)
    pd.DataFrame(SOURCES).to_csv(OUT/'matched_source_counts.csv',index=False)
    save('summary.json',dict(overlap=all_results,responses=responses,intervention_inventory=inventory))
    assert all(digest(p)==h for p,h in HASHES.items())
    save('verification.json',dict(status='pass',inputs_sha256=HASHES,input_hashes_unchanged=True,
        all_controlled_matchings_have_min_vertex_cover_certificates=True,
        no_original_audio_or_model_access=True,elapsed_seconds=time.monotonic()-start,
        new_GPU_forward=False,new_classifier_fit=False,server_state_checked=False))
    print(pd.DataFrame(all_results)[['dataset','matching','caliper','availability','n_pairs']].to_string(index=False),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
