"""Plan non-overlapping equal-length partitions using duration metadata only.

An original chooses ONE pattern (k equal pieces). No speed change, padding,
concatenation, overlap, or multiple alternate partitions of one recording.
Tail waste must be strictly below 0.5 seconds. Segment grid defaults to 50 ms;
originals shorter than the minimum segment are preserved unchanged.

The fractional relaxation uses Charnes-Cooper normalization for a fixed ratio
of spoof/bona-fide piece counts. Ratio search is bounded and explicit. Integer
plans use group rounding then a fixed-budget MILP polish. Claims of optimality
are limited to the documented model and solver bounds.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import linprog, milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix, csr_matrix, hstack, vstack

SR=16000


@dataclass
class PartitionModel:
    groups: list
    patterns: pd.DataFrame
    G: csr_matrix
    B: csr_matrix
    S: csr_matrix
    N: np.ndarray
    nb: int
    ns: int
    bins: np.ndarray


def candidate_patterns(lo, hi, minimum=8000, grid=800, waste_limit=8000):
    if hi<minimum:
        return [(1,0,lo//grid)]  # preserve each original's true sample count
    assert lo>=minimum
    candidates=[]
    for k in range(1,lo//minimum+1):
        first=max((minimum+grid-1)//grid,(hi-waste_limit+1+k*grid-1)//(k*grid))
        last=lo//(k*grid)
        candidates.extend((k,j*grid,j) for j in range(first,last+1))
    assert candidates, (lo,hi)
    return candidates


def build_model(df,minimum=8000,grid=800):
    df=df.copy()
    df['samples']=np.rint(df.duration.to_numpy()*SR).astype('int64')
    assert np.allclose(df.samples/SR,df.duration,rtol=0,atol=1e-12)
    assert df.utterance_id.is_unique and (df.samples>0).all()
    df['bucket']=df.samples//grid
    groups=[];patterns=[]
    for gid,((label,bucket),f) in enumerate(df.groupby(['y','bucket'],sort=True)):
        lo,hi=int(f.samples.min()),int(f.samples.max())
        candidates=candidate_patterns(lo,hi,minimum,grid)
        ids=[]
        for k,length,hbin in candidates:
            ids.append(len(patterns))
            patterns.append({'gid':gid,'y':int(label),'k':k,'length_samples':length,'bin':hbin})
        groups.append({'gid':gid,'y':int(label),'N':len(f),'lo':lo,'hi':hi,
                       'indices':f.index.to_numpy(),'patterns':ids})
    p=pd.DataFrame(patterns)
    n=len(p);bins=np.sort(p.bin.unique());bmap={b:i for i,b in enumerate(bins)}
    bidx=np.array([bmap[b] for b in p.bin])
    col=np.arange(n)
    G=coo_matrix((np.ones(n),(p.gid,col)),shape=(len(groups),n)).tocsr()
    mask=p.y.to_numpy()==1
    B=coo_matrix((p.k[mask],(bidx[mask],col[mask])),shape=(len(bins),n)).tocsr()
    S=coo_matrix((p.k[~mask],(bidx[~mask],col[~mask])),shape=(len(bins),n)).tocsr()
    return df,PartitionModel(groups,p,G,B,S,np.array([g['N'] for g in groups]),
                            int((df.y==1).sum()),int((df.y==0).sum()),bins)


def relaxation(model,ratio,secondary=False,overlap_floor=None):
    m=model;n=len(m.patterns);j=len(m.bins)
    # Variables: scaled pattern counts x, common normalized bin mass z, s=NB/KB.
    group=hstack([m.G,csr_matrix((len(m.groups),j)),csr_matrix((-m.N/m.nb).reshape(-1,1))])
    totals=vstack([hstack([csr_matrix(m.B.sum(axis=0)),csr_matrix((1,j+1))]),
                   hstack([csr_matrix(m.S.sum(axis=0)),csr_matrix((1,j+1))])])
    eq=vstack([group,totals]).tocsr()
    beq=np.r_[np.zeros(len(m.groups)),1,ratio]
    eye=csr_matrix(np.eye(j))
    ub=vstack([hstack([-m.B,eye,csr_matrix((j,1))]),
               hstack([-m.S/ratio,eye,csr_matrix((j,1))])]).tocsr()
    bub=np.zeros(2*j)
    objective=np.r_[np.zeros(n),-np.ones(j),0.]
    if secondary:
        ub=vstack([ub,csr_matrix(np.r_[np.zeros(n),-np.ones(j),0.].reshape(1,-1))])
        bub=np.r_[bub,-overlap_floor]
        objective=np.r_[np.zeros(n+j),-1.]
    result=linprog(objective,A_ub=ub,b_ub=bub,A_eq=eq,b_eq=beq,
                   bounds=[(0,None)]*(n+j)+[(0,1)],method='highs')
    if not result.success:return None
    s=result.x[-1]
    return {'overlap':float(result.x[n:n+j].sum()),'scale':float(s),
            'allocation':result.x[:n]*m.nb/s,'ratio':float(ratio),
            'equality_residual':float(np.max(np.abs(eq@result.x-beq)))}


def overlap(model,allocation):
    b=np.asarray(model.B@allocation).ravel();s=np.asarray(model.S@allocation).ravel()
    return float(np.minimum(b/b.sum(),s/s.sum()).sum()),b,s


def round_groups(model,allocation):
    out=np.zeros(len(allocation),dtype=np.int64)
    for group in model.groups:
        ix=np.asarray(group['patterns'])
        a=np.maximum(allocation[ix],0)
        a=a/a.sum()*group['N']
        counts=np.floor(a+1e-9).astype('int64')
        left=group['N']-counts.sum()
        assert 0<=left<=len(ix)
        counts[np.argsort(-(a-counts),kind='stable')[:left]]+=1
        out[ix]=counts
    assert np.array_equal(model.G@out,model.N)
    return out


def polish(model,initial,seconds=30):
    m=model;n=len(m.patterns);j=len(m.bins)
    before,b,s=overlap(m,initial)
    kb,ks=int(b.sum()),int(s.sum());ratio=ks/kb
    eye=csr_matrix(np.eye(j))
    eq=vstack([hstack([m.G,csr_matrix((len(m.groups),j))]),
               hstack([csr_matrix(m.B.sum(axis=0)),csr_matrix((1,j))]),
               hstack([csr_matrix(m.S.sum(axis=0)),csr_matrix((1,j))])]).tocsr()
    beq=np.r_[m.N,kb,ks]
    ub=vstack([hstack([-m.B,eye]),hstack([-m.S/ratio,eye])]).tocsr()
    c=np.r_[np.zeros(n),-np.ones(j)/kb]
    limits=np.r_[m.N[m.patterns.gid.to_numpy()],np.full(j,np.inf)]
    result=milp(c,integrality=np.r_[np.ones(n),np.zeros(j)],
                bounds=Bounds(np.zeros(n+j),limits),
                constraints=[LinearConstraint(eq,beq,beq),LinearConstraint(ub,-np.inf,0)],
                options={'time_limit':seconds,'mip_rel_gap':1e-6,'presolve':True})
    record={'status':int(result.status),'message':str(result.message),
            'initial_overlap':before,'fixed_bona_pieces':kb,'fixed_spoof_pieces':ks}
    for name in ['mip_gap','mip_node_count','mip_dual_bound']:
        value=getattr(result,name,None)
        record[name]=None if value is None else float(value)
    chosen=initial
    if result.x is not None:
        candidate=np.rint(result.x[:n]).astype('int64')
        assert np.array_equal(m.G@candidate,m.N)
        ov,bb,ss=overlap(m,candidate)
        assert int(bb.sum())==kb and int(ss.sum())==ks
        if ov>before:chosen=candidate
    record['final_overlap']=overlap(m,chosen)[0]
    if record['mip_dual_bound'] is not None:
        record['fixed_budget_upper_overlap']=min(1.,-record['mip_dual_bound'])
    return chosen,record


def materialize(df,m,allocation,grid=800):
    records=[]
    for group in m.groups:
        f=df.loc[group['indices']].sort_values('utterance_id')
        # UID hashing removes a file-order preference in assigning valid patterns.
        order=np.argsort([hashlib.sha256(str(u).encode()).hexdigest() for u in f.utterance_id])
        f=f.iloc[order];position=0
        for pi in group['patterns']:
            count=int(allocation[pi])
            if not count:continue
            p=m.patterns.iloc[pi]
            selected=f.iloc[position:position+count];position+=count
            for row in selected.itertuples():
                length=int(p.length_samples) or int(row.samples)
                k=int(p.k);waste=int(row.samples)-k*length
                assert 0<=waste<8000
                assert length>=8000 or (row.samples<8000 and k==1 and waste==0)
                records.append({'utterance_id':row.utterance_id,'y':int(row.y),
                    'original_samples':int(row.samples),'pieces':k,'piece_samples':length,
                    'discarded_samples':waste,'histogram_bin':int(p.bin)})
        assert position==len(f)
    out=pd.DataFrame(records)
    assert len(out)==len(df) and out.utterance_id.is_unique
    assert set(out.utterance_id)==set(df.utterance_id)
    assert (out.pieces*out.piece_samples+out.discarded_samples==out.original_samples).all()
    return out


def segment_boundaries(original_samples,pieces,piece_samples):
    """Yield [start,end) sample intervals; all are disjoint, in source order."""
    assert 0<=original_samples-pieces*piece_samples<8000
    for k in range(pieces):yield k*piece_samples,(k+1)*piece_samples


def self_check():
    twelve=12*SR
    p=candidate_patterns(twelve,twelve)
    assert all((k,L,int(L//800)) in p for k,L in [(3,4*SR),(4,3*SR),(2,6*SR)])
    assert (1,twelve-8000,(twelve-8000)//800) not in p  # strictly less than .5s
    for samples in [6400,8000,8001,15999,16000,193200,468992]:
        for k,L,_ in candidate_patterns(samples,samples):
            L=L or samples
            bounds=list(segment_boundaries(samples,k,L))
            assert sum(e-s for s,e in bounds)==k*L
            assert all(bounds[i][1]==bounds[i+1][0] for i in range(len(bounds)-1))
    print('self-check passed',flush=True)


def run(output,limit=None,milp_seconds=30):
    root=Path(__file__).resolve().parents[2]
    import sys
    sys.path.insert(0,str(root/'artifacts/duration_matching_20261109'))
    from analyze import load
    datasets,provenance=load()
    output.mkdir(parents=True,exist_ok=True)
    summary=[];histograms=[];details={}
    for di,(key,title,source) in enumerate(datasets):
        if limit is not None and di>=limit:break
        start=time.time()
        df,m=build_model(source.reset_index(drop=True))
        tb=float(df.loc[df.y==1,'duration'].sum());ts=float(df.loc[df.y==0,'duration'].sum())
        low=(ts-.5*m.ns)/tb;high=ts/(tb-.5*m.nb)
        ratios=np.unique(np.r_[np.linspace(low,high,9),ts/tb])
        results=[]
        for ratio in ratios:
            result=relaxation(m,float(ratio))
            if result is not None:results.append(result)
        assert results
        best=max(results,key=lambda r:(r['overlap'],r['scale']))
        # Refine the ratio locally; this is an explicit search, not a global proof.
        span=(high-low)/8
        for ratio in np.linspace(max(low,best['ratio']-span),min(high,best['ratio']+span),5):
            result=relaxation(m,float(ratio))
            if result is not None:results.append(result)
        best=max(results,key=lambda r:(r['overlap'],r['scale']))
        secondary=relaxation(m,best['ratio'],secondary=True,overlap_floor=best['overlap']-1e-8)
        if secondary is not None:best=secondary
        allocation=round_groups(m,best['allocation'])
        rounded_overlap=overlap(m,allocation)[0]
        allocation,solver=polish(m,allocation,milp_seconds)
        plan=materialize(df,m,allocation)
        plan.to_parquet(output/f'{key}_partition_plan.parquet',index=False)
        pd.DataFrame(m.patterns).assign(assigned_originals=allocation).to_csv(output/f'{key}_patterns.csv',index=False)
        actual,b,s=overlap(m,allocation)
        row={'dataset':key,'original_bonafide':m.nb,'original_spoof':m.ns,
             'pieces_bonafide':int(b.sum()),'pieces_spoof':int(s.sum()),
             'overlap_pct':100*actual,'fractional_best_overlap_pct':100*best['overlap'],
             'rounded_overlap_pct':100*rounded_overlap,
             'original_total_seconds':float(plan.original_samples.sum()/SR),
             'discarded_seconds':float(plan.discarded_samples.sum()/SR),
             'discarded_audio_pct':float(100*plan.discarded_samples.sum()/plan.original_samples.sum()),
             'mean_discard_per_original_s':float(plan.discarded_samples.mean()/SR),
             'max_discard_per_original_s':float(plan.discarded_samples.max()/SR),
             'pieces_per_original':float(plan.pieces.sum()/len(plan)),
             'mean_piece_seconds':float((plan.original_samples.sum()-plan.discarded_samples.sum())/plan.pieces.sum()/SR),
             'short_originals_preserved':int((plan.original_samples<8000).sum()),
             'one_piece_originals_pct':float(100*(plan.pieces==1).mean()),
             'runtime_s':time.time()-start}
        for cls in [1,0]:
            sub=plan[plan.y==cls]
            row[f'class{cls}_discarded_audio_pct']=float(100*sub.discarded_samples.sum()/sub.original_samples.sum())
            row[f'class{cls}_mean_piece_seconds']=float((sub.original_samples.sum()-sub.discarded_samples.sum())/sub.pieces.sum()/SR)
        summary.append(row)
        for stage in ['before','after']:
            for cls,arr in [(1,b),(0,s)]:
                if stage=='before':
                    orig=df.loc[df.y==cls,'bucket'].value_counts()
                    counts=np.array([orig.get(int(j),0) for j in m.bins])
                else:counts=arr
                for hb,count in zip(m.bins,counts):
                    histograms.append({'dataset':key,'stage':stage,'y':cls,'bin_left_s':hb*.05,
                        'count':float(count),'percent':float(100*count/counts.sum())})
        # Independently reconstruct produced histogram from every original plan.
        for cls,arr in [(1,b),(0,s)]:
            counted=plan[plan.y==cls].groupby('histogram_bin').pieces.sum()
            assert np.array_equal(np.array([counted.get(int(j),0) for j in m.bins]),arr)
        details[key]={'ratio_search':[{'ratio':r['ratio'],'overlap':r['overlap'],'scale':r['scale']} for r in results],
                      'fractional_solution_equality_residual':best['equality_residual'],
                      'solver':solver,'all_originals_preserved_once':True,
                      'all_waste_strictly_less_than_0_5_seconds':True,
                      'all_generated_pieces_at_least_0_5_seconds':True,
                      'short_originals_preserved_unchanged':True,
                      'independent_histogram_reconstruction':True}
        pd.DataFrame(summary).to_csv(output/'summary.csv',index=False)
        pd.DataFrame(histograms).to_csv(output/'histograms.csv',index=False)
        (output/'validation.json').write_text(json.dumps({'sources':provenance,'datasets':details,
            'minimum_segment_seconds':.5,'grid_seconds':.05,'tail_loss_seconds_strict_upper':.5,
            'model':'One equal-length partition per original; no overlap, speed, padding, cross-file join or source reuse.',
            'short_original_exception':'Originals below .5s remain unchanged.',
            'normalization':'Piece counts within each class. Source recording weights are no longer preserved because all disjoint pieces are used.',
            'scope':'Each dataset separately; equal partitions on a .05s grid, conservative original-length groups; finite ratio search. Integer plans are feasible lower bounds on the unrestricted mixed-length partition optimum.',
            'no_audio_processed':True},indent=2),encoding='utf-8')
        print(json.dumps(row),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[2]/'artifacts/duration_partitions_20261109')
    parser.add_argument('--limit',type=int)
    parser.add_argument('--milp-seconds',type=int,default=30)
    parser.add_argument('--self-check',action='store_true')
    args=parser.parse_args()
    if args.self_check:self_check()
    else:run(args.output,args.limit,args.milp_seconds)
