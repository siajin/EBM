"""Descriptive spoof geometry and bona-fide proximity; cached CPU inputs only.

No model fitting, network transfer, GPU import or edits to earlier experiments.
Continuous chord/short-great-circle distances are evaluated in the full space.
"""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '2'

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import time
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from geometry_cache import DEFAULT_ROOT, load_embeddings, validate_run

ROOT = DEFAULT_ROOT / 'local_analysis/spoof_path_geometry_v1'
SEED = 11052026
N_EDGES = 600
REP_NAMES = {'unit_z': '전체 단위 임베딩', 'residual_u': '잔여방향'}
TYPE_NAMES = {'local_1': '최근접', 'local_20': '20번째 이웃',
              'random': '무작위', 'pc1_extremes': '긴 축 양끝', 'same_score_local': '코사인 유사 이웃'}


def save_json(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-14)


def pca(x):
    mean = x.mean(0)
    y = x - mean
    cov = y.T @ y / (len(x) - 1)
    ev, vec = np.linalg.eigh(cov)
    order = np.argsort(ev)[::-1]
    ev, vec = np.maximum(ev[order], 0), vec[:, order].T
    for j in range(len(vec)):
        if vec[j, np.argmax(np.abs(vec[j]))] < 0:
            vec[j] *= -1
    return mean, ev, vec


def continuous_distances(a, b, normals, radii, kind, basis=None):
    """Per-edge min distance and min distance/radius, interior t in [.1,.9].

    For geodesics t is fraction of arc angle; chord t is affine fraction.
    Full data coordinates (or explicitly supplied PCA projection) are used.
    """
    aa, bb, yy = a, b, normals
    if basis is not None:
        aa, bb, yy = a @ basis.T, b @ basis.T, normals @ basis.T
    if kind == 'chord':
        delta = bb - aa
        dd = np.sum(delta * delta, 1)
        ay = aa @ yy.T
        dy = delta @ yy.T
        ad = np.sum(aa * delta, 1)
        tt = np.clip((dy - ad[:, None]) / np.maximum(dd[:, None], 1e-14), .1, .9)
        d2 = (np.sum(aa * aa, 1)[:, None] + np.sum(yy * yy, 1)[None, :]
              - 2 * ay + 2 * tt * (ad[:, None] - dy) + tt * tt * dd[:, None])
    else:
        dot = np.clip(np.sum(a * b, 1), -1, 1)
        angle = np.arccos(dot)
        v = (b - dot[:, None] * a) / np.maximum(np.sin(angle)[:, None], 1e-14)
        ay, vy = a @ normals.T, v @ normals.T
        lo, hi = .1 * angle[:, None], .9 * angle[:, None]
        phase = np.arctan2(vy, ay)
        score = np.maximum(ay * np.cos(lo) + vy * np.sin(lo),
                           ay * np.cos(hi) + vy * np.sin(hi))
        score = np.maximum(score, np.where((phase >= lo) & (phase <= hi), np.hypot(ay, vy), -np.inf))
        d2 = 2 - 2 * score
    d = np.sqrt(np.maximum(d2, 0))
    ratio = d / radii[None, :]
    ix = np.argmin(ratio, 1)
    return d.min(1), ratio[np.arange(len(a)), ix], ix


def endpoint_ratio(a, b, normals, radii):
    d2a = np.maximum(2 - 2 * a @ normals.T, 0)
    d2b = np.maximum(2 - 2 * b @ normals.T, 0)
    return np.minimum((np.sqrt(d2a) / radii).min(1), (np.sqrt(d2b) / radii).min(1))


def edge_list(x, s, pc, ids, rng):
    anchors = rng.choice(len(x), min(N_EDGES, len(x)), replace=False)
    nn = NearestNeighbors(n_neighbors=min(101, len(x)), algorithm='brute', n_jobs=1).fit(x)
    _, idx = nn.kneighbors(x[anchors])
    records = []
    for kind, rank in [('local_1', 1), ('local_20', 20)]:
        for a, candidates in zip(anchors, idx):
            candidates = candidates[candidates != a]
            if len(candidates) >= rank:
                records.append((kind, int(a), int(candidates[rank - 1])))
    for a, candidates in zip(anchors, idx):
        candidates = candidates[(candidates != a) & (np.abs(s[candidates] - s[a]) <= .02)]
        if len(candidates):
            records.append(('same_score_local', int(a), int(candidates[0])))
    for a in anchors:
        b = int(rng.integers(len(x) - 1))
        b += b >= a
        records.append(('random', int(a), b))
    qlo, qhi = np.quantile(pc, [.1, .9])
    left, right = np.flatnonzero(pc <= qlo), np.flatnonzero(pc >= qhi)
    for a, b in zip(rng.choice(left, N_EDGES), rng.choice(right, N_EDGES)):
        records.append(('pc1_extremes', int(a), int(b)))
    out = pd.DataFrame(records, columns=['edge_type', 'a', 'b'])
    out['edge_key'] = [f'{k}:{min(a,b)}:{max(a,b)}' for k, a, b in records]
    out = out.drop_duplicates('edge_key').reset_index(drop=True)
    out['a_row'] = ids[out.a]
    out['b_row'] = ids[out.b]
    out['score_delta'] = np.abs(s[out.a] - s[out.b])
    return out


def verify_math():
    rng = np.random.default_rng(35)
    a, b, y = unit(rng.normal(size=(7, 8))), unit(rng.normal(size=(7, 8))), unit(rng.normal(size=(11, 8)))
    radius = np.linspace(.15, .7, 11)
    errors = {}
    for kind in ['chord', 'arc']:
        _, exact, _ = continuous_distances(a, b, y, radius, kind)
        sampled = []
        for ai, bi in zip(a, b):
            t = np.linspace(.1, .9, 10001)
            if kind == 'chord':
                pts = (1-t[:,None])*ai+t[:,None]*bi
            else:
                w = np.arccos(np.clip(ai @ bi, -1, 1))
                pts = np.sin((1-t[:,None])*w)/np.sin(w)*ai + np.sin(t[:,None]*w)/np.sin(w)*bi
            sampled.append((np.linalg.norm(pts[:,None,:]-y[None,:,:],axis=2)/radius).min())
        errors[kind] = float(np.max(np.abs(exact-sampled)))
        assert errors[kind] < 2e-5
    # Crossing, out-of-plane projection, and endpoints excluded.
    a = unit(np.array([[1.,-1.,0.]])); b = unit(np.array([[1.,1.,0.]]))
    n = unit(np.array([[0.,0.,1.]]))
    _, r, _ = continuous_distances(a,b,n,np.array([.1]),'chord')
    assert r[0] > 10
    return {'status': 'pass', 'dense_10001_point_max_ratio_error': errors}


def run_dataset(alias, index):
    print('LOAD', alias, flush=True)
    source_alias = '21LA_eval' if alias == '21LA_pilot' else alias
    data = load_embeddings(source_alias)
    rows, z = data.rows, data.z
    if alias == '21LA_pilot':
        lock_path = DEFAULT_ROOT / 'analysis_manifests/frozen_data_sizes_v1/lock.json'
        lock = json.loads(lock_path.read_text(encoding='utf-8'))['views'][alias]
        manifest = lock_path.parent / lock['manifest_file']
        assert digest(manifest) == lock['manifest_sha256']
        assert data.provenance['cache_fingerprint'] == lock['source_cache_fingerprint']
        selected = pd.read_parquet(manifest)
        ix = selected.cache_row.to_numpy()
        assert rows.iloc[ix].utterance_id.tolist() == selected.utterance_id.tolist()
        rows, z = rows.iloc[ix].reset_index(drop=True), z[ix]
    y = rows.y.to_numpy()
    assert set(np.unique(y)) == {0, 1} and not rows.utterance_id.duplicated().any()
    x = unit(z.astype('float64'))
    c = data.centroid
    s = x @ c
    r = x - s[:,None] * c
    valid = np.linalg.norm(r, axis=1) >= 1e-4
    assert np.isfinite(x).all() and valid.all()
    u = unit(r)
    assert np.max(np.abs(u @ c)) < 1e-9
    assert np.max(np.abs(s-rows.cosine_s.to_numpy())) < 1e-9
    result = {'dataset': alias, 'n': len(rows), 'spoof_n': int((y==0).sum()), 'bona_n': int((y==1).sum()),
              'cache_fingerprint': data.provenance['cache_fingerprint'],
              'checkpoint_sha256': data.provenance['identity']['checkpoint_sha256'], 'representations': {}}
    plot_data = {}
    all_metrics, all_loadings, all_groups = [], [], []
    for rep, features in [('unit_z', x), ('residual_u', u)]:
        print('ANALYZE', alias, rep, flush=True)
        rng = np.random.default_rng(SEED + index * 17)
        spoof_ids, bona_ids = np.flatnonzero(y==0), np.flatnonzero(y==1)
        sf, bf = features[spoof_ids], features[bona_ids]
        mean, ev, vec = pca(sf)
        frac = ev / ev.sum()
        coordinates = (features - mean) @ vec[:2].T
        within_trace = 0.
        group_records = []
        for name, group in rows.loc[y==0].groupby('attack_id', dropna=False):
            xx = features[group.index.to_numpy()]
            if len(xx) < 2:
                continue
            gg = xx-xx.mean(0)
            within_trace += float(np.sum(gg*gg))
            if len(xx)>=50:
                _, ge, gv = pca(xx)
                group_records.append({'dataset':alias, 'representation':rep, 'attack_id':str(name), 'n':len(xx),
                                      'pc1_variance_fraction':float(ge[0]/ge.sum()),
                                      'pc1_axis_alignment_abs':float(abs(gv[0]@vec[0]))})
        total_ss = float(np.sum((sf-mean)**2))
        shape = {'pc1_variance_fraction':float(frac[0]), 'pc2_variance_fraction':float(frac[1]),
                 'pc12_variance_fraction':float(frac[:2].sum()),
                 'pc1_pc2_axis_ratio':float(np.sqrt(ev[0]/max(ev[1],1e-14))),
                 'effective_rank':float(ev.sum()**2/np.sum(ev**2)),
                 'dimensions_for_90pct':int(np.searchsorted(np.cumsum(frac), .9)+1),
                 'between_attack_trace_fraction':float(1-within_trace/total_ss),
                 'within_attack_group_count_n_ge_50':len(group_records),
                 'within_attack_pc1_fraction_median':float(np.median([g['pc1_variance_fraction'] for g in group_records])) if group_records else None}
        for pc in [0,1]:
            order = np.argsort(vec[pc]**2)[::-1]
            for rank,j in enumerate(order):
                all_loadings.append({'dataset':alias,'representation':rep,'pc':pc+1,'rank':rank+1,
                                     'channel_0based':int(j),'signed_loading':float(vec[pc,j]),
                                     'squared_loading_fraction':float(vec[pc,j]**2)})
        all_groups.extend(group_records)
        # Independent spoof support points are excluded from all edge endpoints.
        perm = rng.permutation(len(sf))
        support_ids = spoof_ids[perm[:len(bf)]]
        pool_ids = spoof_ids[perm[len(bf):]]
        pool = features[pool_ids]
        edge = edge_list(pool, s[pool_ids], (pool-mean)@vec[0], pool_ids, rng)
        aa, bb = features[edge.a_row], features[edge.b_row]
        bona_nn = NearestNeighbors(n_neighbors=6, algorithm='brute',n_jobs=1).fit(bf)
        bd, _ = bona_nn.kneighbors(bf)
        assert np.min(bd[:,5]) > 1e-10
        radius1, radius5 = np.maximum(bd[:,1],1e-10), bd[:,5]
        edge['endpoint_min_bona_radius5_ratio'] = endpoint_ratio(aa,bb,bf,radius5)
        edge['angle_deg'] = np.degrees(np.arccos(np.clip(np.sum(aa*bb,1),-1,1)))
        assert np.max(edge.angle_deg) < 179.99
        for kind in ['chord','arc']:
            distance, ratio, ni = continuous_distances(aa,bb,bf,radius5,kind)
            edge[kind+'_distance'] = distance
            edge[kind+'_radius5_ratio'] = ratio
            edge[kind+'_nearest_bona_row'] = bona_ids[ni]
            _, ratio1, _ = continuous_distances(aa,bb,bf,radius1,kind)
            edge[kind+'_radius1_ratio'] = ratio1
        # Same bona-derived full-space radius in 2D: isolate projection loss.
        _, projected_ratio, _ = continuous_distances(aa,bb,bf,radius5,'chord',basis=vec[:2])
        edge['projected2_chord_radius5_ratio'] = projected_ratio
        midpoint = unit(aa+bb)
        for label, support in [('bona',bf),('spoof',features[support_ids])]:
            distances, _ = NearestNeighbors(n_neighbors=1, algorithm='brute',n_jobs=1).fit(support).kneighbors(midpoint)
            edge['midpoint_nearest_'+label+'_distance'] = distances[:,0]
        edge['midpoint_closer_bona_balanced'] = edge.midpoint_nearest_bona_distance < edge.midpoint_nearest_spoof_distance
        # The full-space location of a direction-only arc with midpoint s held fixed.
        # Endpoints are moved to their mean s; exact arc distance still has a closed form.
        if rep == 'residual_u':
            s0 = (s[edge.a_row]+s[edge.b_row])/2
            q0 = np.sqrt(np.maximum(1-s0*s0,0))
            w = np.arccos(np.clip(np.sum(aa*bb,1),-1,1))
            vv = (bb-np.cos(w)[:,None]*aa)/np.maximum(np.sin(w)[:,None],1e-14)
            ya, yv = aa@x[bona_ids].T, vv@x[bona_ids].T
            phase=np.arctan2(yv,ya)
            lo,hi=.1*w[:,None],.9*w[:,None]
            mx=np.maximum(ya*np.cos(lo)+yv*np.sin(lo),ya*np.cos(hi)+yv*np.sin(hi))
            mx=np.maximum(mx,np.where((phase>=lo)&(phase<=hi),np.hypot(ya,yv),-np.inf))
            score=s0[:,None]*s[bona_ids][None,:]+q0[:,None]*mx
            distance=np.sqrt(np.maximum(2-2*score,0))
            full_bd,_=NearestNeighbors(n_neighbors=6,algorithm='brute',n_jobs=1).fit(x[bona_ids]).kneighbors(x[bona_ids])
            edge['fixed_s_full_arc_radius5_ratio']=(distance/full_bd[:,5]).min(1)
        for col in ['attack_id','source_corpus','codec','speaker_group']:
            if col in rows:
                edge['same_'+col]=rows.iloc[edge.a_row][col].to_numpy()==rows.iloc[edge.b_row][col].to_numpy()
        edge['dataset'],edge['representation']=alias,rep
        edge.to_parquet(ROOT/f'{alias}_{rep}_edges.parquet',index=False)
        for kind, group in edge.groupby('edge_type',sort=False):
            record={'dataset':alias,'representation':rep,'edge_type':kind,'edges':len(group),
                    'angle_median_deg':float(group.angle_deg.median()),
                    'score_delta_median':float(group.score_delta.median()),
                    'same_attack_fraction':float(group.same_attack_id.mean()),
                    'midpoint_closer_bona_balanced_percent':float(100*group.midpoint_closer_bona_balanced.mean())}
            for path in ['chord','arc']:
                for k in [1,5]:
                    record[f'{path}_enters_bona_radius{k}_percent']=float(100*(group[f'{path}_radius{k}_ratio']<=1).mean())
                record[path+'_new_entry_radius5_percent']=float(100*((group[path+'_radius5_ratio']<=1)&(group.endpoint_min_bona_radius5_ratio>1)).mean())
                record[path+'_min_ratio_median']=float(group[path+'_radius5_ratio'].median())
            record['projected_only_chord_overlap_percent']=float(100*((group.projected2_chord_radius5_ratio<=1)&(group.chord_radius5_ratio>1)).mean())
            if rep=='residual_u':
                record['fixed_s_full_arc_enters_bona_radius5_percent']=float(100*(group.fixed_s_full_arc_radius5_ratio<=1).mean())
            all_metrics.append(record)
        for condition_col in ['same_attack_id','same_source_corpus','same_codec','same_speaker_group']:
            if condition_col in edge:
                cond=edge.groupby(['edge_type',condition_col]).agg(edges=('a','size'),arc_hits=('arc_radius5_ratio',lambda xx:int((xx<=1).sum())))
                cond.to_csv(ROOT/f'{alias}_{rep}_{condition_col}.csv')
        result['representations'][rep]={'shape':shape,'bona_5nn_radius_median':float(np.median(radius5)),
                                      'bona_1nn_radius_median':float(np.median(radius1)),
                                      'support_points_per_class':len(bf), 'edge_count':len(edge)}
        plot_data[rep]=(coordinates,vec,frac,edge,features,bona_ids)
    pd.DataFrame(all_loadings).to_csv(ROOT/f'{alias}_channel_loadings.csv',index=False)
    pd.DataFrame(all_groups).to_csv(ROOT/f'{alias}_within_attack_shape.csv',index=False)
    plot_dataset(alias,y,plot_data)
    return result,all_metrics


def plot_dataset(alias,y,plot_data):
    fig,axes=plt.subplots(2,2,figsize=(13,10))
    rng=np.random.default_rng(SEED)
    for row,(rep,(coords,vec,frac,edge,features,bona_ids)) in enumerate(plot_data.items()):
        ax=axes[row,0]
        for label,color,marker,name in [(0,'#D38B32','.', '스푸프'),(1,'#3276A8','x','정상')]:
            ids=np.flatnonzero(y==label)
            ids=rng.choice(ids,min(2200,len(ids)),replace=False)
            ax.scatter(coords[ids,0],coords[ids,1],s=9 if label==0 else 13,alpha=.25 if label==0 else .55,c=color,marker=marker,label=name,rasterized=True)
        ax.set_xlabel(f'스푸프 PC1 ({100*frac[0]:.1f}% 분산)')
        ax.set_ylabel(f'스푸프 PC2 ({100*frac[1]:.1f}% 분산)')
        ax.set_title(REP_NAMES[rep]+'의 스푸프 기준 투영',loc='left',fontweight='bold')
        ax.legend(frameon=False);ax.set_aspect('equal',adjustable='datalim')
        ax=axes[row,1]
        channels=np.argsort(vec[0]**2)[-12:]
        ax.barh(np.arange(len(channels)),vec[0,channels],color=['#3276A8' if v<0 else '#D38B32' for v in vec[0,channels]])
        ax.set_yticks(np.arange(len(channels)),[str(i) for i in channels])
        ax.axvline(0,c='#777777',lw=.8)
        ax.set_xlabel('PC1 계수 · 축의 부호는 임의')
        ax.set_ylabel('채널 번호 (0부터)')
        ax.set_title(REP_NAMES[rep]+'의 긴 축을 만드는 채널',loc='left',fontweight='bold')
    for ax in axes.ravel():
        ax.spines[['top','right']].set_visible(False)
        ax.grid(alpha=.12);ax.set_axisbelow(True)
    fig.suptitle(f'{alias} · 스푸프 구조와 채널 방향',fontsize=17,fontweight='bold')
    fig.text(.02,.01,'PCA는 스푸프만으로 계산 · 정상은 같은 축에 표시 · 2D 겹침은 160차원 경로의 교차를 뜻하지 않음',fontsize=10)
    fig.tight_layout(rect=[0,.04,1,.95]);fig.savefig(ROOT/f'{alias}_shape.png',dpi=170,facecolor='white');plt.close(fig)


def plot_summary(metrics):
    frame=pd.DataFrame(metrics)
    fig,axes=plt.subplots(1,2,figsize=(13,5.5),sharey=True)
    datasets=['T','V','DF_pilot','21LA_pilot']
    kinds=['local_1','local_20','same_score_local','random','pc1_extremes']
    colors=['#3276A8','#D38B32','#71833C','#B45E8B']
    for ax,rep in zip(axes,['unit_z','residual_u']):
        for j,(ds,color) in enumerate(zip(datasets,colors)):
            chosen=frame[(frame.dataset==ds)&(frame.representation==rep)].set_index('edge_type')
            values=[chosen.loc[k,'arc_enters_bona_radius5_percent'] if k in chosen.index else np.nan for k in kinds]
            ax.bar(np.arange(5)+(j-1.5)*.19,values,width=.18,color=color,label=ds)
        ax.set_xticks(np.arange(5),[TYPE_NAMES[k].replace(' ','\n') for k in kinds])
        ax.set_title(REP_NAMES[rep],loc='left',fontweight='bold')
        ax.set_ylim(0,100);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[0].set_ylabel('경로가 정상 주변 영역에 진입한 비율 (%)')
    axes[1].legend(frameon=False,ncol=2)
    fig.suptitle('스푸프 연결 경로 사이에 정상 음성이 있는가?',fontsize=16,fontweight='bold')
    fig.text(.02,.015,'단위구면의 최단 호 · 양끝 10% 제외 · 정상 영역: 각 정상 표본의 5번째 정상 이웃 거리\n표본 경로의 탐색 통계이며 실제 음성 생성 경로나 보간 정답을 증명하지 않음',fontsize=10)
    fig.tight_layout(rect=[0,.10,1,.93]);fig.savefig(ROOT/'path_overlap.png',dpi=180,facecolor='white');plt.close(fig)


def main():
    ROOT.mkdir(parents=True,exist_ok=False)
    start=time.time()
    protocol={'created_utc':datetime.now(timezone.utc).isoformat(),'seed':SEED,'CPU_threads':2,
              'datasets':['T','V','DF_pilot','21LA_pilot'],'representations':['unit_z','residual_u'],
              'edges_per_type_target':N_EDGES,'interior_fraction':[.1,.9],
              'normal_neighborhood':'per bona 1st and 5th other bona Euclidean distances in the same representation',
              'path_types':['Euclidean chord','shortest great-circle arc'],
              'controlled_residual_path':'also reconstruct full unit embedding at mean endpoint cosine; endpoint delta s reported',
              'scope':'descriptive cached geometry; earlier benchmark observations; no independent confirmatory test or causal claim',
              'source_script_sha256':digest(__file__)}
    save_json(ROOT/'protocol.json',protocol)
    verification=verify_math()
    checks=[]
    for alias in ['T','DF_pilot','21LA_eval']:
        checks.append(validate_run(alias,verify_hashes=True))
    verification['cache_checks']=checks
    save_json(ROOT/'verification.json',verification)
    results=[];metrics=[]
    plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':10})
    with threadpool_limits(limits=2):
        for i,alias in enumerate(protocol['datasets']):
            result,mm=run_dataset(alias,i);results.append(result);metrics.extend(mm)
            save_json(ROOT/'progress.json',{'completed_datasets':[r['dataset'] for r in results]})
            pd.DataFrame(metrics).to_csv(ROOT/'path_metrics.csv',index=False)
        plot_summary(metrics)
    summary={'status':'complete_exploratory','elapsed_seconds':time.time()-start,'datasets':results,'path_metrics':metrics}
    save_json(ROOT/'summary.json',summary)
    (ROOT/'complete').write_text('0\n')
    print('COMPLETE',round(time.time()-start,1),str(ROOT),flush=True)


if __name__=='__main__':
    main()
