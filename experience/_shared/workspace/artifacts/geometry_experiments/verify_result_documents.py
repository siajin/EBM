"""Verify displayed scientific claims against saved, independently audited data."""
from pathlib import Path
import json
import re
import hashlib
import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits
from audit_cached_core_results import independent_eer,independent_auc

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'artifacts/geometry_cache/local_analysis/frozen_all_v1';DOCS=ROOT/'docs/experiments';REV=OUT/'analysis_md_revision_v1'
CAL_A,CAL_B=8.710543812499347,-0.7151751091389169

def read(path):return json.loads(path.read_text(encoding='utf8'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ll(y,p):
    p=np.clip(p,1e-7,1-1e-7)
    return float(-.5*(np.log(p[y==1]).mean()+np.log1p(-p[y==0]).mean()))

def main():
    checks=[];sources=set();main_doc=DOCS/'ebm_frozen_all_results_20261105.md';text=main_doc.read_text(encoding='utf8');metric_checks=0
    def check(name,ok,**details):
        assert ok,(name,details)
        checks.append(dict(name=name,status='pass',**details))
    hmetrics=pd.read_csv(OUT/'head_metrics.csv');sources.add(OUT/'head_metrics.csv')
    contrast=read(OUT/'head_paired_effects.json');sources.add(OUT/'head_paired_effects.json')
    baseline=read(OUT/'native_baselines.json');sources.add(OUT/'native_baselines.json')
    for dataset in ['DF_pilot','21LA_pilot']:
        saved=read(OUT/'verification.json');check('completed_head_source_audit',saved['status']=='pass_within_completed_scope')
        predictions={}
        for seed in [101,102,103]:
            path=OUT/'P1'/str(seed)/'dynamic'/(dataset+'_scores.parquet');f=pd.read_parquet(path);sources.add(path);predictions[seed]=f.score.to_numpy();y=f.y.to_numpy()
            row=hmetrics[(hmetrics.dataset==dataset)&(hmetrics.model=='dynamic')&(hmetrics.seed==seed)].iloc[0]
            check('dynamic_independent_EER',abs(independent_eer(y,f.score.to_numpy())-row.eer_percent)<1e-9,dataset=dataset,seed=seed)
            check('dynamic_independent_AUC',abs(independent_auc(y,f.score.to_numpy())-row.auroc)<1e-12)
            check('dynamic_independent_balanced_LL',abs(ll(y,expit(CAL_A*f.score.to_numpy()+CAL_B))-row.balanced_logloss)<1e-12);metric_checks+=3
        selected=hmetrics[(hmetrics.dataset==dataset)&(hmetrics.model=='dynamic')]
        expected=f'| {dataset} | dynamic | {selected.eer_percent.mean():.6f} | {selected.auroc.mean():.9f} | {selected.balanced_logloss.mean():.6f} |'
        check('displayed_dynamic_seed_mean',expected in text,expected_row=expected)
        path=OUT/(dataset+'_head_bootstrap.npz');sources.add(path)
        with np.load(path) as boot:
            delta=sum(boot['native']-boot[f'dynamic_{seed}'] for seed in [101,102,103])/3
            calculated=np.nanquantile(delta,[.005,.995])
            rec=next(r for r in contrast if r['dataset']==dataset and r['baseline']=='native' and r['candidate']=='dynamic')
            check('documented99_CI_seed_mean_not_ensemble',np.max(np.abs(calculated-rec['CI99']))<1e-12)
        p=OUT/'P3_P4/A'/dataset/'predictions.parquet';pred=pd.read_parquet(p);sources.add(p)
        v=read(OUT/'P3_P4/A/verification.json');check('P3_transfer_audited',v['status']=='pass')
        reductions=[];eers=[]
        for seed in [101,102,103]:
            base=pred[f'scalar_{seed}'].to_numpy();direction=pred[f'direction_linear_{seed}'].to_numpy();yy=pred.y.to_numpy()
            reductions.append(ll(yy,expit(base))-ll(yy,expit(direction)));eers.append(independent_eer(yy,base)-independent_eer(yy,direction));metric_checks+=2
        path=OUT/'P3_P4/A'/dataset/'paired_effects.json';sources.add(path);rec=next(r for r in read(path) if r['baseline']=='scalar' and r['candidate']=='direction_linear')
        check('P3_logloss_effect_not_EER_improvement',abs(np.mean(reductions)-rec['mean_balanced_LL_reduction'])<1e-12 and abs(np.mean(eers)-rec['mean_delta_EER_pp'])<1e-9)
        check('P3_displayed_LL_effect',f'{rec["mean_balanced_LL_reduction"]:.9f}' in text)
    f2=read(OUT/'F2/summary.json');sources.add(OUT/'F2/summary.json')
    for r in f2:
        check('matching_inconclusive_not_information_absent',r['pairs']<200 and r['status']=='inconclusive_low_overlap')
        check('matching_counts_in_main',f"| {r['dataset']} | {r['condition']} | {r['pairs']} | {r['speaker_groups']} |" in text)
    reg=read(OUT/'F3/metric_registry.json');check('V_selected_covariance_identity',reg['selected']=={'diagonal':'diagonal_1.0','full':'full_1.0'});sources.add(OUT/'F3/metric_registry.json')
    effect=next(r for r in read(OUT/'F3/paired_effects.json') if r['dataset']=='DF_pilot' and r['model']=='diagonal_0.01');sources.add(OUT/'F3/paired_effects.json')
    check('F3_pointwise_vs_adjusted_conclusion',effect['CI95'][0]>0 and effect['CI_adjusted'][0]<=0<=effect['CI_adjusted'][1])
    path=OUT/'F4/vector_variance.csv';eta=pd.read_csv(path);sources.add(path)
    for r in eta[eta.stratum=='all'].itertuples():
        expected=f'| {r.dataset} | {"정상" if r.y else "spoof"} | {r.factor} | {r.eta_squared:.6f} |'
        check('F4_full_vector_eta_display',expected in text)
    check('multiple_comparison_scope', '두 셋의10개 비교 전체를 같은95% family로 통제했다고 주장하지 않는다' in text)
    check('native_and_probes_distinguished','기존 DF 내부' in text and '새 T/V 전이' in text)
    check('current_incomplete_work_not_claimed_complete','완료된 결과에 한정한 중간 분석' in text and '아직 최종 결과가 없다' in text)
    check('new_BPL_not_replayed','BPL은 정상 long/short에만 적용' in text)
    check('original_model_uses_angle_and_channels','방향이나 채널을 전혀 사용하지 않는다는 설명은 틀리다' in text)
    total_links=images=0
    for p in DOCS.rglob('*.md'):
        content=p.read_text(encoding='utf8');check('UTF8_and_fences','\ufffd' not in content and sum(l.startswith('```') for l in content.splitlines())%2==0,path=str(p.relative_to(ROOT)))
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',content):
            if '://' in target or target.startswith('#'):continue
            check('local_evidence_link',(p.parent/target.split('#',1)[0]).resolve().exists(),document=p.name,target=target);total_links+=1
        current_images=re.findall(r'!\[[^\]]*\]\(([^)]+)\)',content);images+=len(current_images)
        before=REV/'before'/p.relative_to(DOCS)
        if before.exists():check('working_figure_embeds_preserved',current_images==re.findall(r'!\[[^\]]*\]\(([^)]+)\)',before.read_text(encoding='utf8')))
    for p in [*DOCS.rglob('*.md'),*Path(__file__).parent.glob('*frozen*.py')]:
        if p.suffix=='.py':
            import ast
            ast.parse(p.read_text(encoding='utf8'),filename=str(p))
    verification=dict(status='pass_within_current_completed_evidence',checks=len(checks),independent_metric_or_effect_recomputations=metric_checks,Markdown_documents=len(list(DOCS.rglob('*.md'))),local_links=total_links,preserved_image_embeds=images,new_training_or_audio_inference=False,all_experiments_complete=False,source_sha256={str(p.relative_to(ROOT)):digest(p) for p in sorted(sources)},document_sha256={str(p.relative_to(ROOT)):digest(p) for p in sorted(DOCS.rglob('*.md'))},details=checks)
    (REV/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in verification.items() if k not in ['details','source_sha256','document_sha256']},ensure_ascii=False))

if __name__=='__main__':
    with threadpool_limits(2):main()
