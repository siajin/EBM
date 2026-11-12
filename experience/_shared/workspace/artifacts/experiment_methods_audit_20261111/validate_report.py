"""Independent documentation/metadata/numeric audit; no model import or inference."""
import csv
import hashlib
import json
import math
import re
from collections import Counter,defaultdict
from datetime import datetime
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
REPORT=ROOT/'docs/experiments/ebm_completed_evidence_interpretation_20261111.md'
checks=[]
def check(name,condition,detail=None):
    checks.append({'name':name,'passed':bool(condition),'detail':detail})
    if not condition:raise AssertionError((name,detail))
def load(path):return json.loads((ROOT/path).read_text(encoding='utf-8-sig'))

records=load('artifacts/experiment_methods_audit_20261111/experiment_registry.json')
by_id={r['id']:r for r in records}
text=REPORT.read_text(encoding='utf-8')
check('110_contiguous_unique_items',len(records)==len(by_id)==110 and set(by_id)=={f'E{i:03}' for i in range(1,111)})
required={'purpose','model_and_data','procedure','conditions','measurement','observed_result','interpretation_and_limits','sources','status','title'}
check('all_method_fields_populated',all(required<=r.keys() and all(r[k] for k in required) and len(r['procedure'])>=2 for r in records))
check('all_110_body_sections_and_toc',all(text.count(f'<a id="{r["id"].lower()}"></a>')==1 and f'[{r["id"]}](#{r["id"].lower()})' in text for r in records))
refs={x for r in records for k,v in r.items() if k not in ('id','sources') for s in (v if isinstance(v,list) else [v]) for x in re.findall(r'\bE\d{3}\b',s)}
check('internal_item_references_exist',refs<=set(by_id),sorted(refs-set(by_id)))
check('no_replacement_characters_or_non_korean_title', '\ufffd' not in text and not any(re.search('[一-龥ぁ-ヿ]',r['title']) for r in records))

sources=load('artifacts/experiment_methods_audit_20261111/source_hashes.json')
check('all_source_files_still_match_snapshot',all((ROOT/r['path']).is_file() and hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['sha256'] for r in sources),len(sources))
links=re.findall(r'\]\(([^)]+)\)',text)
bad=[]
for link in links:
    if link.startswith(('#','http://','https://')):continue
    if link.endswith('/validation.json'):continue # this file is created by this validation
    target=(REPORT.parent/link.split('#')[0]).resolve()
    if not target.exists():bad.append(link)
check('all_local_report_links_resolve',not bad,{'links':len(links),'broken':bad})
coverage=load('artifacts/experiment_methods_audit_20261111/source_coverage.json')
check('folder_document_and_training_coverage',len(coverage['artifact_directories'])==94 and len(coverage['documents'])==36 and len(coverage['server_training_directories'])==11)
old=load('artifacts/experiment_methods_audit_20261111/historical_aggregate_receipts.json')
check('all_26_historical_server_folders_accounted_for',set(old)==set(coverage['historical_server_analysis_directories']) and len(old)==26)
check('old_empty_folder_not_counted_as_finished_performance',not old['EBM80_21DF_tail_factors']['aggregate_json'] and '빈' in by_id['E090']['procedure'][3])

metrics=load('artifacts/ebm_B_C_checkpoints_20261109/metrics.json')
saved_min=load('artifacts/ebm_B_C_checkpoints_20261109/completed_B_C_dataset_minima.json')
computed_min={}
for dataset in ['19LA','21LA','21DF']:
    computed_min[dataset]={}
    for model in ['B','C']:
        vals=[(metrics[dataset][f'{model}_e{ep}']['EER_percent'],ep) for ep in range(5,51,5)]
        value,ep=min(vals)
        saved=saved_min[dataset][model]
        check(f'minimum_{dataset}_{model}',abs(saved['EER_percent']-value)<1e-12 and ep==saved['epoch'])
        check(f'minimum_in_report_{dataset}_{model}',f'{value:.6f}%, {ep}' in text)
        computed_min[dataset][model]={'eer_percent':value,'epoch':ep,'candidates':10}
for model in ['B','C']:
    for ep in range(5,51,5):
        expected=f'|{model}|{ep}|'+ '|'.join(f'{metrics[d][f"{model}_e{ep}"]["EER_percent"]:.9f}' for d in ['19LA','21LA','21DF'])+'|'
        check(f'all_epoch_table_{model}_{ep}',expected in text)
legacy=load('artifacts/ebm_B_C_checkpoints_20261109/dataset_minimum_comparison_pending_DF.json')['legacy240']
check('legacy_minimum_uses_final_50epoch_LA_record',all(f'{v["EER_percent"]:.6f}%, {v["epoch"]}' in text for v in legacy.values()))

def eer_tie_blocks(rows):
    # Independent ascending tie-block ROC, with score >= threshold meaning normal.
    b=sum(y==1 for _,y in rows); f=len(rows)-b
    accepted_f=f; rejected_b=0
    previous_fpr,previous_fnr,previous_d=1.0,0.0,1.0
    ordered=sorted(rows)
    i=0
    while i<len(ordered):
        score=ordered[i][0]
        while i<len(ordered) and ordered[i][0]==score:
            if ordered[i][1]:rejected_b+=1
            else:accepted_f-=1
            i+=1
        fpr,fnr=accepted_f/f,rejected_b/b
        d=fpr-fnr
        if d<=0:
            w=previous_d/(previous_d-d)
            return 100*(previous_fpr+w*(fpr-previous_fpr))
        previous_fpr,previous_fnr,previous_d=fpr,fnr,d
    raise AssertionError('No EER crossing')

current_file=ROOT/'artifacts/paired_min_eer_20261111/current_eval5_scores.csv'
rows=list(csv.DictReader(current_file.open(encoding='utf-8')))
current=load('artifacts/paired_min_eer_20261111/current_eval5_validation.json')
check('current19LA_rows_labels_indices',len(rows)==71237 and Counter(int(r['label']) for r in rows)=={1:7355,0:63882} and {int(r['index']) for r in rows}==set(range(71237)))
current_eer=eer_tie_blocks([(float(r['score']),int(r['label'])) for r in rows])
check('current19LA_independent_ROC',abs(current_eer-current['eer_percent'])<1e-10,{'recomputed':current_eer,'saved':current['eer_percent']})
check('current19LA_score_hash',hashlib.sha256(current_file.read_bytes()).hexdigest()==current['scores_sha256'])
check('current_model_target_results_not_claimed','21LA·DF 결과' in by_id['E026']['observed_result'] and '없' in by_id['E026']['observed_result'])

probe_rows=[]
for p in (ROOT/'artifacts/repetition_shortcut_20261111/new_results').glob('probe_rank*.csv'):
    probe_rows.extend(csv.DictReader(p.open(encoding='utf-8')))
uids=defaultdict(dict)
for r in probe_rows:
    check_key=(r['uid'],r['condition'])
    if r['condition'] in uids[r['uid']]:raise AssertionError(('duplicate probe',check_key))
    uids[r['uid']][r['condition']]=r
check('336_sources_31_conditions_10416_rows',len(uids)==336 and len(probe_rows)==10416 and all(len(v)==31 for v in uids.values()))
check('probe_original_class_counts',Counter(v['natural4']['label'] for v in uids.values())=={'0':208,'1':128})
check('probe_2sec_repeat2_is_identity',all(v[f'{end}_2s_native']['wave_sha256']==v[f'{end}_2s_repeat2']['wave_sha256'] for v in uids.values() for end in ['head','tail']))

full_by_uid=defaultdict(dict);count=0;length_counts=Counter();exact2=Counter()
for p in (ROOT/'artifacts/repetition_shortcut_20261111/new_results').glob('full_rank*.csv'):
    for r in csv.DictReader(p.open(encoding='utf-8')):
        count+=1;key=(r['phase'],r['uid'])
        if r['condition'] in full_by_uid[key]:raise AssertionError('duplicate full result')
        full_by_uid[key][r['condition']]=r
        if r['phase']=='21LA' and r['condition']=='native':
            n,y=int(r['original_samples']),int(r['label'])
            if n<32000:length_counts[(y,'lt')]+=1
            if n<=32000:length_counts[(y,'le')]+=1
            if n==32000:exact2[y]+=1
        assert all(math.isfinite(float(r[k])) for k in ['score_legacy240','score_B'])
check('173020_sources_two_inputs_346040_rows',count==346040 and len(full_by_uid)==173020 and all(set(v)=={'native','wrap4'} for v in full_by_uid.values()))
noops=[v for v in full_by_uid.values() if int(v['native']['original_samples'])>=64600]
check('27406_long_noops',len(noops)==27406 and all(v['native']['wave_sha256']==v['wrap4']['wave_sha256'] and all(v['native'][k]==v['wrap4'][k] for k in ['score_legacy240','score_B']) for v in noops))
check('strict_vs_inclusive_2sec_denominators',length_counts=={(1,'lt'):507,(1,'le'):571,(0,'lt'):48696,(0,'le'):49193} and exact2=={1:64,0:497},str(dict(length_counts)))
f1=load('artifacts/geometry_cache/local_analysis/frozen_all_v1/F1/masks.json')
check('all_298_coordinate_conditions_explicit',len(f1['original'])==169 and all(len(f1[str(seed)])==43 for seed in [511,512,513]) and sum(map(len,f1.values()))==298)

# Verify the documented conditional loss equals its per-row weighting definition.
max_formula_error=0.0
for n in [1,6,12]:
    for k in range(n+1):
        a=[[((i+1)*(j+3)%17)/13 for j in range(n)] for i in range(n)]
        baseline=sum(map(sum,a))/(n*n)
        if k:
            eligible=sum(sum(row) for row in a[:k])/(k*n)
            b=sum((i+j+2)/19 for i in range(k) for j in range(k))/(k*k)
            formula=baseline-k/(2*n)*eligible+k/(2*n)*b
            expanded=sum((0.5 if i<k else 1)*a[i][j] for i in range(n) for j in range(n))/(n*n)+k/(2*n)*b
        else:formula=expanded=baseline
        max_formula_error=max(max_formula_error,abs(formula-expanded))
check('conditional_loss_matches_documented_row_weights',max_formula_error<1e-14,max_formula_error)

receipt=load('artifacts/experiment_methods_audit_20261111/last_hidden_export_receipt.json')
check('last_hidden_completed_not_pending',receipt['status']['status']=='complete' and receipt['summary']['shape']==[25380,201,1024] and set(receipt['completion_markers'].values())=={'0'})
check('last_hidden_byte_count',sum(v['file_bytes'] for v in receipt['summary']['files'])==25380*201*1024*4+256)
cancel=load('artifacts/paired_min_eer_20261111/queue_cancelled.json')
check('future_extra_evaluation_remains_cancelled',cancel['status']=='cancelled_before_evaluation_launch' and not cancel['new_GPU_evaluation_started'] and cancel['training_untouched'])
check('unexecuted_designs_separated',all(any(w in by_id[x]['status'] for w in ['미실행','보류','취소']) for x in ['E078','E079','E080','E081']))
check('failed_runs_have_no_performance', 'EER 없음' in by_id['E004']['observed_result'] and 'EER도 없다' in by_id['E025']['observed_result'])

historical_numeric={
    'E089_multicenter_mean_EER':{k:v['eer_pct']['mean'] for k,v in old['EBM80_21DF_multicenter']['aggregate_json']['summary.json']['summary'].items()},
    'E100_primary':[{k:r[k] for k in ['family','label','contrast','n','mean','median','toward_n']} for r in old['EBM80_native_quiet_strict']['aggregate_json']['final/native_quiet_strict_summary.json']['primary']],
}
check('old_BPL_replication_completed',bool(old['EBM80_BPL_frozen_audit_20261030_v1']['aggregate_json']['checkpoint_replication/COMPLETED.json']))
check('old_56224_native_10320_input_records',old['ASV_quiet_BPL_distribution_20261030_v1']['aggregate_json']['COMPLETED.json']['native']==56224 and old['ASV_quiet_BPL_distribution_20261030_v1']['aggregate_json']['COMPLETED.json']['pairs']==10320)

result={'status':'passed','checked_at_local':datetime.now().astimezone().isoformat(),
    'method_items':110,'source_files':len(sources),'report_sha256':hashlib.sha256(REPORT.read_bytes()).hexdigest(),
    'checks':checks,'check_count':len(checks),'computed_B_C_minima':computed_min,
    'current19LA_independent_eer_percent':current_eer,'historical_numeric_receipts':historical_numeric,
    'new_training':False,'new_audio_inference':False,'active_training_modified':False,'new_paid_resources':False,
    'scope_limits':['Method documentation and arithmetic/metadata verification; not a rerun of all historical inference or bootstrap fits.',
                    'Some original-family dependencies and historical codec execution remain unresolved.',
                    'Source inventory coverage does not assert every raw file or log line was read anew.']}
(OUT/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':result['status'],'checks':len(checks),'methods':len(records),'sources':len(sources),'current19LA_eer':current_eer},ensure_ascii=False))
