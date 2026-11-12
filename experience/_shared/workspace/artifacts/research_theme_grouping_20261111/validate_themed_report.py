"""Validate reclassification, preservation, links, tables and core synthesis values."""
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
TARGET=ROOT/'docs/experiments/ebm_five_research_themes_20261111.md'
ORIGINAL=ROOT/'docs/experiments/ebm_completed_evidence_interpretation_20261111.md'
REG=ROOT/'artifacts/experiment_methods_audit_20261111/experiment_registry.json'
def load(p):return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
def rows(p):return list(csv.DictReader((ROOT/p).open(encoding='utf-8-sig')))
checks=[]
def check(name,ok,detail=None):
    checks.append({'name':name,'passed':bool(ok),'detail':detail})
    if not ok:raise AssertionError((name,detail))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

records=json.loads(REG.read_text(encoding='utf-8'))
mapping=load('artifacts/research_theme_grouping_20261111/experiment_theme_map.json')
plan=load('artifacts/research_theme_grouping_20261111/plan.json')
prior=load('artifacts/experiment_methods_audit_20261111/validation.json')
text=TARGET.read_text(encoding='utf-8')
check('original_report_unchanged',sha(ORIGINAL)==plan['original_report_sha256']==prior['report_sha256'])
check('source_registry_unchanged',sha(REG)==plan['source_registry_sha256'])
ids={r['id'] for r in records}
check('all_110_have_one_primary_membership',len(mapping)==len({m['experiment_id'] for m in mapping})==110 and {m['experiment_id'] for m in mapping}==ids)
check('exactly_five_requested_theme_groups',set(m['primary_theme'] for m in mapping)=={1,2,3,4,5})
check('primary_counts_reconcile',Counter(m['primary_theme'] for m in mapping)=={1:42,2:17,3:19,4:4,5:28})
check('secondary_memberships_valid_nonduplicated',all(set(m['secondary_themes'])<={1,2,3,4,5} and m['primary_theme'] not in m['secondary_themes'] and len(m['secondary_themes'])==len(set(m['secondary_themes'])) for m in mapping))

anchors=re.findall(r'<a id="([^"]+)"></a>',text)
check('unique_110_item_and_five_theme_anchors',len(anchors)==len(set(anchors))==115 and set(anchors)=={i.lower() for i in ids}|{f'theme{i}' for i in range(1,6)})
positions=sorted((text.index(f'<a id="{r["id"].lower()}"></a>'),r) for r in records)
by_map={m['experiment_id']:m for m in mapping}
for ix,(start,r) in enumerate(positions):
    end=positions[ix+1][0] if ix+1<len(positions) else len(text)
    section=text[start:end]
    check(f'complete_preserved_method_{r["id"]}',all(v in section for k,v in r.items() if k not in ('id','sources','procedure')) and all(step in section for step in r['procedure']) and all(s in section for s in r['sources']))
    theme_at_start=max((text.rfind(f'<a id="theme{i}"></a>',0,start),i) for i in range(1,6))[1]
    check(f'placed_under_primary_theme_{r["id"]}',theme_at_start==by_map[r['id']]['primary_theme'])

bad_links=[]; local_link_count=0
for target in re.findall(r'\]\(([^)]+)\)',text):
    if target.startswith(('https://','http://')):continue
    local_link_count+=1
    if target.startswith('#'):
        if target[1:] not in anchors:bad_links.append(target)
        continue
    name,_,fragment=target.partition('#')
    file=(TARGET.parent/name).resolve()
    if file==OUT/'validation.json':continue
    if not file.exists():bad_links.append(target);continue
    if fragment and file==ORIGINAL and re.fullmatch(r'e\d{3}',fragment):
        if f'<a id="{fragment}"></a>' not in ORIGINAL.read_text(encoding='utf-8'):bad_links.append(target)
check('all_markdown_links_resolve',not bad_links,{'local_links':local_link_count,'broken':bad_links})
columns=None; table_count=0
for line in text.splitlines():
    if line.startswith('|'):
        n=len(re.findall(r'(?<!\\)\|',line))-1
        if columns is None:columns=n;table_count+=1
        if n!=columns:raise AssertionError(('table cell count',n,columns,line))
    else:columns=None
check('markdown_table_cell_counts_and_norm_pipe_escaping',table_count>=10,table_count)
check('utf8_content_clean','\ufffd' not in text and not re.search('[一-龥ぁ-ヿ]',text))

# Values in the new research summaries are checked against saved numeric results.
full=rows('artifacts/repetition_shortcut_20261111/reviewed_analysis/independent_full_eer.csv')
la=[r for r in full if r['phase']=='21LA']
check('repeat_matrix_four_EER_values',len(la)==4 and all(f'{float(r["eer"]):.6f}%' in text for r in la))
ipf=rows('artifacts/repetition_shortcut_20261111/reviewed_analysis/weighted_eer_21LA.csv')
selected=[r for r in ipf if r['condition']=='native' and r['policy'] in ['common_support_observed','duration_attack_codec_preserved']]
check('length_reweighting_four_values',len(selected)==4 and all(f'{float(r["eer"]):.6f}' in text for r in selected))
oper=rows('artifacts/repetition_shortcut_20261111/reviewed_analysis/operating_points_21LA.csv')
sel=[r for r in oper if r['model']=='legacy240' and r['group']=='le2']
check('inclusive_2sec_class_denominators',all(int(r['n'])==({'1':571,'0':49193}[r['label']]) for r in sel))
cross=rows('artifacts/geometry_cache/local_analysis/bonafide_cross_recheck_20261106_v1/class_replacement_eer.csv')
sel=[r for r in cross if r['baseline']=='19LA_eval']
check('normal_spoof_cross_matrix_four_values',len(sel)==4 and all(f'{float(r["eer_percent"]):.6f}%' in text for r in sel))
values={(r['bona_dataset'],r['spoof_dataset']):float(r['eer_percent']) for r in sel}
b=values['DF_full','19LA_eval']-values['19LA_eval','19LA_eval']
s=values['19LA_eval','DF_full']-values['19LA_eval','19LA_eval']
check('cross_replacement_deltas',f'+{b:.6f}%p' in text and f'−{abs(s):.6f}%p' in text,{'normal_delta_pp':b,'spoof_delta_pp':s})

screen=rows('artifacts/geometry_cache/local_analysis/6ac3505da0708324/metrics.csv')
v={r['case']:r for r in screen if r['dataset']=='DF_pilot'}
check('norm_screen_is_relative_to_correct_baseline',all(abs(float(v[k]['eer_percent'])-e)<1e-10 for k,e in [('C0',17.1),('C1',14.55),('native_cosine',3.35)]) and '17.10→14.55%' in text and '원래 코사인 3.35%보다 높다' in text)
core=rows('artifacts/geometry_cache/local_analysis/cached_core_v1/probe_metrics.csv')
v={r['case']:r for r in core}
for case in ['scalar','direction','noise','metadata','metadata_direction']:
    check(f'residual_table_{case}',f'{float(v[case]["eer_percent"]):.6f}%' in text and f'{float(v[case]["balanced_logloss"]):.6f}' in text)
for case,eer in [('preLN_h',1.4944444444444445),('postLN_z',1.7),('preLN_stats',3.2444444444444445)]:
    check(f'pre_post_LayerNorm_{case}',abs(float(v[case]['eer_percent'])-eer)<1e-10)
for dataset in ['DF_pilot','21LA_pilot']:
    p=f'artifacts/geometry_cache/local_analysis/cache_hypotheses_20261106_v2/H5_{dataset}_saved_readout_decomposition.json'
    d=load(p);m={r['case']:r for r in d['metrics']}
    check(f'scale_fixed_logloss_values_{dataset}',all(f'{m[k]["balanced_logloss"]:.6f}' in text for k in ['saved_h','scale_fixed_to_outer_train_mean']))
    check(f'scale_fixed_metric_directions_{dataset}',m['scale_fixed_to_outer_train_mean']['balanced_logloss']>m['saved_h']['balanced_logloss'] and ((m['scale_fixed_to_outer_train_mean']['eer_percent']>m['saved_h']['eer_percent'])==(dataset=='DF_pilot')))

plain=text.replace(r'\|','|')
check('causal_and_norm_qualifications_retained',all(s in plain for s in ['순수 인과효과는 아직 분리되지 않았다','정상 다양성과 위조 다양성을 학습에서 각각 독립적으로 늘린 실험은 없다','norm 한 값만 추가하는 실험과 동일하지 않다','출처 자체를 통째로 평가에만 남긴 실험은 아니다','ρ=||x−sc||=sqrt(1−s²)']))
check('failure_and_unexecuted_records_retained',all(by_map[e]['status']==next(r['status'] for r in records if r['id']==e) for e in ['E004','E025','E078','E079','E080','E081']))

validation={'status':'passed','checked_at':datetime.now().astimezone().isoformat(),'checks':checks,'check_count':len(checks),
    'method_items':110,'primary_counts':{str(k):v for k,v in Counter(m['primary_theme'] for m in mapping).items()},
    'report_sha256':sha(TARGET),'source_registry_sha256':sha(REG),'original_report_sha256':sha(ORIGINAL),
    'original_report_preserved':True,'all_full_methods_preserved':True,'new_training':False,'new_audio_inference':False,
    'remote_state_queries':False,'new_paid_resources':False,'cost_usd':0,
    'scope':'Reclassification and synthesis of existing evidence; no new causal experiment, model fitting or inference.'}
(OUT/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
plan['status']='complete';plan['validation_check_count']=len(checks)
(OUT/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':'passed','checks':len(checks),'items':110,'counts':validation['primary_counts'],'tables':table_count},ensure_ascii=False))
