"""Source-bound checks for the duration research narrative; no inference."""
import csv
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
REPORT=ROOT/'docs/experiments/ebm_duration_bias_research_20261111.md'
text=REPORT.read_text(encoding='utf-8')
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
def rows(p):return list(csv.DictReader((ROOT/p).open(encoding='utf-8-sig')))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
checks=[]
def check(name,ok,detail=None):
    checks.append({'name':name,'passed':bool(ok),'detail':detail})
    if not ok:raise AssertionError((name,detail))
def fmt(x,d=6):return f'{float(x):.{d}f}'
def signed(x,d=4):return f'{float(x):+.{d}f}'.replace('-','−')

plan=read('artifacts/duration_bias_research_20261111/plan.json')
for p,h in plan['preserved_documents'].items():check('preserved_'+Path(p).name,sha(ROOT/p)==h)
old=read('artifacts/experiment_methods_audit_20261111/validation.json')
themed=read('artifacts/research_theme_grouping_20261111/validation.json')
check('old_documents_match_previous_validations',sha(ROOT/'docs/experiments/ebm_completed_evidence_interpretation_20261111.md')==old['report_sha256'] and sha(ROOT/'docs/experiments/ebm_five_research_themes_20261111.md')==themed['report_sha256'])
manifest=read('artifacts/duration_bias_research_20261111/sources.json')
check('all_sources_exist_and_file_hashes_match',all((ROOT/s['path']).exists() and (s['kind']=='directory' or sha(ROOT/s['path'])==s['sha256']) for s in manifest),len(manifest))
bad=[];link_count=0
for ref in re.findall(r'\]\(([^)]+)\)',text):
    if ref.startswith(('http://','https://','#')):continue
    link_count+=1
    p=(REPORT.parent/ref.split('#')[0]).resolve()
    if p==OUT/'validation.json':continue
    if not p.exists():bad.append(ref)
check('all_document_links_resolve',not bad,{'count':link_count,'broken':bad})
columns=None;tables=0
for line in text.splitlines():
    if line.startswith('|'):
        n=len(re.findall(r'(?<!\\)\|',line))-1
        if columns is None:columns=n;tables+=1
        assert n==columns,('table columns',line)
    else:columns=None
check('markdown_tables_well_formed',tables>=10,tables)
check('readable_utf8_no_private_experiment_ids','\ufffd' not in text and not re.search(r'\bE\d{3}\b',text))
check('logical_story_order',all(text.index(a)<text.index(b) for a,b in zip(['## 2.','## 3.','## 4.','## 5.','## 6.','## 7.','## 8.','## 9.'],['## 3.','## 4.','## 5.','## 6.','## 7.','## 8.','## 9.','## 10.'])))

geometry=read('artifacts/geometry_cache/local_analysis/B_final_analysis_20261108_v1/analysis/geometry_statistics.json')
for phase in ['21LA','21DF']:
    for y in [1,0]:
        r=next(r for r in geometry if r['model']=='legacy240' and r['phase']==phase and r['label']==y and r['length_bin']=='under2')
        line='|'+ '|'.join([phase,'정상' if y else '위조',str(r['N']),fmt(r['native_mean_score'],4),fmt(r['wrap_mean_score'],4),signed(r['wrap_mean_score']-r['native_mean_score'])])+'|'
        check(f'same_file_shift_{phase}_{y}',line in text)
        values=[next(r for r in geometry if r['model']==m and r['phase']==phase and r['label']==y and r['length_bin']=='under2') for m in ['legacy240','B']]
        line='|'+ '|'.join([phase,'정상' if y else '위조',fmt(values[0]['mean_absolute_score_change'],4),fmt(values[1]['mean_absolute_score_change'],4)])+'|'
        check(f'nonuniform_sensitivity_change_{phase}_{y}',line in text)
dose=rows('artifacts/repetition_shortcut_20261111/reviewed_analysis/dose_effects.csv')
for length in ['1','2']:
    for treatment,label in [('repeat2','2초'),('repeat3','3초'),('repeat4','4.0375초')]:
        v=[next(r for r in dose if r['model']=='legacy240' and r['label']==str(y) and r['crop_seconds']==length and r['treatment']==treatment and r['control']=='native') for y in [1,0]]
        line='|'+ '|'.join([length+'초',label,signed(v[0]['mean_delta']),signed(v[1]['mean_delta'])])+'|'
        check(f'crop_repeat_dose_{length}_{treatment}',line in text)
dup=read('artifacts/geometry_cache/local_analysis/padding_causal_20261106_v1/pooling_mechanism/summary.json')['duplication']
check('feature_duplication_effect_sizes',all(fmt(dup[k]['mean_abs_score_change']) in text for k in ['2','4']))
front=rows('artifacts/geometry_cache/local_analysis/padding_causal_20261106_v1/frontend_context/group_effects.csv')
selected=[r for r in front if r['source']=='vcc2018' and r['y']=='1' and r['duration_bin']!='long']
check('fixed_201_frame_context_angles',sum(int(r['N']) for r in selected)==80 and all(fmt(sum(float(r[k])*int(r['N']) for r in selected)/80,3)+'°' in text for k in ['mask0_score_mean_angle','mask7_score_mean_angle']))

train=rows('artifacts/repetition_shortcut_20261111/training_original_lengths.csv')
distribution=rows('artifacts/repetition_shortcut_20261111/original_length_distribution.csv')
for y in [1,0]:
    r=next(r for r in train if r['threshold_samples']=='32000' and r['label']==str(y))
    check(f'train_inclusive_2sec_{y}',f"{int(r['le']):,}/{int(r['n']):,}" in text and fmt(r['le_percent'],2)+'%' in text)
for phase in ['dev','19LA','21LA','21DF']:
    for y in [1,0]:
        r=next(r for r in distribution if r['phase']==phase and r['label']==str(y))
        check(f'class_length_distribution_{phase}_{y}',f"{int(r['le2']):,}/{int(r['n']):,}" in text and fmt(r['le2_percent'],2)+'%' in text)
interp=read('artifacts/repetition_shortcut_20261111/reviewed_analysis/interpretation.json')
check('seam_rescue_percentages',all(fmt(100*r['fraction_of_mean_score_drop_recovered'],2)+'%' in text for r in interp['key_results']['seam_smoothing_rescue']))

metrics=read('artifacts/ebm_B_C_checkpoints_20261109/metrics.json')
for model in ['B','C']:
    for d in ['19LA','21LA','21DF']:
        value,ep=min((metrics[d][f'{model}_e{e}']['EER_percent'],e) for e in range(5,51,5))
        check(f'observed_minimum_{model}_{d}',f'{value:.6f}% ({ep}epoch)' in text)
comparison=read('artifacts/ebm_coverage_e20_eval_20261110/comparison.json')
for r in comparison['fixed50_reference']:
    line='|'+ '|'.join([r['phase']]+[fmt(r[f'{m}_EER_percent'])+'%' for m in ['legacy240','B','C']])+'|'
    check('common_nonrepeat_50epoch_'+r['phase'],line in text)
for r in comparison['same_epoch_comparison']:
    line='|'+ '|'.join([r['phase']]+[fmt(r[f'{m}_EER_percent'])+'%' for m in ['B','C','coverage']])+'|'
    check('coverage20_not_mislabeled_as_minimum_'+r['phase'],line in text)
check('minima_and_coverage_selection_limits_explicit',all(v in text for v in ['15~50epoch의 8개','5~50epoch의 10개','최소값 표가 아니라 같은 20epoch','두 자료의 여러 epoch 최소값은 확보하지 않았다']))
c=read('artifacts/ebm_C_20261108/metadata_validation.json')
check('C_is_training_long_cap_ablation',c['sampler_all50epochs_exactly_matches_B'] and c['evaluation_fixed_B_input'] and c['noaug_short_exactly_matches_B'] and c['class_length_counts']['1']['longer_than_B_cap']==576 and c['class_length_counts']['0']['longer_than_B_cap']==6195)
check('C_context_result_directions',all(metrics[d]['B_e50']['EER_percent']>metrics[d]['C_e50']['EER_percent'] for d in ['19LA']) and all(metrics[d]['B_e50']['EER_percent']<metrics[d]['C_e50']['EER_percent'] for d in ['21LA','21DF']))
check('short_view_policy_80percent',sum([.10,.20,.30,.20])==.8 and '目標' not in text and '목표 80%가 2초 이하' in text)

short=read('artifacts/mild_duration_data_analysis_20261111/shortness_comparison.json')['2.0']
check('mild_inclusive_boundary_percentages',all(fmt(short[str(y)]['expected_le_percent'],4)+'%' in text for y in [1,0]))
activity=rows('artifacts/train_crop_io_activity_20261110/results/activity_by_length_role.csv')
check('early_activity_content_not_final_policy',all(fmt(100*float(next(r for r in activity if r['role']==role and r['crop_seconds']=='0.5')['speech_below100ms_fraction']),2)+'%' in text for role in ['OC_bona','OC_spoof']) and '최종 학습에도 이 비율이 그대로 남았다는 뜻은 아니다' in text)
check('failed_and_ongoing_results_not_fabricated',all(s in text for s in ['완료 EER가 없다','첫 5epoch 19LA EER','최종 21LA·DF 결과는 없다','두 효과가 독립적으로 분리된 구성은 아니다']))
check('causal_and_input_limits_explicit',all(s in text for s in ['기준 모델과 B 차이를 반복 학습 하나의 효과로만 해석하지 않는다','4초 상한 제거 탓으로 볼 수 없다','평가 길이·공격·코덱 조정 후에도 B/C 대비 열세가 남은','현재 학습을 수정·중단하거나 취소한 평가를 재등록하라는 뜻이 아니다']))

result={'status':'passed','checked_at':datetime.now().astimezone().isoformat(),'checks':checks,'check_count':len(checks),
        'report_sha256':sha(REPORT),'table_count':tables,'source_manifest_count':len(manifest),
        'corrections':{'mild_boundary':'Used <=2s 8.9123/8.9386 from shortness_comparison.json; earlier summary under2_percent 8.9079/8.9342 describes <2s and is not substituted.'},
        'new_training':False,'new_audio_inference':False,'remote_state_queries':False,'active_training_modified':False,
        'spending_ceiling_usd':0,'resources_left_running':0,
        'scientific_scope':'Synthesis of archived experiments. Fixed-model input effects are distinguished from causal effects of training repetition; EER conditions and selected checkpoints remain explicit.'}
(OUT/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
plan['status']='complete';plan['validation_check_count']=len(checks)
(OUT/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':'passed','checks':len(checks),'sources':len(manifest),'tables':tables,'bytes':REPORT.stat().st_size},ensure_ascii=False))
