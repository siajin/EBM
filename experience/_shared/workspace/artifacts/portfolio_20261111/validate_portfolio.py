"""Independent source, arithmetic, context and deliverable checks for the portfolio."""
import csv
import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from statistics import mean

from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
REPORT=ROOT/'docs/portfolio/audio_deepfake_research_portfolio_20261111.md'
text=REPORT.read_text(encoding='utf-8')
checks=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def j(p):return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
def c(p):
    with (ROOT/p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def check(name,ok,detail=None):
    checks.append(dict(name=name,passed=bool(ok),detail=detail))
    if not ok:raise AssertionError((name,detail))
def row(values):return '|'+'|'.join(map(str,values))+'|'

plan=j('artifacts/portfolio_20261111/plan.json')
for p,h in plan['preserved_documents'].items():check('preserved_'+Path(p).name,sha(ROOT/p)==h)
manifest=j('artifacts/portfolio_20261111/sources.json')
for source in manifest:check('source_hash_'+source['path'],sha(ROOT/source['path'])==source['sha256'])
snapshot=j('artifacts/portfolio_20261111/metrics_snapshot.json')
comparison=j('artifacts/ebm_coverage_e20_eval_20261110/comparison.json')
fixed={r['phase']:r for r in comparison['fixed50_reference']}
for d in ['19LA','21LA','21DF']:
    r=fixed[d]
    check('common_input_fixed50_'+d,row([d,f"{r['legacy240_EER_percent']:.4f}%",f"{r['B_EER_percent']:.4f}%",f"{r['B_EER_percent']-r['legacy240_EER_percent']:+.4f}%p"]) in text)
df=fixed['21DF']; relative=(df['legacy240_EER_percent']-df['B_EER_percent'])/df['legacy240_EER_percent']*100
check('relative_DF_reduction',abs(relative-snapshot['relative_DF_reduction_percent'])<1e-10 and f'{relative:.2f}%' in text)
for r in comparison['same_epoch_comparison']:
    check('coverage20_'+r['phase'],row([r['phase']]+[f"{r[k+'_EER_percent']:.4f}%" for k in ['B','C','coverage']]) in text)
bc=j('artifacts/ebm_B_C_checkpoints_20261109/metrics.json')
minimum=j('artifacts/ebm_B_C_checkpoints_20261109/completed_B_C_dataset_minima.json')
for d in ['19LA','21LA','21DF']:
    cells=[d]
    for m in ['B','C']:
        value,epoch=min((bc[d][f'{m}_e{e}']['EER_percent'],e) for e in range(5,51,5))
        check('all_checkpoint_minimum_'+d+'_'+m,abs(value-minimum[d][m]['EER_percent'])<1e-10 and epoch==minimum[d][m]['epoch'])
        cells.extend([f'{value:.4f}%',epoch])
    check('minimum_row_'+d,row(cells) in text)
complete=j('artifacts/ebm_B_C_checkpoints_20261109/complete.json')
check('official_eval_total',sum(complete['phase_counts'][d] for d in ['19LA','21LA','21DF'])==753341 and '753,341' in text)
check('phase_completion_evidence',all(complete['phase_counts'][d]==n for d,n in [('19LA',71237),('21LA',148176),('21DF',533928)]))

paired=c('artifacts/repetition_shortcut_20261111/reviewed_analysis/independent_full_eer.csv')
for model,name in [('legacy240','기존 반복 학습'),('B','반복 제거 학습 B')]:
    a={r['condition']:r for r in paired if r['model']==model and r['phase']=='21LA'}
    check('paired_full21LA_'+model,row([name]+[f"{float(a[p]['eer']):.4f}%" for p in ['native','wrap4']]) in text)
directions=j('artifacts/paired_score_direction_20261111/results.json')['rows']
check('direction_class_counts',next(r for r in directions if r['model']=='legacy240' and r['true_class']=='bona')['predicted_bona_to_spoof']==608 and next(r for r in directions if r['model']=='legacy240' and r['true_class']=='spoof')['predicted_bona_to_spoof']==12862 and next(r for r in directions if r['model']=='B' and r['true_class']=='spoof')['predicted_spoof_to_bona']==9561)
crop=c('artifacts/crop_repetition_eer_20261111/requested_comparison.csv')
for r in crop:
    cells=[r['content_seconds']+'초',f"{float(r['B_native_EER_percent']):.4f}%",f"{float(r['B_change_from4_pp']):+.4f}%p",f"{float(r['original_repeat4_EER_percent']):.4f}%",f"{float(r['original_change_from4_pp']):+.4f}%p"]
    check('crop_requested_'+r['content_seconds'],row(cells) in text)
within=c('artifacts/crop_repetition_eer_20261111/within_model_contrasts.csv')
r=next(r for r in within if r['model']=='legacy240' and r['position']=='head' and r['content_seconds']=='1')
check('same1sec_repeat_benefit',all(f'{float(r[k]):.4f}%' in text for k in ['native_EER_percent','repeated_EER_percent']) and f"{abs(float(r['repeat_minus_native_pp'])):.4f}%p" in text)

swap=c('artifacts/geometry_cache/local_analysis/bonafide_cross_recheck_20261106_v1/class_replacement_eer.csv')
def get(b,s):return float(next(r['eer_percent'] for r in swap if r['baseline']=='19LA_eval' and r['bona_dataset']==b and r['spoof_dataset']==s))
for b,label in [('19LA_eval','19LA 정상'),('DF_full','DF 정상')]:check('class_swap_'+b,row([label]+[f'{get(b,s):.4f}%' for s in ['19LA_eval','DF_full']]) in text)
delta=get('DF_full','19LA_eval')-get('19LA_eval','19LA_eval')
check('normal_swap_delta',f'+{delta:.4f}%p' in text)
ci=next(r for r in c('artifacts/geometry_cache/local_analysis/bonafide_cross_recheck_20261106_v1/paired_contrasts.csv') if r['baseline']=='19LA_eval' and r['contrast']=='normal_swap_with_baseline_spoof')
check('normal_swap_adjusted_CI',f"[{float(ci['ci98_75_lower']):+.4f}, {float(ci['ci98_75_upper']):+.4f}]%p" in text and '98.75%' in text)

norm=c('artifacts/geometry_cache/local_analysis/frozen_all_v1/P3_P4/B/21LA_pilot/metrics.csv')
for v,name in [('direction_linear','코사인·최종 크기·잔여 방향'),('direction_noise2','같은 입력 + 무작위 숫자 2개'),('direction_hstat','같은 입력 + 정규화 전 평균·크기 통계')]:
    group=[r for r in norm if r['variant']==v]
    check('norm_three_probe_seeds_'+v,len(group)==3 and {r['seed'] for r in group}=={'101','102','103'})
    check('norm_table_'+v,row([name,f"{mean(float(r['balanced_logloss']) for r in group):.6f}",f"{mean(float(r['eer_percent']) for r in group):.4f}%"]) in text)
norm_ci=next(r for r in j('artifacts/geometry_cache/local_analysis/frozen_all_v1/P3_P4/B/21LA_pilot/paired_effects.json') if r['baseline']=='direction_linear' and r['candidate']=='direction_hstat')
check('norm_effect_CI',f"[{norm_ci['LL_CI95'][0]:.6f}, {norm_ci['LL_CI95'][1]:.6f}]" in text)

io=j('artifacts/train_crop_io_activity_20261110/results/io_benchmark.json')
check('IO_bit_exact',io['all_samples_equal'] and io['pairs_verified_exact']==512 and io['augmented_pairs_verified_exact']==64 and io['repeats']==3)
ratio=io['epoch1_weighted_mean_ms']['full']/io['epoch1_weighted_mean_ms']['partial']
reduction=100*(io['epoch1_weighted_mean_ms']['full']-io['epoch1_weighted_mean_ms']['partial'])/io['epoch1_weighted_mean_ms']['full']
check('IO_weighted_speedup_and_reduction',abs(ratio-io['epoch1_weighted_speedup'])<1e-12 and f'{ratio:.3f}배' in text and f'{reduction:.1f}%' in text)
check('IO_scope_not_training_time','증강과 GPU 계산을 포함한 전체 학습 시간이 같은 비율로 줄었다고 주장하지 않았습니다' in text)
coverage=j('artifacts/source_coverage_20261110/state.json')
check('coverage_policy50',coverage['source_epochs_verified']==50 and coverage['all_originals_in_OC_every_epoch'] and coverage['expected_original_probabilities_unchanged'])
get_time=lambda mode:next(r['original_time_fraction'] for r in coverage['epoch1_time_coverage'] if r['mode']==mode and r['population']=='bona_OC_plus_BPL')
check('coverage_time_change',f"{get_time('replacement')*100:.2f}%→{get_time('coverage')*100:.2f}%" in text)
initial=j('artifacts/analysis/ebm-performance-diagnosis-20261104.json')
for d in ['19LA','21LA']:
    for model in ['local160','global160']:
        check('initial_minimum_'+d+'_'+model,f"{initial['datasets'][d]['best'][model]['eer_percent']:.4f}%" in text)

c_state=j('artifacts/ebm_C_20261108/state.json')
bc_state=j('artifacts/ebm_B_C_checkpoints_20261109/state.json')
check('C_finished_actual50',c_state['run_exit']==0 and c_state['last_flushed_metric_rows']==5400 and c_state['training_finished_KST']=='2026-11-09T06:36:35+09:00')
check('BC_evaluation_exit',bc_state['run_exit']==0 and bc_state['finished_KST']=='2026-11-09T20:11:34+09:00')
c_meta=j('artifacts/ebm_C_20261108/metadata_validation.json')
check('BC_controls',c_meta['sampler_all50epochs_exactly_matches_B'] and c_meta['evaluation_fixed_B_input'] and c_meta['noaug_short_exactly_matches_B'])
max_error=max(j('artifacts/ebm_C_20261108/verification_rank'+str(rank)+'.json')['mask_solo_batch_embedding_max_error'] for rank in [0,1])
check('C_GPU_observed_embedding_error',max_error<1.44e-6,max_error)
short=j('artifacts/short32_train_20261111_v2/stop25_eval_status.json')
check('short32_archived_status_not_final_gain',short['last_completed_epoch']==25 and short['training_status']=='user_stopped' and '최종 21LA·DF 개선은 본 포트폴리오의 성과로 사용하지 않았습니다' in text)

chart=j('artifacts/portfolio_20261111/chart_data.json')
for m in ['legacy240','B','C']:check('chart_fixed50_'+m,chart['fixed50_'+m]==[fixed[d][m+'_EER_percent'] for d in ['19LA','21LA','21DF']])
check('chart_swap_binding',chart['class_swap_matrix']==[[get(b,s) for s in ['19LA_eval','DF_full']] for b in ['19LA_eval','DF_full']])
images=[]
for path in sorted((REPORT.parent/'figures').glob('*.png')):
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        check('figure_dimensions_'+path.name,image.width>=1200 and image.height>=700,list(image.size))
        images.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),size=list(image.size)))
check('three_report_figures',len(images)==3 and text.count('![')==3)
for ref in re.findall(r'\]\(([^)]+)\)',text):
    if ref.startswith(('http','#')):continue
    p=(REPORT.parent/ref.split('#')[0]).resolve()
    check('link_'+ref,p.exists() or p==OUT/'validation.json')
tables=0; columns=None
for line in text.splitlines():
    if line.startswith('|'):
        n=line.count('|')-1
        if columns is None:columns=n;tables+=1
        check('table_columns_'+str(len(checks)),n==columns)
    else:columns=None
check('all_requested_sections',all(s in text for s in ['핵심 성과','목표:','기반 모델','기간별 모델','문제와 고민','해결 접근','연구 판단','연구를 지속할','기대와 달랐던','다음 연구']))
check('user_confirmed_role',all(s in text for s in ['아이디어 제안','실험 계획','실험 수행·검증','결과 해석 주도','대학원생 멘토에게 매주','신규성과 연구 의의']))
check('no_invented_ownership',all(s not in text for s in ['단독 개발','기여도 100%','세계 최고','채용 보장','실서비스 배포','상용화 완료','110개 실험 수행']))
check('upstream_credit','선행 연구의 구현을 사용했습니다' in text and '../../EBM/README.md' in text)
check('critical_scope_disclosures',all(s in text for s in ['단일 초기값의 결과','사후 관측 최소','배치 160·50epoch 고정 모델','동일 20epoch의 보조 비교','반복 제거 학습 하나의 순수 효과','새로운 미관찰 평가 자료','EER은 0.8%로 같았습니다']))
check('no_unfilled_placeholders',not re.search(r'\b(?:TODO|TBD|FIXME)\b|작성 예정|추후 작성',text) and '\ufffd' not in text)
git_after=subprocess.check_output(['git','-C','EBM','status','--porcelain'],text=True)
check('existing_worktree_status_unchanged',git_after==plan['git_status_before'])
result=dict(status='passed',checked_at=datetime.now().astimezone().isoformat(),report_sha256=sha(REPORT),
            checks=checks,check_count=len(checks),source_count=len(manifest),table_count=tables,images=images,
            new_training=False,new_audio_inference=False,remote_queries=False,source_code_modified=False,
            spending_ceiling_usd=0,resources_left_running=0,
            validation_scope='Recomputed derived summaries, minima and ratios from archived metrics; preserved prior raw-score ROC validation as evidence. No new model training or inference.',
            visual_review='All three exported figures inspected. Final versions of the changed repetition and class-swap figures re-inspected for labels, overlap, contrast, scales and source agreement.')
(OUT/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
plan.update(status='complete',validation_check_count=len(checks),report=str(REPORT.relative_to(ROOT)))
(OUT/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status='passed',checks=len(checks),sources=len(manifest),tables=tables,figures=len(images),report_bytes=REPORT.stat().st_size)))
