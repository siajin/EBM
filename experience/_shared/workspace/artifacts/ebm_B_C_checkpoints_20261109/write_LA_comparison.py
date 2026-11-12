"""Bind reviewed LA metrics into an inline comparison and the experiment record."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
OUT=HERE/'LA_analysis'
a=json.loads((OUT/'analysis.json').read_text())
bar={'id':'la-fixed50-comparison','title':'고정 50 epoch · 같은 비반복 입력의 EER','description':'19LA 71,237개·21LA 148,176개. 기존 모델도 B·C와 같은 앞64,600샘플·비반복·mask 조건으로 재평가한 값입니다. 낮을수록 좋습니다.','height':260,
     'spec':{'type':'bar','x':'평가셋','y':['기존','B','C'],'yLabel':'EER (%)','showLegend':True,'showValues':True,'valueDecimals':3,'colors':{'기존':'var(--chart-3)','B':'var(--chart-1)','C':'var(--chart-2)'}},'columns':['평가셋','기존','B','C'],
     'rows':[[phase]+[a['models'][phase][model]['EER_percent'] for model in ['legacy240','B_e50','C_e50']] for phase in ['19LA','21LA']]}
charts=[bar]
for phase in ['19LA','21LA']:
    charts.append({'id':f'la-epochs-{phase.lower()}','title':f'{phase} · B/C 전체 체크포인트 EER','description':'Target 결과의 사후 비교이며, 이 곡선의 최저값으로 최종 모델을 선택하지 않습니다.','height':240,
                   'spec':{'type':'line','x':'Epoch','y':['B','C'],'xLabel':'Epoch','yLabel':'EER (%)','showLegend':True,'valueDecimals':4,'colors':{'B':'var(--chart-1)','C':'var(--chart-2)'}},
                   'columns':['Epoch','B','C'],'rows':[[e,a['models'][phase][f'B_e{e}']['EER_percent'],a['models'][phase][f'C_e{e}']['EER_percent']] for e in range(5,51,5)]})
fragment=HERE/'la-model-comparison.html'
text=fragment.read_text(encoding='utf-8')
assert text.count('CHART_PAYLOAD')==1
fragment.write_text(text.replace('CHART_PAYLOAD',json.dumps({'charts':charts},ensure_ascii=False,indent=2)),encoding='utf-8')

for phase in ['19LA','21LA']:
    for model in ['legacy240','B_e50','C_e50']:
        m=a['models'][phase][model];bins=a['length_bins'][phase][model]
        assert sum(b['normal_n'] for b in bins.values())==m['bonafide']
        assert sum(b['spoof_n'] for b in bins.values())==m['spoof']
        assert sum(b['normal_reject_count'] for b in bins.values())==m['normal_reject_count']
        assert sum(b['spoof_accept_count'] for b in bins.values())==m['spoof_accept_count']
        assert sum(b['accepted'] for b in a['attack_errors'][phase][model].values())==m['spoof_accept_count']
    b=a['models'][phase]['B_e50'];c=a['models'][phase]['C_e50'];p=a['paired_B_C'][phase]['50']['paired_errors']
    assert p['normal']['C_only_wrong']-p['normal']['B_only_wrong']==c['normal_reject_count']-b['normal_reject_count']
    assert p['spoof']['C_only_wrong']-p['spoof']['B_only_wrong']==c['spoof_accept_count']-b['spoof_accept_count']

validation={'status':'passed','checks':a['validation'],'length_and_attack_totals_reconcile':True,'paired_error_deltas_reconcile':True,'chart_values_bound_directly_to_verified_metrics':True,'historical_legacy_scope':'Original repeated-input pipeline, not controlled common-input reference','limitation':'Single model pair/seed; checkpoint outcomes correlated; no uncertainty intervals estimated; DF pending'}
(OUT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n',encoding='utf-8')

report=HERE.parents[1]/'docs/experiments/ebm_B_C_all_checkpoint_evaluation_20261109.md'
block='\n\n## 완료된19LA·21LA의 기존/B/C 중간 비교\n\n19LA71,237개(정상7,355/spoof63,882),21LA148,176개(정상14,816/spoof133,360)의완료CSV만읽어CPU분석했다. DF는이비교에사용하지않았고진행중평가를변경하지않았다. 기존모델은보존된동일B입력고정50평가점수를재사용했다. 각UID/index/speaker/label/입력길이/attack이현재결과와정확히같고B50모든점수도오차0으로재현되어참조를연결했다. 전21모델×두집합EER를별도정렬/동점그룹ROC로검산했다.\n\n| 고정50 모델·조건 | 19LA EER(%) | 21LA EER(%) |\n|---|---:|---:|\n'
for model,label in [('legacy240','기존·공통비반복입력'),('B_e50','B·공통비반복입력'),('C_e50','C·공통비반복입력')]:
    block+=f"| {label} | {a['models']['19LA'][model]['EER_percent']:.6f} | {a['models']['21LA'][model]['EER_percent']:.6f} |\n"
block+=f"| 기존·과거반복입력파이프라인 | {a['historical_legacy']['19LA']:.6f} | {a['historical_legacy']['21LA']:.6f} |\n"
block+='\n공통입력에서B/C는기존보다두LA집합의EER가낮다. 그러나원래기존파이프라인의과거반복평가와비교하면둘다낮지않다. 이조건차이를학습효과로혼동하지않는다. BvsC에서는C의19LA EER가0.095173pp(상대14%)낮지만21LA는0.371351pp(상대7.7492%)높다. C의Long상한해제가LA전반에개선된다는결론은지지되지않는다. 단일seed/모델쌍의점추정이며불확실성구간을계산하지않았으므로통계적유의성을주장하지않는다.\n\n| Epoch | 19LA B | 19LA C | 21LA B | 21LA C |\n|---|---:|---:|---:|---:|\n'
for e in range(5,51,5):block+='| '+str(e)+' | '+' | '.join(f"{a['models'][phase][f'{model}_e{e}']['EER_percent']:.6f}" for phase in ['19LA','21LA'] for model in ['B','C'])+' |\n'
block+='\n21LA는동일epoch의10개비교모두C가B보다높다. 19LA는C가6개epoch에서낮고4개에서높다. 체크포인트는상관된관측이며10개의독립실험으로취급하지않는다. Dev15~50EER가동일했던사실은target일반화성능동일성을보장하지않는다. Target최저epoch를최종모델로선택하지않는다.\n\n각모델의dev임계값을고정한운영점오류:\n\n| 집합 | 모델 | 정상거부 수/분모 (%) | spoof통과 수/분모 (%) |\n|---|---|---:|---:|\n'
for phase in ['19LA','21LA']:
    for model in ['legacy240','B_e50','C_e50']:
        m=a['models'][phase][model]
        block+=f"| {phase} | {model} | {m['normal_reject_count']}/{m['bonafide']} ({m['normal_reject_percent']:.4f}) | {m['spoof_accept_count']}/{m['spoof']} ({m['spoof_accept_percent']:.4f}) |\n"
block+='\nC는19LA정상거부1개를유지하면서spoof통과를236개줄였다. 21LA에서는spoof통과180개감소와정상거부210개증가가함께발생했다. 정상오류의순증210개는B에서만실패25개/C에서만실패235개의차이다. 21LA정상거부증가는길이구간별11·86·83·30개로모든구간에서발생했다.\n\n원길이2초미만정상거부는19LA 기존0/265,B1/265,C1/265이며,21LA 기존27/507(5.3254%),B67/507(13.2150%),C78/507(15.3846%)다. 이는각모델별dev운영점오류이고targetEER와다르다. C가아주짧은정상문제를해결했다는근거는이두LA집합에서없다. 기존운영점의낮은정상거부율에는훨씬높은짧은spoof통과율이동반되어단일오류율만으로우열을판단하지않는다.\n\n[전체 수치](../../artifacts/ebm_B_C_checkpoints_20261109/LA_analysis/analysis.json), [epoch별 요약](../../artifacts/ebm_B_C_checkpoints_20261109/LA_analysis/epoch_summary.csv), [길이별 요약](../../artifacts/ebm_B_C_checkpoints_20261109/LA_analysis/length_summary.csv), [검산](../../artifacts/ebm_B_C_checkpoints_20261109/LA_analysis/validation.json), [재현 코드](../../artifacts/ebm_B_C_checkpoints_20261109/analyze_completed_LA.py).\n'
text=report.read_text(encoding='utf-8');assert '## 완료된19LA·21LA의 기존/B/C 중간 비교' not in text
report.write_text(text+block,encoding='utf-8')
statefile=HERE/'state.json';state=json.loads(statefile.read_text());state.update(LA_analysis_complete=True,LA_analysis='artifacts/ebm_B_C_checkpoints_20261109/LA_analysis/analysis.json',analysis_complete=False);statefile.write_text(json.dumps(state,indent=2)+'\n')
print(json.dumps({'chart_payloads':len(charts),'validation':'passed','report':str(report)},indent=2))
