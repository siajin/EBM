"""Verify neighborhood nesting and report existing arc density results."""
from spoof_path_density import *


def mdtable(frame):
    lines=['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    lines+=['| '+' | '.join(str(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)]
    return '\n'.join(lines)


def main():
    assert (DENSITY_ROOT/'complete').read_text().strip()=='0'
    summary=json.loads((DENSITY_ROOT/'summary.json').read_text(encoding='utf-8'))
    metrics=pd.read_csv(DENSITY_ROOT/'density_metrics.csv')
    radii=pd.read_csv(DENSITY_ROOT/'radii.csv')
    checks=[];previous_witnesses=[];median_counts=[]
    witnesses=pd.read_csv(OUT/'continuous_labels/witnesses.csv')
    for ds in DATASETS:
        rows,x,s,u=load_view(ds);y=rows.y.to_numpy();nb=int((y==1).sum());ns=int((y==0).sum())
        for rep,f in [('unit_z',x),('residual_u',u)]:
            frame=pd.read_parquet(DENSITY_ROOT/f'{ds}_{rep}_edge_density.parquet')
            radius=float(radii[(radii.dataset==ds)&(radii.representation==rep)].base_radius.iloc[0])
            for measure,increasing in [('t5_empty_fraction',False),('mean_neighbor_bona',True),('mean_neighbor_spoof',True),('tube_unique_bona',True),('tube_unique_spoof',True)]:
                piv=frame.pivot(index='edge_row',columns='radius_multiplier',values=measure)[[.5,1.,2.]].to_numpy()
                changes=np.diff(piv,axis=1)
                assert np.all(changes>=-1e-10) if increasing else np.all(changes<=1e-10)
            for _,rec in witnesses[(witnesses.dataset==ds)&(witnesses.representation==rep)].iterrows():
                a,b=f[int(rec.a_row)],f[int(rec.b_row)];t=float(rec.arc_fraction);w=np.arccos(np.clip(a@b,-1,1))
                point=(np.sin((1-t)*w)*a+np.sin(t*w)*b)/np.sin(w)
                distances=np.linalg.norm(f-point,axis=1);distances[[int(rec.a_row),int(rec.b_row)]]=np.inf
                for mult in [.5,1.,2.]:
                    count_b=int(np.sum((distances<=radius*mult)&(y==1)))
                    count_s=int(np.sum((distances<=radius*mult)&(y==0)))
                    equivalent=count_b+count_s*nb/(ns-2)
                    state='empty' if count_b+count_s==0 else ('few' if equivalent<5 else ('bona' if count_b/nb>=2*count_s/(ns-2) else ('spoof' if count_s/(ns-2)>=2*count_b/nb else 'mixed')))
                    previous_witnesses.append({'dataset':ds,'representation':rep,'edge_type':rec.edge_type,'a_row':int(rec.a_row),'b_row':int(rec.b_row),
                        'arc_fraction':t,'radius_multiplier':mult,'radius':radius*mult,'bona_count':count_b,'spoof_count':count_s,
                        'state':state,'nearest_bona_distance':float(distances[y==1].min()),'nearest_spoof_distance':float(distances[y==0].min())})
            # A different integral computation checks event-sweep means.
            for kind in ['local_1','pc1_extremes']:
                er=frame[(frame.edge_type==kind)&(frame.radius_multiplier==1)].iloc[0]
                a,b=f[int(er.a_row)],f[int(er.b_row)];w=np.arccos(np.clip(a@b,-1,1));v=(b-(a@b)*a)/np.sin(w)
                lo,hi=ball_intervals(f@a,f@v,w,radius,[int(er.a_row),int(er.b_row)])
                length=np.maximum(hi-lo,0)/.8
                mean_b=float(length[y==1].sum());mean_s=float(length[y==0].sum())
                assert abs(mean_b-er.mean_neighbor_bona)<1e-8 and abs(mean_s-er.mean_neighbor_spoof)<1e-8
                checks.append({'dataset':ds,'representation':rep,'edge_type':kind,'mean_bona_error':float(abs(mean_b-er.mean_neighbor_bona)),
                               'mean_spoof_error':float(abs(mean_s-er.mean_neighbor_spoof))})
            for (kind,mult),g in frame.groupby(['edge_type','radius_multiplier']):
                median_counts.append({'dataset':ds,'representation':rep,'edge_type':kind,'radius_multiplier':mult,
                    'median_tube_unique_bona':float(g.tube_unique_bona.median()),'median_tube_unique_spoof':float(g.tube_unique_spoof.median()),
                    'median_mean_neighbor_bona':float(g.mean_neighbor_bona.median()),'median_mean_neighbor_spoof':float(g.mean_neighbor_spoof.median())})
    wf=pd.DataFrame(previous_witnesses);wf.to_csv(DENSITY_ROOT/'previous_nearest_label_witness_counts.csv',index=False)
    pd.DataFrame(median_counts).to_csv(DENSITY_ROOT/'median_counts.csv',index=False)
    save_json(DENSITY_ROOT/'independent_verification.json',{'status':'pass','radius_nesting':'all edges, all datasets and representations',
         'independent_interval_integral_checks':checks,'previous_witness_count':len(wf)//3})
    # This changes the interpretation of the preceding nearest-label analysis.
    selected=[]
    for ds in ['DF_pilot','21LA_pilot']:
        for kind in ['local_1','same_score_local','random','pc1_extremes']:
            r=metrics[(metrics.dataset==ds)&(metrics.representation=='residual_u')&(metrics.edge_type==kind)&(metrics.radius_multiplier==1)].iloc[0]
            rec={'자료':ds,'연결':TYPE_NAMES[kind],'경로 수':int(r.edges)}
            for name in ['spoof','bona','mixed','few','empty']:rec[STATE_LABELS[name]+' (%)']=f'{100*r[f"t5_{name}_fraction"]:.2f}'
            selected.append(rec)
    mass=[]
    for ds in ['DF_pilot','21LA_pilot']:
        for kind in ['local_1','pc1_extremes']:
            r=metrics[(metrics.dataset==ds)&(metrics.representation=='residual_u')&(metrics.edge_type==kind)&(metrics.radius_multiplier==1)].iloc[0]
            mass.append({'자료':ds,'연결':TYPE_NAMES[kind],'평균 주변 보나 수':f'{r.mean_neighbor_bona:.2f}',
                '평균 주변 스푸프 수':f'{r.mean_neighbor_spoof:.2f}',
                '보나 1000개당':f'{r.mean_bona_per1000:.3f}','스푸프 1000개당':f'{r.mean_spoof_per1000:.3f}'})
    sensitivity=[]
    for ds in ['DF_pilot','21LA_pilot']:
        for mult in [.5,1.,2.]:
            for kind in ['local_1','pc1_extremes']:
                r=metrics[(metrics.dataset==ds)&(metrics.representation=='residual_u')&(metrics.edge_type==kind)&(metrics.radius_multiplier==mult)].iloc[0]
                sensitivity.append({'자료':ds,'연결':TYPE_NAMES[kind],'반경 배수':mult,
                    '둘 다 없음 (%)':f'{100*r.t5_empty_fraction:.2f}','희박 (%)':f'{100*r.t5_few_fraction:.2f}',
                    '스푸프 우세 (%)':f'{100*r.t5_spoof_fraction:.2f}','보나 우세 (%)':f'{100*r.t5_bona_fraction:.2f}',
                    '혼재 (%)':f'{100*r.t5_mixed_fraction:.2f}'})
    radius_table=radii.copy()
    for col in ['base_radius','normal_5nn_median_in_balanced_pool','spoof_5nn_median_in_balanced_pool']:radius_table[col]=radius_table[col].map(lambda a:f'{a:.6f}')
    primary_w=wf[wf.radius_multiplier==1]
    empty_w=int((primary_w.state=='empty').sum());few_w=int((primary_w.state=='few').sum())
    one=primary_w[(primary_w.dataset=='DF_pilot')&(primary_w.representation=='residual_u')].iloc[0]
    doc=Path(__file__).resolve().parents[2]/'docs/experiments/spoof_path_density_20261105.md'
    text='''# 스푸프 연결 경로 주변의 보나와 스푸프 표본 수

2026년 11월 5일. 기존 고정 EBM의 저장 표현과 앞선 분석의 연결 쌍을 재사용했다. 최근접 클래스만 정하는 대신 같은 반경 안의 클래스별 표본 수를 세어, 관찰 표본이 없는 경로와 스푸프가 실제로 많은 경로를 구분한다.

앞선 ‘정상 최근접 영역을 지나지 않음’은 스푸프 표본이 경로를 채운다는 증거가 아니다. 이번 결과는 표본의 빈 공간을 명시적으로 포함한 후속 해석이며, 앞선 최근접 판정의 수치를 바꾸지는 않는다.

## 집계 기준

T 25,380개, V 14,983개, 기존 DF·21LA pilot 각 20,000개(보나 2,000, 스푸프 18,000)의 같은 체크포인트를 사용했다. 전체 단위 임베딩 x와 코사인 성분을 제거한 잔여방향 u를 따로 계산했다. 아래 주표는 잔여방향 공간이다.

각 자료·공간에서 고정 난수로 최대 2,000개씩 클래스 균형 표본을 골라, 혼합된 표본 안에서 다섯 번째 다른 이웃까지 거리의 중앙값을 기본 반경으로 정했다. 같은 반경을 두 클래스와 경로의 모든 위치에 적용하며, 절반·기본·두 배를 모두 보고했다. 반경은 탐색용 공간 척도이고 유일한 정답은 아니다.

집계에는 각 클래스의 전체 저장 표본을 쓰되 연결의 양끝 스푸프 두 개는 제외했다. 경로의 양끝 10%도 제외했다. 그래서 끝점 자체로 인해 경로가 스푸프로 채워져 보이는 효과를 줄였다. 단위구면 최단 호와 각 표본 반경 공이 만나는 구간을 해석적으로 구하고, 진입·이탈 사건을 정렬해 연속 경로 길이 비율과 주변 표본 수를 적분했다. 중간점 격자만으로 빈 구간을 정하지 않았다.

경로의 각 위치에서 보나 수 nB, 스푸프 수 nS를 센다. 클래스별 전체 수 NB, NS에 대해 b=nB/NB, f=nS/(NS−2)를 함께 기록해 스푸프 표본이 약 9배 많은 효과를 보정했다. 동일한 공의 부피는 두 클래스 비교에서 상쇄되므로 이는 클래스별 국소 표본 비율의 비교다. 원래 클래스 비율의 사후확률은 아니다.

- **둘 다 없음:** nB=nS=0. 관찰된 표본에 대한 빈 구간이다.
- **희박:** 0 < NB×(b+f) < 5. 클래스 균형 표본으로 환산해 5개 미만의 지지다.
- **보나 우세:** 환산 지지가 5개 이상이고 b≥2f.
- **스푸프 우세:** 환산 지지가 5개 이상이고 f≥2b.
- **혼재:** 지지가 5개 이상이며 두 클래스 비율 차이가 2배 미만.

‘많음’과 ‘희박’은 운영상 탐색 기준이다. 환산 지지 1·5·10의 결과를 모두 저장했다. 분류 비율은 각 경로 내부 길이에서 구한 뒤 경로마다 같은 가중치로 평균했다. 따라서 ‘경로 중 몇 개가 전부 비었는가’와는 다른 값이다.

## 기본 반경에서의 경로 구성

'''+mdtable(pd.DataFrame(selected))+'''

![](../../artifacts/geometry_cache/local_analysis/spoof_path_geometry_v1/path_density_v1/density_support.png)

같은 반경에서 보나와 스푸프가 모두 0인 구간은 ‘스푸프 우세’에 포함하지 않는다. 반대로 빈 구간이 아니어도 표본이 충분하지 않으면 희박으로 따로 남긴다.

## 실제 주변 개수와 클래스별 비율

'''+mdtable(pd.DataFrame(mass))+'''

위 개수는 경로의 임의 위치 주변에서 기대되는 표본 수를 경로 길이에 대해 평균한 것이다. 경로 전체에서 한 번이라도 가까웠던 고유 표본 수는 별도 `tube_unique_bona`, `tube_unique_spoof`에 있고, 평균·중앙값을 모두 저장했다. 긴 경로의 고유 표본 수가 많다고 경로의 중간이 조밀하다는 뜻은 아니다.

## 반경에 따른 민감도

'''+mdtable(pd.DataFrame(sensitivity))+'''

반경이 커지면 주변 표본 수가 늘고 빈 구간이 줄어드는지 모든 연결에서 확인했다. 서로 다른 공간의 반경 숫자는 직접 비교하지 않는다. 클래스별 밀도와 국소 구조가 달라 기본 반경만으로 결론을 고정하면 안 된다.

'''+mdtable(radius_table)+f'''

## 이전 최근접 반례를 다시 세었을 때

앞선 분석의 정상 최근접 진입 증거점 30개 중 기본 반경에서 **둘 다 0인 지점은 {empty_w}개**, **희박한 지점은 {few_w}개**였다. 이 개수는 서로 독립적인 반례 수가 아니고, 중복 끝점·다른 연결 종류가 포함된 30개 경로 관찰이다.

DF 잔여방향의 유일한 긴 축 반례 지점에서는 기본 반경 {one.radius:.6f} 안에 보나 {int(one.bona_count)}개, 스푸프 {int(one.spoof_count)}개였다. 가장 가까운 보나까지 거리는 {one.nearest_bona_distance:.6f}, 스푸프까지 거리는 {one.nearest_spoof_distance:.6f}였다. ‘보나가 더 가깝다’와 ‘근처에 보나가 많다’가 서로 다름을 보여준다. 절반·두 배 반경의 개수도 `previous_nearest_label_witness_counts.csv`에 있다.

## 해석과 검증

학습 가설에는 보나 침범 여부뿐 아니라 경로의 스푸프 지지와 빈 구간 비율을 함께 써야 한다. 스푸프 지지가 많은 구간은 연결 학습 후보가 될 수 있지만, 빈 구간은 정답을 알 수 없는 외삽 구간으로 남긴다. 두 클래스가 함께 조밀한 구간은 별도로 취급한다. 이번 결과는 실제 보간 음성의 정답이나 새 학습법의 효과를 증명하지 않는다.

캐시 해시·동일 체크포인트·행 대응을 확인했다. 모든 반경에서 각 경로의 다섯 상태 비율 합이 1인지, 반경 확대에 따라 표본 수가 단조 증가하고 빈 비율이 감소하는지 확인했다. 실제 표본의 직접 160좌표 거리 집계와 구간 계산을 대조했고, 사건 누적 적분과 각 표본의 구간 길이 합도 독립 대조했다. 합성 빈 공간·부분 지지 사례를 포함했다.

관찰 표본이 0이라고 실제 음성 분포가 완전히 비었다고 증명한 것은 아니다. 표본 수와 반경의 한계가 있으며, 기존 benchmark를 이미 반복 관찰한 탐색 분석이다. DF와 21LA는 독립 반복 실험으로 세지 않고 원발화 계열 의존성도 미해결이다. 기존 실험·모델·원본 캐시는 바꾸지 않았다.

결과는 `artifacts/geometry_cache/local_analysis/spoof_path_geometry_v1/path_density_v1/`에 있다. `density_metrics.csv`는 전체 상태 비율·평균 개수, 각 `edge_density.parquet`는 경로별 값, `radii.csv`는 반경, `median_counts.csv`는 중앙값, 두 verification JSON은 검증 근거다. 재현 코드는 `artifacts/geometry_experiments/spoof_path_density.py`, 보고서와 독립 검증 코드는 `report_spoof_path_density.py`다.
'''
    doc.write_text(text,encoding='utf-8')
    old=doc.with_name('spoof_path_geometry_20261105.md')
    oldtext=old.read_text(encoding='utf-8')
    note='> 후속 밀도 분석: 최근접 클래스와 실제 주변 표본 수는 다르다. 빈 구간을 포함한 [경로 주변 표본 수 분석](spoof_path_density_20261105.md)을 함께 읽는다. 아래 최근접 진입 수치는 보존하지만 스푸프가 경로를 채운다는 뜻으로 해석하지 않는다.\n\n'
    if note not in oldtext:
        backup=DENSITY_ROOT/'previous_geometry_report_before_density.md'
        if not backup.exists():backup.write_text(oldtext,encoding='utf-8')
        first,rest=oldtext.split('\n\n',1)
        old.write_text(first+'\n\n'+note+rest,encoding='utf-8')
    print('REPORT',doc)
    print(pd.DataFrame(selected).to_string(index=False))
    print('WITNESS',len(primary_w),empty_w,few_w)
    print('DF RESIDUAL WITNESS',one.to_dict())


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
