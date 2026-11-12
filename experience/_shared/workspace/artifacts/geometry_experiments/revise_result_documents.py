"""Revise every related interpretation while preserving historical results/images."""
from pathlib import Path
import hashlib
import json
import re
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd

ROOT=Path(__file__).resolve().parents[2];DOCS=ROOT/'docs/experiments';DETAIL=DOCS/'cached_core_20261105';OUT=ROOT/'artifacts/geometry_cache/local_analysis/frozen_all_v1'
BACKUP=OUT/'analysis_md_revision_v1';CHANGED=[]
STAMP=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M KST')

def write(path,text):
    old=path.read_text(encoding='utf8')
    if text==old:return
    relative=path.relative_to(DOCS);copy=BACKUP/'before'/relative;copy.parent.mkdir(parents=True,exist_ok=True)
    if not copy.exists():copy.write_text(old,encoding='utf8')
    path.write_text(text,encoding='utf8');CHANGED.append(dict(path=str(path.relative_to(ROOT)),before_sha256=hashlib.sha256(old.encode()).hexdigest(),after_sha256=hashlib.sha256(text.encode()).hexdigest()))

def banner(text,scope):
    first,rest=text.split('\n',1)
    notice=f'\n\n> **해석 갱신: {STAMP}.** {scope} 최신 통합 판정은 [방향·채널 전체 분석](../ebm_frozen_all_results_20261105.md)에 있다. 기존 실험의 수치와 그림은 해당 실행의 결과로 보존한다.\n'
    return first+notice+rest

def section(text,start,end,body):
    a=text.index(start);b=text.index(end,a) if end else len(text)
    return text[:a]+start+'\n\n'+body.strip()+'\n\n'+text[b:]

def main():
    BACKUP.mkdir(exist_ok=False)
    # Direction: explicitly separate old internal diagnostics, strict matching,
    # completed transfer and still-incomplete new internal CV.
    p=DETAIL/'01_direction.md';s=banner(p.read_text(encoding='utf8'),'아래 기존 표는 DF 내부 spline-linear 진단이다. 새 T/V 전이 결과와 strict matching을 추가해 해석을 수정했다.')
    new='''### 후속 결과: 같은 질문의 서로 다른 평가 경로

| 실험 | 실제 결과 | 현재 해석 |
| --- | --- | --- |
| 기존 DF 내부 spline-linear | scalar LL0.098882 → direction LL0.038342; EER3.5056% →1.20% | 정한 진단 계열에서 DF의 다른 화자에게 읽히는 방향 신호 |
| 새 T/V 전이, 선형 잔차 | DF 평균 LL 감소0.002387855, EER3.35% →3.40%;21LA LL 감소0.000216622, EER0.80% 유지 | 작은 확률 예측 이득과 EER 개선 실패를 구분 |
| 새 T/V 전이, MLP 잔차 | V가 epoch0을 선택해 native와 같은 최종 점수 | 이 선택 절차에서 비영 MLP 잔차를 채택할 근거가 없음 |
| 새 strict F2 | DF27쌍/9그룹,21LA10쌍/6그룹 | 최소200쌍·10그룹 미달로 판정 불가 |
| 새 DF/21LA 내부 P3-B | 전체 fold·seed·모델 계열의 CV와 검수 진행 중 | 일부 fold의 결과를 전체 성능으로 보고하지 않음 |

새 전이의 LL 감소에 대한 개별95% 구간은 DF[0.001384802,0.003495527],21LA[0.000081649,0.000385810]이다. 반면 DF EER 개선량은−0.05pp,개별95% 구간[−0.088194,0]으로 개선되지 않았다. 세 seed 중 비영 선형 방향 잔차는 seed101에서만 선택됐다. LL 이득을 세 번의 독립 재현이나 전체 탐색 보정 후의 확증으로 부르지 않는다.

아래의 기존1.20%/0.60%는 여전히 **DF 내부 진단 성능**이다. 위의 전이 결과와 섞어 평균내거나 같은 방법의 재현 실패라고 부르지 않는다. 모델 계열·학습 자료·규제와 선택 규칙이 다르다.

'''
    s=s.replace('## 4. 결과\n','## 4. 결과\n\n'+new,1)
    body='''**현재 z에서 잔여 방향 관련 신호를 읽을 수 있다는 기존 DF 내부 근거는 유효하다. 그러나 그 신호를 T에서 학습한 readout으로 사용해 평가셋의 EER를 개선한다는 근거는 얻지 못했다.** 새 선형 전이는 LL을 작게 줄였으므로 “방향을 사용해도 아무 효과가 없다”는 표현도 피한다.

기존106쌍은 같은 source×codec을 요구했지만 같은 화자를 요구하지 않았다. 새 F2는 DF의 source×speaker×codec,21LA의 speaker×codec×transmission을 맞추고 caliper0.02/0.002를 적용했다. 옛106쌍과 새27쌍은 같은 실험의 모순된 수치가 아니다. 새 넓은 caliper 민감도에서도 DF58쌍/15그룹,21LA21쌍/14그룹으로200쌍에 못 미쳤다. 결론은 표본 부족이지 방향 정보 부재가 아니다.

MMD²와 평균 방향 각도가 0보다 크다는 관찰만으로 조건부 방향 차이가 증명되지 않는다. 근사 matching이며,biased MMD의 bootstrap 구간은 동일분포 null 검정이 아니다. 방향의 평균 차이·분산 구조는 작은 매칭 표본의 기술 통계로 한정한다.

u는 z의 결정적 변환이다. 추가 입력에서 더 잘 읽힌다는 것은 정한 probe의 표현 접근성에 관한 결과이며, 정보량 생성이나 정확한 조건부 상호정보량의 측정이 아니다. metadata 대조가 있어도 측정하지 않은 조건·비선형 교란과 원발화 family 의존성이 남는다.

원래 EBM은 centroid 각도를 이용하고 채널 혼합·layer/time attention도 학습한다. “방향을 전혀 사용하지 않는다”보다 “최종 scalar에 남지 않은 잔여 방위를 이 readout이 활용할 수 있는가”가 정확한 질문이다. 새 공동 학습 attention/방향 loss의 효과는 이번 동결 실험으로 확정하지 않는다.

기존2,000회 개별95% CI와 새10,000회 전이 CI는 서로 다른 fit·선택에 조건부다. 기존 DF1.20%/0.60%를 T→DF 또는 새 EBM의 일반화 성능으로 제시하지 않는다. 새 P3-B가 끝나면 같은 모델 계열의 내부/전이 차이를 별도로 판단한다.'''
    s=section(s,'## 5. 해석과 한계','## 6. 결과 파일',body);write(p,s)
    # LayerNorm: retain the complete old methods and values; replace inference.
    p=DETAIL/'03_layernorm.md';s=banner(p.read_text(encoding='utf8'),'기존 h/z 선형 probe 결과와 새 scalar+u 증분 P4를 구별한다. 신규 P4의 최종 결과는 아직 없다.')
    s=section(s,'## 6. 해석과 한계','## 7. 수치와 구현 근거','''**기존 DF 내부에서 h를 사용한 동일 계열 probe가 z를 사용한 probe보다 LL이 낮았다.** 이는 정한 분류기의 정보 접근성 차이를 조사할 근거다. EER1.4944%는 h와 기존 centroid의 cosine 성능이 아니라 별도 로지스틱 probe의 점수다.

z는 h의 결정적 변환이지만, 한 선형 probe의 성능 차이만으로 어떤 정보가 얼마나 사라졌는지 또는 LayerNorm이 기존 탐지 오류를 일으켰는지 판정할 수 없다. 좌표 배치·scale·규제·수치 조건이 같은 정보의 읽기 쉬운 정도를 바꿀 수 있다.

LayerNorm은 발화 내부의 공통 shift에 정확히 불변이다. 양의 scale에는 epsilon>0일 때 일반적으로 정확히 불변하지 않다. 여기 epsilon은1e-12이고 관찰 h분산은 훨씬 크므로 scale 영향이 강하게 정규화되는 상황이다. “평균과 분산 정보가 전부 수학적으로 완전히 소실됐다”는 표현은 쓰지 않는다. 이 통계는 원음의 음량이 아니라 최종160좌표의 통계다.

기존 scalar+h-stat가 전체 h보다 이득이 작았다는 사실만으로 h/z 차이를 두 통계가 설명하거나 설명하지 못한다고 단정할 수 없다. 옛 실험은 scalar+u에 h-stat/h를 추가하고 같은 차원의 noise와 비교한 대조가 아니다.

새 P4는 **scalar+u**를 공통 기준으로 mean(h)·log(std(h)) 두 열 또는 h160을 추가하고,각각 noise2/noise160과 비교한다. DF/21LA 각2만 개에 별도5×3 grouped CV를 쓰며 scaler·ridge·epoch를 fold 안에서만 정한다. 현재 전체 실행·OOF·검수가 끝나지 않았으므로 이 증분의 성공 여부와21LA 재현은 미정이다.

보고 가능한 결론은 “옛 DF 내부 probe에서 h의 접근성이 더 좋았다”다. 보고할 수 없는 결론은 “LayerNorm 제거로 기존 EBM이 개선됐다”,“정규화가 나빠진 원인이다”,“새 attention이 이 정보를 이용할 것이다”다. 후자는 공동 학습 또는 적절한 통제 개입이 필요하다.

옛 LL 개선의 개별95% 구간과 EER 차이의 추론을 구분한다. 옛 EER0.2056pp 차이에 별도 EER CI를 계산한 것은 아니다. 이미 관찰한 하나의 DF 표본·checkpoint와 저장 OOF 예측에 조건부인 bootstrap이며,원발화 family와 전체 재학습 불확실성은 미해결이다.''')
    write(p,s)
    # Conditions: add the completed full-vector F4 statistics, keep all NMI tables.
    p=DETAIL/'04_conditions.md';s=banner(p.read_text(encoding='utf8'),'기존 K=4 NMI와 새 DF/21LA 전체 잔여방향의 class별 분산 통계를 함께 해석한다.')
    eta=pd.read_csv(OUT/'F4/vector_variance.csv');eta=eta[eta.stratum=='all']
    rows=['| 데이터 | class | 조건 | vector eta² |','| --- | --- | --- | ---: |']
    for r in eta.itertuples():rows.append(f'| {r.dataset} | {"정상" if r.y else "spoof"} | {r.factor} | {r.eta_squared:.6f} |')
    extra='''### 후속 F4: 두 평가셋의 전체 잔여방향 분산

각 class 안에서 조건별 평균의 between-group 제곱합을 전체 u 분산의 제곱합으로 나눈 eta²다. 군집 번호의 NMI가 아니며 두 지표의 크기를 직접 비교하지 않는다.

'''+ '\n'.join(rows)+'''

T-fit PCA1과 spoof padding의 Spearman은 DF−0.600272,21LA−0.577569였다. duration은각각+0.601706,+0.576446이었다. PCA 축의 부호는 임의적이며 duration과 padding은 고정 crop 규칙으로 연결된다. 둘을 독립된 원인 두 개로 세지 않는다.

'''
    s=s.replace('## 5. 해석과 한계',extra+'## 5. 해석과 한계',1)
    s=section(s,'## 5. 해석과 한계','## 6. 결과 파일','''**방향에는 클래스 관련 신호와 데이터 조건 관련 구조가 함께 나타난다.** DF 정상 source eta²0.307754,spoof padding0.163591과21LA 정상 codec0.168577,spoof padding0.158263은 이 관찰을 구체화한다. 기존 K=4에서 codec NMI가 작았어도 원래 u에 codec 관련 구조가 없다는 뜻은 아니다.

eta²는 관찰 표본에서 조건별 평균 차이가 차지하는 분산 비율이다. 해당 조건을 제거하면 그만큼 탐지 정보가 사라진다는 인과적 기여도가 아니다. source·codec·transmission·duration·padding은 서로 연결될 수 있고,각각의 수치를 독립적인 원인 기여도로 더하지 않는다. 한 transmission 안에 codec이 하나뿐이면 eta²0은 비교 변이가 없다는 뜻이다.

기존 metadata 포함 probe에서도 방향 추가의 LL 이득이 남았다는 근거는 유효하다. 다만 같은 DF OOF 예측을 재사용했으므로 독립 재현이 아니며,측정한 조건을 선형 입력으로 넣는 것이 모든 nuisance를 통제하는 것은 아니다. 방향이 전부 shortcut이라는 결론과 방향이 전부 위조 고유 신호라는 결론을 모두 피한다.

새 strict matching은 DF27쌍/9그룹과21LA10쌍/6그룹으로 최소200쌍·10그룹을 충족하지 못했다. 조건을 같게 만든 표본에서 방향 정보가 남는지에 관한 강한 판정은 아직 불가하다. 작은 cell·적은 화자 수·class 누락은 coverage 파일에 남겨 두었다.

동일 음성의 noise/codec 개입인 R3는 별도 실행 대기/진행 단계다. 아직 결과가 없으므로 이 관찰 통계로 음향적 원인을 확정하거나 원래 attention 제거의 효과를 보고하지 않는다. 이전 K=4와 현재 전체-vector 통계는 탐색 범위가 다르며,t-SNE의 분리 정도를 정량적인 일반화 성능으로 읽지 않는다.''')
    write(p,s)
    # All channel analyses: preserve measured tables and figures, revise claims.
    p=DETAIL/'02_channel_ablation.md';s=banner(p.read_text(encoding='utf8'),'기존 좌표 부분집합의 DF 결과와 새 학습 gate/metric 전이 결과를 분리해 판정한다.')
    a=s.index('## 5. 해석');s=s[:a]+'''## 5. 해석

### 5.1 확인된 관찰

현재 checkpoint의 최종160좌표 중 특정 부분집합만 사용해도 이 DF pilot에서 낮거나 비슷한 EER의 점 추정치를 얻었다. 하위2개 EER2.9222%는 native3.35%보다 낮았지만,차이 CI는0을 포함했다. 부분집합에 따라 성능이 달랐고 T의 높은 단변량 Fisher 순위가 DF의 좋은 cosine 부분집합을 보장하지 않았다.

같은 표본에서 하위2개의 AUROC는0.987659로 전체0.995943보다 낮았다. V EER도0.1907%로 native0%보다 높았다. 전체160개와 두 좌표의 ROC는 교차하며,정상거부율1% 이하에서는 전체 표현이 유리했다. 따라서 현재 EER 지점의 관찰을 전체 성능 동등성 또는 비열등성 입증으로 바꾸지 않는다.

### 5.2 새 gate/metric 실험과의 관계

후속 P1은 T에서 static/dynamic gate와 같은 크기 MLP를 학습하고 V에서 선택했다. dynamic의 DF 평균 EER는3.342593%,native 대비 개선0.007407pp의99% CI는[−0.054719,+0.065027]이다. 21LA pilot은0.80%로 같았다. static과 P2 metric은 epoch0 identity가 선택됐다.

즉,일부 좌표만 읽어도 낮은 EER가 나온다는 관찰이 **입력별로 학습한 gate가 전이 EER를 개선한다는 주장으로 이어지지 않았다.** P1은 c를 고정한 bounded 변환이고 기존 F1은 z/c에 같은 binary mask를 준 점수다. 서로 다른 점수 정의와 학습 범위를 혼동하지 않는다.

### 5.3 아직 확대할 수 없는 결론

- 전체 encoder·160차원 LayerNorm을 계산한 뒤 점수에 쓰는 좌표만 줄였다. 처음부터2차원인 모델·추론 연산량 절감·표현의 본질적 차원을 검증하지 않았다.
- 113·95는 T-Fisher 하위 좌표다. 나머지158개가 모두 잡음이거나 어떤 부분집합에도 불필요하다고 볼 수 없다.
- 1차원 cosine은 연속 각도 대신 부호±1만 남는다. 그 EER는 보간 ROC 값이며 실제 특정 문턱값의 정상거부·spoof통과와 구분한다.
- 같은 숫자의 native V threshold를 모든 부분집합에 적용한 오류 변화와,각 점수별 V threshold를 새로 고정한 운용 비교를 구분한다.
- 10→2→1개 확장은 이미 관찰한 DF 표본에서 이어진 탐색이다. 개별 CI가 이전 탐색 전체를 보정하지 않는다.

### 5.4 신규 panel의 상태

명세의 원래169조건,직교회전3개의25% 대조,각 mask의 V threshold,10,000회 speaker/mask 공동 재표집을 수행 중이다. 전체 완료·검수 전에는 좌표계에 무관한 채널 집중이나21LA 재현을 확정하지 않는다. DF/21LA는67화자그룹을 공유하고 source-family도 미해결이다. 최신 완료 여부와 수치는 [전체 분석](../ebm_frozen_all_results_20261105.md)을 따른다.
''';write(p,s)
    channel_bodies={
    '02_channel_reduction_to10.md':'''**이 DF 표본에서 하위10개를 남긴 cosine EER의 점 추정치는2.9444%였다.** 전체160개의3.35%보다 낮았지만,이를 전체 ROC 또는 통계적 성능 동등성으로 읽지 않는다. 상위10개는5.05%,random10개20회의중앙값은3.6694%여서 어떤10개나 충분했던 것도 아니다.

고정된 옛 native V threshold에서 하위10개의 DF 정상거부는42.80%로 native39.00%보다 높았다. EER 부근의 이득이 같은 운용점의 이득과 다를 수 있다. 각 점수별 V 재보정 비교와도 구별한다.

후속5/3/2/1개 실험은 이미 완료됐다. 하위2개는 EER2.9222%지만 AUROC와 V EER가 나빠졌고,하위1개의 EER는4.345939%였다. 이 문서의10개는 역사적 실행 범위이며 “현재까지 시험한 최소 수”가 아니다.

새 P1 dynamic gate는 두2만 개 전이 평가에서 우월성을 입증하지 못했고,static/metric은 V에서 identity가 선택됐다. 소수 좌표 관찰과 학습 attention의 유용성은 다른 가설이다. 새 F1 회전·공유 bootstrap 전체 묶음은 진행 중이다.

모든 조건에서 원래160차원 encoder/LN은 유지됐다. 처음부터10차원만 생성하는 모델의 성능이나 계산량 감소를 주장하지 않는다. 기존 DF 결과를 반복 관찰한 탐색이며 원발화 family와 전체 탐색 이력의 불확실성이 남는다.''',
    '02_channel_reduction_to2.md':'''**하위2개가 이 DF pilot의 EER 지점에서 낮은 점 추정치2.9222%를 보였다는 관찰은 유효하다.** 그러나 native 대비 차이의 개별95% CI는0을 포함하고,이를 성능 동등성·비열등성 입증으로 바꾸지 않는다. AUROC0.987659와 V EER0.1907%는 각각 native0.995943와0%보다 나빴다.

상위2개 EER6.85%,random2개 중앙값5.5750%였으므로 부분집합 선택이 중요했다. “2개면 충분하다”보다 “이 checkpoint의 특정 두 좌표가 이 표본의 EER에서 작동했다”가 정확하다. 다른158개가 모든 점수·공격·운용점에서 불필요하다는 뜻은 아니다.

후속1개 실험도 완료됐으며 하위1개 EER는4.345939%다. 2개는 최소 필요 차원으로 확인된 값이 아니고,현재 시험 범위의 최소 수도 아니다. 1차원에서는 cosine이±1만 남는 구조적 변화가 있다.

전체160개와 두 좌표의 ROC 교차,낮은 정상거부율에서 전체 표현의 이점,동일 숫자의 옛 threshold에 따른 점수 분포 변화도 함께 읽어야 한다. 원래160차원 encoder/LN은 모두 계산했다.

새 dynamic gate의 DF EER 이득은0.007407pp,99% CI는0을 포함했고21LA EER는 같았다. 이 후속 결과는 두 좌표의 기술적 관찰을 부정하지 않지만,그 관찰만으로 입력별 attention의 전이 개선을 예측할 수 없음을 보여준다. 반복 탐색·미해결 family 의존성은 남아 있다.''',
    '02_channel_reduction_to1_and_explanation.md':'''**전체 신경망이 생성한 요약 좌표 일부에 클래스 차이가 들어 있기 때문에 소수 좌표의 cosine 점수도 작동할 수 있다.** 원음의 정보가 두 개뿐이거나 모델 전체가2차원이었다는 실험은 아니다. 두 좌표의 pooled 상관 약0.973에는 클래스 평균 차이도 포함된다.

하위1개 EER4.345939%는 하위2개의2.9222%보다 높았다. 1차원 cosine은0이 아닌 입력에서 sign(z[j])sign(c[j])로 정확히±1이다. 동점을 나누지 않은 보간 EER와 실제 부호 판정의 정상거부4.45%·spoof통과2.0556%를 구분한다.

두 좌표가 낮은 EER를 보여도 AUROC·V·일부 ROC 구간에서는 전체160개가 좋았다. 모든 정보를 보존했다거나 나머지 좌표가 전부 잡음이라는 결론은 나오지 않는다. 성능이 “유지됐다”는 표현도 이 표본의 EER 점 추정치 범위로 한정한다.

후속 T/V 학습 gate는 전이 EER 개선을 입증하지 못했다. 관찰된 좌표의 중복 가능성과 입력 적응형 gate의 유용성은 별도 질문이다. 새 회전 대조와 엄격한 평가 panel은 진행 중이며,새 방향 loss의 공동 학습은 수행하지 않았다.

기존 DF 표본의 반복 탐색,원발화 family 미해결,저장 예측에 조건부인 CI를 유지한다. 원래160차원 encoder와 LayerNorm을 계산했으므로1/2차원 모델의 학습 성능·추론 연산량 절감을 입증하지 않는다.''',
    '02_why_more_channels_can_worsen_eer.md':'''**확인한 직접적 사실은 좌표 선택 후의 순위 변화와 ROC 교차다.** 하위2개는 EER 부근에서,전체160개는 낮은 정상거부율 구간과 AUROC에서 유리했다. “채널이 많을수록 항상 나빠진다” 또는 “추가158개가 모두 잡음”은 이 결과와 맞지 않는다.

고정 cosine은 좌표를 추가할 때 유용성을 학습해 자동 제외하는 함수가 아니다. 분자와 재정규화가 함께 바뀌므로 순위가 달라질 수 있다. 그중 어떤 음향적·도메인 원인이 작용했는지는 ROC 비교만으로 식별하지 못한다.

새 P1 dynamic gate는 DF EER를 평균0.007407pp 낮췄지만99% 구간이0을 포함했고21LA EER는 같았다. 또한 dynamic의 DF LL은 약간 나빠졌다. 학습 가능한 가중치를 붙였다는 사실만으로 이 ROC trade-off가 해결되는 것은 아니었다.

새 P3 선형 방향 잔차는 반대로 LL을 작게 줄였지만 DF EER는0.05pp 나빠졌다. 이는 ranking·EER 운용점·확률 예측을 분리해 해석해야 한다는 추가 사례다. 한 지표의 부정 결과를 모든 효과 부재로,한 지표의 이득을 전체 성능 개선으로 바꾸지 않는다.

모든 수치는 이미 관찰한 DF pilot과 고정 checkpoint의 탐색이다. 기존160차원 encoder/LN을 유지했고,독립 일반화나 인과 원인·본질적 차원을 증명한 결과가 아니다.'''}
    for name,body in channel_bodies.items():
        p=DETAIL/name;s=banner(p.read_text(encoding='utf8'),'과거 실행의 수치·그림을 보존하고 새 후속 결과에 맞춰 결론과 남은 범위를 수정했다.')
        a=s.index('## 5. 해석');oldtail=s[a:];evidence=oldtail[oldtail.index('근거:'):].strip() if '근거:' in oldtail else ''
        s=s[:a]+'## 5. 해석\n\n'+body+'\n\n'+evidence+'\n';write(p,s)
    # Legacy integrated report: retain historical evidence; align all headlines.
    p=DOCS/'ebm_cached_core_results_20261105.md';s=p.read_text(encoding='utf8');first,rest=s.split('\n',1)
    notice='''

## 최신 결과와의 관계

이 문서의 본문은09:06 KST 완료한 `cached_core_v1`의 **역사적 결과와 방법**이다. 이후의 최신 통합 판정은 [고정 EBM 전체 분석](ebm_frozen_all_results_20261105.md)에 있다. 옛 수치를 최신 모델의 성능으로 바꾸지 않는다.

- DF 내부 방향 probe의 이득은 유효하지만 T→DF 전이 성능이 아니다. 새 P3 선형 전이는 LL을 작게 줄였으나 DF EER는3.35%→3.40%,21LA EER는0.80% 유지였다.
- 소수 채널의 낮은 EER 점 추정치는 전체 ROC·AUROC·V 또는 통계적 동등성의 입증이 아니다. 후속1개 실험까지 완료됐고,새 gate의 전이 우월성은 확인하지 못했다.
- 옛106쌍은 source×codec matching이다. 새 same-speaker 포함 F2는DF27쌍/9그룹,21LA10쌍/6그룹으로 최소 표본 기준에 미달했다.
- h/z probe 접근성 차이와 LayerNorm의 인과적 정보 손실을 구분한다. 새 scalar+u+h/noise P4의 최종 결과는 아직 없다.
- 새 F4는 두 데이터셋의 전체 u에 대한 eta²와 padding/duration 연관을 계산했다. 옛 K=4 NMI와 같은 지표가 아니다.

아래의 “Q는 열지 않았다”,“새 추론이 없었다”는 **이09시 실행의 범위**다. 이후 Q평가와 새 GPU 추론 작업이 시작된 현재 상태 전체를 뜻하지 않는다. 현재 진행·대기는 주 명세0절을 따른다.

## 기존 실행의 결과
'''
    s=first+notice+rest
    s=s.replace('**DF 내부에서는 코사인·크기만 쓰는 진단용 분류기보다 잔여 방향을 추가한 분류기가 좋았다. 반면 T에서 고른 상위 채널을 제거해도 기존 코사인 EER는 거의 달라지지 않았다.**','**기존 DF 내부 probe에서는 방향 추가의 이득이 관찰됐고,시험한 상위 채널 제거의 EER 점 추정치는 native와 가까웠다.** 이 관찰을 전이 개선이나 성능 동등성의 증명으로 확대하지 않는다.')
    s=s.replace('**판정:** 시험한 부분집합에서는 기존 코사인의 DF pilot EER를 유지하는 데 160개 좌표를 모두 남길 필요가 없다는 근거를 얻었다.', '**판정:** 시험한 일부 부분집합에서160개 native와 가까운 또는 낮은 DF pilot EER의 점 추정치를 얻었다. 사전 비열등성·동등성 검정을 수행한 것은 아니다.')
    write(p,s)
    # Entry point: rewrite the entire index around current interpretation.
    p=DETAIL/'README.md';s='''# 방향·채널 결과 문서 안내

최신 통합 분석은 [고정 EBM 전체 결과와 연구 가설](../ebm_frozen_all_results_20261105.md)이다. 아래 문서는 기존 DF 캐시 진단의 방법·수치를 보존하면서 새 T/V 전이·matching·조건 분석에 맞춰 해석을 수정했다. 실험 수행 완료와 가설 지지를 구분한다.

| 문서 | 현재 읽어야 할 결론 |
| --- | --- |
| [① 잔여 방향](01_direction.md) | DF 내부 접근성 이득과 T/V 전이의 EER/LL 결과를 분리한다. 새 strict matching은 표본 부족이다. |
| [② 채널 부분집합](02_channel_ablation.md) | 특정 좌표의 낮은 EER와 학습 gate의 전이 개선은 다른 주장이다. 전체 성능 동등성은 입증되지 않았다. |
| [80→10개](02_channel_reduction_to10.md) | 하위10개 DF EER2.9444%는 해당 표본·점수의 관찰이다. 후속1개까지 이미 시험했다. |
| [5·3·2개](02_channel_reduction_to2.md) | 하위2개 EER2.9222%지만 AUROC·V와 일부 ROC 구간은 악화됐다. |
| [1개와 두 좌표 설명](02_channel_reduction_to1_and_explanation.md) | 하위1개 EER4.345939%;1D cosine은부호±1이며두좌표가모델전체차원은아니다. |
| [ROC 교차](02_why_more_channels_can_worsen_eer.md) | EER·AUROC·운용점·확률 log-loss가 서로 다른 질문임을 구분한다. |
| [③ LayerNorm](03_layernorm.md) | h가 옛 동일 probe에서 더 잘 읽혔으나 정보 손실의 인과 증명은 아니다. 새 증분 P4는 미완료다. |
| [④ 데이터 조건](04_conditions.md) | K=4 NMI와 전체 u의 eta²를 구분하며 source/codec/padding 연관을 인과 효과로 읽지 않는다. |
| [관련 문헌과 우리 결과](02_related_work_fulltext_review_20261105.md) | 문헌의 보고 결과와 우리 후속 결과를 분리하고 미실행 목록을 정정했다. |
| [실제 좌표·PCA/t-SNE 캡션](figures_clusters/captions.md) | 그림 표본과평가표본을구분하고,투영분리로성능·군집수·차원을확정하지않는다. |
| [저자 설명용 그림 캡션](figures_author/captions.md) | 기존DF부분집합그림이며새21LA/gate실험결과가아니다. |

새 P1/P2는 실제 T/V12fit을 완료했다. dynamic의 DF EER 평균 개선0.007407pp의99% CI는0을 포함했고21LA는0.80%로 같았다. static/metric은 V에서epoch0 identity를 선택했다. 새 P3 선형 전이는 LL을 작게 줄였으나 EER는 개선하지 못했다.

기존 DF 내부1.20%/0.60%는 새 probe를 DF에서 학습한 결과다. T에서만 학습한 검출기의 일반화 성능이나 새 EBM의 성능으로 사용하지 않는다. 모든 문서의 scalar baseline은 원래 EBM과 다를 수 있다. 원래 native는cos(z,c)만 쓰며,크기는 별도 판별 입력이 아니다.

진행 중인 F1 전체·P3-B/P4·전체 데이터 추출·R2/R3는 [실행 체크표](../ebm_frozen_model_experiments_20261105.md)를 따른다. 각 문서의 새 해석은 작성 시점의 검수 완료 자료에 한정되며,이후 완료 결과는 최신 통합 분석에 반영된다.

기존 [통합 결과와 검수 이력](../ebm_cached_core_results_20261105.md), [기존 독립 검수](../../../artifacts/geometry_cache/local_analysis/cached_core_v1/output_audit.json), [새 실험 검수](../../../artifacts/geometry_cache/local_analysis/frozen_all_v1/verification.json), [P3 전이 검수](../../../artifacts/geometry_cache/local_analysis/frozen_all_v1/P3_P4/A/verification.json)를 함께 확인할 수 있다.
''';write(p,s)
    # Literature: primary paper summaries are preserved, update our deductions.
    p=DETAIL/'02_related_work_fulltext_review_20261105.md';s=banner(p.read_text(encoding='utf8'),'문헌별 원문 보고는 기존 검토 범위로 보존한다. 우리 실험의 후속 결과·판정·미완료 목록만 최신화했으며 이번에 새 문헌 검색을 수행하지 않았다.')
    s=section(s,'## 7. 다음 실험의 우선순위','## 8. EBM 저자에게 결과를 설명할 때 사용할 수 있는 범위','''초기 문헌 검토 뒤 실제 동결 실험이 진행됐다. 따라서 아래 항목을 모두 “아직 하지 않은 실험”으로 표시하지 않는다.

| 질문 | 현재 수행 범위 | 남은 판정 |
| --- | --- | --- |
| 좌표 수와 전체 ROC의 관계 | DF160→1개,AUROC와ROC교차 및 독립 검수 완료 | 다른 좌표계·두 평가셋의 신규F1전체 통계 진행 중 |
| 거리의 비등방성 | T정상 covariance F3와supervised P2 완료 | V선택은identity;모든metric의가능성기각은아님 |
| 입력별 channel gate | T/V에서static/dynamic/같은크기MLP×seed3 완료 | 두pilot에서우월성미확인;공동학습은미시험 |
| 방향정보 접근성/전이 | 기존DF내부진단과새P3-A 완료 | P3-A는LL소폭개선/EER미개선;새내부P3-B진행 중 |
| source/codec/padding | DF/21LA전체u의class별eta²·조건별오류 완료 | 관찰적연관이며R3동일음성개입은대기/진행 |
| LN전후 정보 | 기존h/z동일probe비교 완료 | 새scalar+u의h/noise증분P4진행 중 |
| 평가범위 | 전체21LAcache/native와동결head보조평가,Q평가 완료 | 19LA·fullDF확장및후속검수진행 |

P1/P2의 부정적 전이 결과만으로 joint attention 또는 새 방향 loss까지 기각할 수 없다. 반대로 기존DF내부probe의 이득으로 이 방법들이 개선될 것이라 보장할 수도 없다. 모듈 용량·변환폭·objective·V선택력·도메인 차이와 표현 공동 학습을 구분해야 한다.

직교 회전 후 full cosine은 불변이지만 부분집합 점수는 달라질 수 있다. 회전 대조는 특정 좌표 효과와 표현의 본질적 차원을 구분하는 도구이며,결과가 끝나기 전에 좌표 불변성을 주장하지 않는다. 미사용 DF부분도 같은 benchmark와 원발화/화자 의존성·과거관찰의 한계가 있어 완전한blind 독립검증으로 부르지 않는다.''')
    s=s.replace('## 5. 이번 결과를 설명할 수 있는 가설과 아직 구분되지 않은 것','## 5. 이번 결과를 설명할 수 있는 가설과 아직 구분되지 않은 것\n\n후속 P1/P2/P3-A와F2/F3/F4의 완료 결과를 함께 고려해야 한다. dynamic gate의 전이EER 우월성은 미확인이고,선형 방향 잔차는 작은LL 이득과 EER 미개선이 함께 관찰됐다. source/codec/padding은class별전체u와연관됐다. 아래 문헌의 관련 사례를 우리 결과의 확정 원인이나 개선 보장으로 사용하지 않는다.')
    write(p,s)
    # Figure captions: retain the working image embeds, correct inference scope.
    for name in ['figures_author/captions.md','figures_clusters/captions.md']:
        p=DETAIL/name;s=p.read_text(encoding='utf8');first,rest=s.split('\n',1)
        notice='''

> 해석 갱신: 아래 이미지는 기존 DF pilot의 좌표·ROC 또는 투영 그림이다. 새21LA·gate·metric·P3-B/P4 결과를 그린 그림이 아니다. [최신 통합 분석](../../ebm_frozen_all_results_20261105.md)의 전이/내부 진단 구분을 함께 적용한다. 이미지와 표시 표본은 변경하지 않았다.
'''
        s=first+notice+rest
        s+='''
## 후속 실험과 함께 읽는 기준

특정 부분집합의 낮은 EER 점 추정치는 전체 성능 동등성이나 본질적 차원2의 증명이 아니다. 원래160차원 encoder/LN을 계산했고,전체 AUROC와 낮은 정상거부율 구간에서는 전체 표현이 유리했다. 후속 학습 dynamic gate도 두pilot 전이EER의 우월성을 입증하지 못했다.

그림의 투영·겹침은 정량 검정이나 원인 분석을 대체하지 않는다. 특히 원래평가20,000개와균형시각화4,000개,같은발화의여러점수조건을독립표본으로합산하지않는다. PCA 큰분산보존을판별정보보존으로동일시하지않는다.
''';write(p,s)
    manifest=dict(KST=STAMP,scope='all related result interpretation Markdown; historical metrics and existing figures preserved',new_experiments=False,modified=CHANGED)
    (BACKUP/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('Revised interpretation documents:',len(CHANGED))

if __name__=='__main__':main()
