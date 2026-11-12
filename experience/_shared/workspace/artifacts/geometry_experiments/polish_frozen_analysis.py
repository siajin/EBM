"""Reviewed prose for the durable result analysis; no numeric transformations."""
import re

PARAGRAPHS={
'길이를 1로 정규화한 최종 표현을': '길이를 1로 정규화한 최종 표현을 `x=z/||z||`, 단위 정상 centroid를 `c`라 두면 원래 점수는 `s=xᵀc`다. c는 평가 데이터의 평균으로 바꾸지 않았다. 잔여 성분은 `r=x−sc`, 잔여 방향은 `u=r/||r||`다. `||r||²=1−s²`이므로 잔여 크기는 s와 별개의 추가 정보가 아니다.',
'R2/R3의각2,000개': 'R2/R3의 각 2,000개(정상200/spoof1,800)는 해당 pilot에 포함된다. 전처리 QC128개(32/96)는 V에서 정했다. 표본 선택에는 점수나 h/z를 쓰지 않았다. DF와21LA는67개의 asvspoof 화자 그룹을 공유하므로 완전히 독립된 반복 검증으로 세지 않는다. 원발화·합성·재인코딩 family 연결은 미해결이다.',
'전체21LA native EER는': '전체21LA native EER는1.0535392921%다. pilot의0.80%와 표본 구성이 다르므로 어느 쪽이 다른 쪽을 대체하지 않는다.',
'하위2개의 native 대비 EER 차이': '하위2개의 native 대비 EER 차이 구간은0을 포함했다. 상위2개 EER는6.85%,random2개20회의 중앙값은5.5750%였다. 특정 두 좌표가 작동한 것과 아무 두 좌표나 충분한 것은 다르다. 하위2개의 V EER는0.1907%로 native0%보다 높았다.',
'169조건의원래좌표': '원래 좌표의169조건과 직교 회전3개의25% panel,10,000회 speaker/mask 재표집이 진행 중이다. 검수된 기존 DF 실험을 인용하되,새 전체 panel과 회전 대조가 완료됐다고 표시하지 않는다.',
'옛source×codec': '기존 source×codec matching106쌍과 새 strict matching을 구분한다. 기존106쌍은 같은 화자를 요구하지 않았고 caliper도0.01/0.002였다. 새 F2는 DF에서 source×speaker×codec,21LA에서 speaker×codec×transmission을 같게 하고 caliper0.02/0.002를 사용한다. 표본 수 차이를 모순이나 재현 실패로 해석하지 않는다.',
'최소200쌍·10그룹을모든조건이': '모든 조건이 최소200쌍·10그룹 기준을 충족하지 못했다. 방향 moment와 MMD 계산은 완료됐지만 주 가설의 판정은 불가하다. biased MMD²의 bootstrap 구간이0보다 크다고 해서 조건부 방향 차이가 입증되는 것은 아니다. biased 추정량은 유한 표본에서 양의 기댓값을 가질 수 있고,여기 CI는 관찰한 두 표본을 재표집한 효과 변동 구간이지 동일분포 null 검정이 아니다. matching도 정확한 s/norm 일치가 아닌 근사다.',
'정상/스푸프의resultant': '정상/스푸프의 resultant length,centered covariance의 participation ratio(PR),uncentered2차 moment의 PR도 저장했다. 중심화 여부가 다른 수치를 같은 “내재 차원”으로 읽지 않는다. 작은 선택 표본의 PR을 전체 표현의 최소 필요 차원으로 해석하지 않는다.',
'단위 x,c에서는': '단위 x,c에서는 `||x−c||²=2−2s`다. 따라서 단위 공간의 단순 유클리드 거리로 바꾸기만 하면 순위·EER·AUROC는 원래 cosine과 같다. raw z의 거리나 새 중심은 norm·center까지 바꾸는 별도 실험이다.',
'F3은T정상': 'F3은 T 정상 unit 표현의 공분산으로 A를 정하고 `S_A=1−½(x−c)ᵀA(x−c)`를 썼다. 중심 c는 고정했다. diagonal/full과 shrinkage 후보를 V EER로 선택했다. 모든 후보의 V EER가0%여서,사전에 정한 큰 gamma 우선 규칙에 따라 identity(gamma1)가 선택됐다.',
'diagonal gamma0.01의DF': 'diagonal gamma0.01의 DF EER3.05%는 관찰한 후보값이다. native 대비 개선0.30pp의 개별95% CI는[0.065818,0.391215]이지만,8비교 보정 구간은[0,0.458775]로0을 포함한다. V에서 고른 주 모델도 아니다. 그러므로 “유클리드/마할라노비스가 개선됐다”는 확정 결론으로 쓰지 않는다.',
'P2는정상공분산': 'P2는 정상 공분산의 비지도 추정과 달리 정답을 이용해 bounded diagonal A를 학습했다. 세 seed 모두 V에서 epoch0이 선택돼 최종 A=I다. [0,0]의 개선 CI는 선택된 점수가 같다는 대수적 결과이지,어떤 metric도 소용없다는 정밀한 부정 증거가 아니다.',
'원래EBM/centroid를고정하고': '원래 EBM/centroid를 고정하고 unit z 뒤에 작은 모듈을 붙였다. static은160파라미터,dynamic과 잔차 MLP는 각5,296파라미터다. gate 범위는0.5~1.5이며 identity 초기화와 변환 규제를 썼다. 학습 objective는 class-balanced weighted OC이고,V 선택에는 모든 head에 같은 T 고정 링크를 적용한 balanced LL을 썼다. 각 seed에서 동일한 epoch/batch 순서를 공유했다.',
'static과metric은epoch0': 'static과 metric은 epoch0,dynamic은 epoch1,MLP는 seed101/102/103 순서로11/9/11epoch가 선택됐다. 아래 CI는 각 draw에서 seed별 paired EER 차이를 먼저 계산한 뒤 평균한 것이다. 예측 평균 ensemble의 성능이 아니다.',
'양수는EER개선이다. dynamic대native': '양수는 EER 개선이다. dynamic 대 native의 DF0.007407pp는 실용 효과 참고값0.1675pp보다도 작고 CI가0을 포함한다. DF LL은 native0.563682보다 dynamic 평균0.565309로 나빠졌다. 21LA EER는 같고 LL도 소폭 나빠졌다. 이 조건의 dynamic 우월성은 확인하지 못했다. 다만 동일성/비열등성을 따로 설계해 입증한 것도 아니다.',
'T평균gate·실제gate': 'T 평균 gate·실제 gate·T에서 정한 score/norm 구간 안의 gate 재배치 개입도 저장했다. 이 개입은 비현실적인 중간 조합을 만들 수 있다. gate 값이나 shuffle 민감도만으로 인과적 채널 중요도를 정하지 않는다.',
'**부정결론의범위는이bounded': '**부정 결론의 범위는 이 bounded posthoc 구조와 fit/선택 규칙에 한정된다.** V가 매우 쉽고,원래 표현을 고정했으며,long 평가 캐시만 사용했다. 새 encoder/classifier와 attention을 공동 학습하거나 다른 목적함수로 학습하는 모든 attention 방식까지 기각한 것은 아니다.',
'기존cached_core_v1은DF': '기존 cached_core_v1은 DF20,000개에서5×3 nested grouped CV를 수행했다. spline(s)+lognorm+valid의 LL0.098882/EER3.5056%에 방향 u를 더하면 LL0.038342/EER1.20%였고,같은 차원 noise는 LL0.123376/EER4.20%였다. 출처·codec·길이·padding까지 입력한 대조에서도 방향 추가의 LL 감소0.033961에 대한 개별95% CI는[0.021772,0.047048]이었다.',
'이결과는정한DF내부모델': '이 결과는 정한 DF 내부 모델 계열에서 추가 방향 관련 신호가 held-out speaker에게 읽힌다는 근거다. u는 z의 결정적 변환이므로 새 정보를 생성한 것은 아니다. 특징 변환이 probe의 접근성·수치 조건을 바꾼 영향도 가능하다. 정확한 conditional mutual information이나 모든 nuisance 제거를 증명하지 않는다.',
'새P3은s에대해': '새 P3은 고정 norm에서 s에 대해 단조인 기본 logit에 선형 또는 hidden16 잔여 방향을 더했다. norm/valid가 다르면 전체 순위는 바뀔 수 있다. ridge 두 후보·epoch0 포함 V 선택·seed3개를 사용했고,s/n을 잔차 입력에 다시 넣지 않았다. 기존 DF spline 진단과 모델·fit 자료·규제·선택 규칙이 다르다.',
'선형방향잔차의LL감소는': '선형 방향 잔차의 LL 감소는 각각0.002387855와0.000216622로,개별95% 구간이0보다 크다. 따라서 작은 확률 예측 개선도 기록해야 한다. 반면 DF EER는0.05pp 악화됐고21LA는 동일하다. 이 구간은 고정된 fit·선택에 조건부이며,전체 탐색의 다중 비교를 보정한 것은 아니다.',
'세seed 중선형direction': '세 seed 중 선형 direction의 seed101만 V에서 비영 잔차가 선택됐고 다른 두 seed는 epoch0이었다. scalar/noise/MLP 잔차도 최종 선택은 epoch0이다. LL 효과가 세 번 독립적으로 같이 재현됐다고 쓰지 않는다.',
'DF와21LA각2만개에서별도의': 'DF와21LA 각2만 개에서 별도의5outer×3inner speaker-grouped CV를 실행 중이다. 모든 seed·fold·variant와 독립 검수가 끝나기 전에는 일부 완료 fold를 합쳐 전체 EER나 최종 결론을 만들지 않는다. 이 새 모델 계열의 내부 진단 재현과21LA 내부 방향 접근성은 아직 판정하지 않는다.',
'기존DF내부동일L2': '기존 DF 내부의 동일 L2 logistic 비교에서 scalar+h는 LL0.040671/EER1.4944%,scalar+z는 LL0.064016/EER1.7000%였다. LL 감소0.023345의 개별95% CI는[0.010456,0.042250]이었다. 이 수치는 원래 centroid cosine이 아닌 새 probe 점수의 결과다.',
'h는최종Linear출력': 'h는 최종 Linear 출력,z는 그 뒤 LayerNorm 출력이다. h160을 z160 대신 읽었을 때 정한 probe에 더 유리했다는 근거는 있다. 그러나 h가 z에 비해 얼마나 많은 정보를 가지는지,그 차이가 기존 탐지기의 오류 원인인지까지 이 결과로 측정하지 못한다.',
'LayerNorm은발화내공통shift': 'LayerNorm은 발화 내 공통 shift에는 정확히 불변이다. 양의 scale에는 epsilon>0일 때 일반적으로 정확히 불변하지 않다. 현재 h 분산은 epsilon1e-12보다 매우 커서 scale의 영향이 강하게 정규화된다. “평균·분산 정보가 모두 수학적으로 완전히 없어진다”는 표현은 피한다.',
'신규P4의scalar+u': '신규 P4의 scalar+u+h-stat/h160 대 scalar+u+동일 차원 noise2/160 비교는 아직 최종 결과가 없다. 기존 h/z 비교가 이 증분 대조까지 완료한 것처럼 쓰지 않는다.',
'기존T-fit K=4방향군집': '기존 T-fit K=4 방향 군집에서는 DF label NMI0.264455,source0.118216,codec0.004659였다. 정상 내부 source NMI는0.343788이었다. 새 F4는 군집 번호로 축약하지 않고 u 전체의 class별 벡터 분산 비율 eta²를 계산했다. NMI와 eta²는 정의가 다르므로 수치의 크기를 직접 비교하지 않는다.',
'DF정상source eta²': 'DF 정상 source eta²0.307754와 spoof padding eta²0.163591,21LA 정상 codec eta²0.168577와 spoof padding eta²0.158263이 관찰됐다. 이는 조건별 평균 차이가 관찰한 방향 분산에서 차지하는 비율이다. 조건을 조작했을 때 제거되는 인과적 분산 비율이나 독립적인 원인 기여도는 아니다.',
'T-fit PCA1과spoof': 'T-fit PCA1과 spoof padding의 Spearman은 DF−0.600272,21LA−0.577569였다. duration은 각각+0.601706,+0.576446이었다. PCA 축 부호는 임의적이고 duration/padding은 고정 crop 규칙으로 연결되므로 두 개의 독립 발견으로 세지 않는다.',
'transmission별codec분석': 'transmission별 codec 분석에서 codec 범주가 하나뿐이면 eta²0은 “그 stratum에서 비교할 codec 변이가 없음”을 뜻한다. codec이 영향 없다는 음향적 증명이 아니다. 각 cell의 정상/spoof 수·화자 수·20개 미만 표시와 조건별 native 오류도 저장했다.',
'기존metadata포함probe': '기존 metadata 포함 probe에서 방향이 추가로 도움됐다고 해도 측정하지 않은 조건과 조건 간 비선형 관계를 모두 제거한 것은 아니다. 반대로 조건 연관이 있다고 해서 그 방향이 전부 불필요한 shortcut이라는 뜻도 아니다. 동일 음성 개입 R3가 이 관찰을 보완한다.',
'각완료자료에는T/V에서이미': '각 완료 자료에는 T/V에서 이미 고른 head들을 추가 fit 없이 적용했다. Q 특징 추출은 T/V head 실행과 병행했지만 Q 결과로 선택하지 않았다. DF 전체는 기존20,000개와 미추출513,928개의 전처리/RNG 필드를 맞춰533,928개로 결합한다. 아직 추출 중인 행을 최종 성능의 표본 수로 세지 않는다.',
'전체21LA에서dynamic': '전체21LA에서 dynamic의 seed평균 EER는1.053539%로 native와 같고,MLP는1.053289%였다. 이 극소 차이를 새 개선 근거로 선택하지 않는다. 전체 평가는 pilot을 포함하는 보조 적용이며 새 독립 반복이 아니다.',
'R2의layer/time attention2×2': 'R2의 layer/time attention2×2 네 조건과 R3의 동일 base noise20/10dB·Opus32k 재압축은 대기/실행 단계이며 최종 분석 자료가 아직 없다. 원래 attention을 제거해도 괜찮다거나,noise/codec에 방향이 안정적이라고 결론내리지 않는다. V128개 전처리 QC와 native 복제·원래 state 불변을 통과한 완료 결과만 추후 반영한다.',
'Opus조건은이미압축된': 'Opus 조건은 이미 압축된 음성의 추가 재압축일 수 있다. small2,000개는 pilot에 포함되고 개입 세 조건도 같은 원발화를 공유한다. 이들을 독립 표본으로 합산하지 않는다.',
'사용자의“원래모델이그정보': '사용자의 “원래 모델이 그 정보를 활용하도록 학습되지 않아서 동결 후처리가 불리했을 수 있다”는 설명은 가능한 가설이다. 그러나 현재 결과만으로 그것을 원인으로 선택할 수는 없다. domain shift,V 선택력,제한된 변환 폭,objective/regularization,표현의 배치·nuisance 등 다른 설명도 남는다. 이를 구분하려면 같은 초기화·데이터·seed·optimizer·선택 규칙 아래에서 표현 공동 학습 비교가 필요하다.',
'새loss를설계할때는': '새 loss를 설계할 때는 여러 spoof 공격을 하나의 임의 방향으로 모으는 것이 목표인지,정상의 국소 분포에서 멀어지게 하는 것인지,공격별 여러 방위를 허용하는 것인지부터 구분해야 한다. 현재 실험은 이 선택 중 하나가 좋다는 증거를 제공하지 않았다.',
'P1/P2주5비교는': 'P1/P2의 주 family는 DF의5개 비교다. 각99% CI로 이 family의95% 범위를 보수적으로 관리했다. 21LA는 보조 평가이며,두 셋의10개 비교 전체를 같은95% family로 통제했다고 주장하지 않는다. F3 후보8개는 별도 보정했고,기존 cached_core는2,000회 개별95% 구간,새 F/P는10,000회 source×speaker paired 재표집이다. 서로 다른 구간 수준과 비교 family를 혼용하지 않는다.',
'CI는고정된T순위': 'CI는 고정된 T 순위·선택된 head·저장된 OOF 예측에 조건부다. 재표집마다 전체 fit/CV/모델 선택을 다시 하지 않으므로 학습 표본과 선택의 전체 불확실성은 포함하지 않는다. 세 seed만으로 학습 변동의 모집단을 정밀하게 추론하지 않는다. 반복 탐색한 DF와 과거 benchmark 관찰을 독립 확인으로 되돌릴 수도 없다.',
'해당검토는저장된결과를': '이 검토는 저장된 결과를 재해석한 작업이며 실행 중인 실험을 중단·재시작하거나 추가 EBM 학습을 하지 않았다. 원음·연구실 원본 소스·checkpoint를 새로 로컬에 복사하지 않았다.'
}
HEADINGS={
'## 8. P1: static·dynamic gate와같은크기MLP':'## 8. P1: static·dynamic gate와 같은 크기 MLP',
'## 9. P3: 방향정보의접근성과전이':'## 9. P3: 방향 정보의 접근성과 전이',
'### 9.1 기존DF내부진단':'### 9.1 기존 DF 내부 진단',
'### 9.2 새T/V전이평가':'### 9.2 새 T/V 전이 평가',
'### 9.3 신규데이터셋내부CV':'### 9.3 신규 데이터셋 내부 CV',
'## 10. P4: LayerNorm이정보를없앴는가':'## 10. P4: LayerNorm이 정보를 없앴는가',
'## 11. F4: 클래스구조와데이터조건':'## 11. F4: 클래스 구조와 데이터 조건',
'## 12. R1/R2/R3: 새추론결과의상태':'## 12. R1/R2/R3: 새 추론 결과의 상태',
'### 12.1 평가범위확장':'### 12.1 평가 범위 확장',
'### 12.2 원래attention제거·입력변형':'### 12.2 원래 attention 제거·입력 변형',
'## 13. 연구아이디어별판정과다음해석':'## 13. 연구 아이디어별 판정과 다음 해석',
'## 14. 통계·검수·근거파일':'## 14. 통계·검수·근거 파일'}

def polish(text):
    output=[]
    for line in text.splitlines():
        line=HEADINGS.get(line,line)
        for prefix,body in PARAGRAPHS.items():
            if line.startswith(prefix):line=body;break
        if not line.startswith('|'):
            # Punctuation spacing only; leave values, units and URLs intact.
            line=re.sub(r',(?=[가-힣A-Za-z“])',', ',line)
        if line.startswith('완료범위의새검수는'):
            line=line.replace('완료범위의새검수는','완료 범위의 새 검수는 ').replace('개metric/','개 metric / ').replace('개metric/','개 metric / ').replace('개metric','개 metric').replace('check, P3전이는','check, P3 전이는 ').replace('후보fit을독립확인했다','후보 fit을 독립 확인했다').replace('독립NumPyhead수식','독립 NumPy head 수식').replace('full tied ROC EER,rank AUC,ID정렬·V선택을사용한다','full tied ROC EER, rank AUC, ID 정렬·V 선택을 사용한다').replace('이수치는현재완료된검수범위이며,진행중F1/P3-B/P4/R2/R3까지모두검증했다는뜻이아니다','이 수치는 현재 완료된 검수 범위이며, 진행 중인 F1/P3-B/P4/R2/R3까지 모두 검증했다는 뜻이 아니다')
        output.append(line)
    text='\n'.join(output)+'\n'
    definitions='''### 지표를 읽는 기준

- EER는 두 오류율이 만나는 운용 구간의 값으로 낮을수록 좋다. AUROC는 전체 점수 순위의 요약으로 높을수록 좋다.
- Balanced LL은 정상과 spoof의 평균 로그 손실을 각각50%씩 반영한다. 두 클래스의 실제 표본 비율을 그대로 평균한 손실과 다르며 낮을수록 좋다. 확률은[1e-7,1−1e-7]로 제한한다.
- 개선량은 baseline EER−candidate EER다. 예를 들어3.35%→3.05%는0.30%p(pp)의 절대 감소다.
- CI가0을 포함하면 이 절차로 개선을 확정하지 못했다는 뜻이다. 동등성이나 효과 부재가 입증됐다는 뜻은 아니다. 끝점이0인 구간도 엄격히 양의 개선 구간으로 읽지 않는다.

'''
    text=text.replace('## 4. F0: 손실의 불변성이 증명한 범위',definitions+'## 4. F0: 손실의 불변성이 증명한 범위',1)
    text=text.replace('## 8. P1: static·dynamic gate와 같은 크기 MLP','기존 native_preserving_v1의 보정 family9개와 새 F3의 후보8개는 범위가 다르므로 보정 CI 끝점이 다를 수 있다. 기존 결과와 새 결과를 같은 구간으로 섞어 쓰지 않는다. F3 후보들은 공통 T 확률 링크를 쓰므로,점수의 LL은 별도 확률 재보정의 결과가 아니다.\n\n## 8. P1: static·dynamic gate와 같은 크기 MLP',1)
    return text
