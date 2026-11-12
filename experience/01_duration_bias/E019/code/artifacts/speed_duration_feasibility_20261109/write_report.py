"""Append reviewed speed feasibility findings to the existing experiment report."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
REPORT=PROJECT/'docs/experiments/ebm_B_C_all_checkpoint_evaluation_20261109.md'
FIG=REPORT.parent/'figures/speed_duration_feasibility_20261109'
FIG.mkdir(parents=True,exist_ok=True)
metrics=pd.read_csv(HERE/'probe/metrics.csv')
rates=pd.read_csv(HERE/'metadata/rate_distribution.csv')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(10,4.7))
phases=['dev','19LA','21LA','21DF'];x=np.arange(4);w=.24
for offset,condition,label,color in [(-1,'native_cap','Original cap, no stretch','#2878a2'),(0,'norm_full4','Full audio to 4.0375 s','#d17928'),(1,'roundtrip_cap','Slow then restore duration','#6b5898')]:
    values=[float(metrics.loc[(metrics.phase==p)&(metrics['head']=='B')&(metrics.condition==condition),'diagnostic_EER_percent'].iloc[0]) for p in phases]
    bars=ax.bar(x+offset*w,values,width=w,label=label,color=color)
    ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=10)
ax.set(xticks=x,xticklabels=phases,ylabel='Diagnostic EER (%)',ylim=(0,50),title='Frozen B: phase-vocoder processing harms class separation')
ax.legend(frameon=False,loc='upper left',ncol=1)
fig.text(.1,.01,'128 utterances per set: 64 genuine / 64 spoof, length-stratified. Not benchmark EER.',fontsize=10)
fig.tight_layout(rect=(0,.04,1,1));fig.savefig(FIG/'phase_vocoder_B_diagnostic.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4.5));datasets=['train','dev','19LA','21LA','21DF'];x=np.arange(len(datasets));w=.34
for offset,cls,label,color in [(-.5,'bonafide','Genuine','#2878a2'),(.5,'spoof','Spoof','#d17928')]:
    values=[float(rates.loc[(rates.dataset==d)&(rates['class']==cls),'rate_lt_0.5_percent'].iloc[0]) for d in datasets]
    bars=ax.bar(x+offset*w,values,width=w,label=label,color=color);ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=10)
ax.set(xticks=x,xticklabels=datasets,ylabel='Within-class share (%)',ylim=(0,45),title='Share requiring more than 2x duration expansion')
ax.legend(frameon=False);fig.tight_layout();fig.savefig(FIG/'extreme_slowdown_class_share.png',dpi=170);plt.close(fig)

rate_table='| 집합·class | 배율 중앙값 | 5–95% | rate<0.5 비율 |\n|---|---:|---:|---:|\n'
for d in ['train','19LA','21LA','21DF']:
    for cls in ['bonafide','spoof']:
        r=rates[(rates.dataset==d)&(rates['class']==cls)].iloc[0]
        rate_table+=f"| {d} {'정상' if cls=='bonafide' else 'spoof'} | {r['rate_q0.5']:.3f} | {r['rate_q0.05']:.3f}–{r['rate_q0.95']:.3f} | {r['rate_lt_0.5_percent']:.2f}% |\n"
probe_table='| 집합 | 모델 | 비반복 cap | 원길이 | 0.8배 | 1.25배 | 전체→4.0375초 | 감속→길이 복원 |\n|---|---|---:|---:|---:|---:|---:|---:|\n'
for phase in phases:
    for head in ['legacy240','B','C']:
        group=metrics[(metrics.phase==phase)&(metrics['head']==head)].set_index('condition')
        values=[f"{group.loc[c,'diagnostic_EER_percent']:.4f}" for c in ['native_cap','full_native','slow_cap','fast_cap','norm_full4','roundtrip_cap']]
        probe_table+='| '+phase+' | '+head+' | '+' | '.join(values)+' |\n'

section='''

## 속도·길이 정규화 및 속도 일관성 학습의 타당성 점검 — 2026-11-09

검토 결론: 완만한 속도 변형에 대한 판정 일관성은 검증할 가치가 있다. 그러나 전 음성을 무제한 배율로 4.0375초에 맞추고 바로 speed-BPL을 적용하는 설계는 보류한다. 길이와 발화속도는 다르고, 정확히 같은 길이로 다시 정규화한 두 전역속도 view는 속도차이가 상쇄된다. 현재 frozen 모델 진단에서는 phase-vocoder가 정상까지 spoof 방향으로 크게 이동시켰다. 이는 재학습 불가능의 증거가 아니라 전처리·손실을 분리해야 한다는 근거다.

### 길이를 맞추는 배율과 말하기 속도는 다르다

목표 T=64600/16000=4.0375초(4.04초와 2.5ms 차이), 원길이 d일 때 librosa speed-rate r=d/T이다. r<1이면 감속, r>1이면 가속한다. 최종 음절/초는 음절 수/T이므로 서로 다른 발화의 말하기 속도를 같게 만든 것은 아니다. 긴 내용과 무음까지 함께 압축·확장한다. 동일 내용 x에 서로 다른 전역 rate를 적용하고 모두 T로 재정규화하면 이상적인 시간변환의 최종 rate는 d/T로 동일해진다. 남는 파형 차이는 처리 이력의 비가역 DSP 효과일 수 있다.

전체 내용 보존·서로 다른 전역 속도·정확한 동일 길이를 동시에 만족시킬 수 없다. 속도 불변성 실험에서는 길이가 다른 view를 개별 처리/마스크로 받거나, 정확한 동일 길이가 필수라면 평균 속도를 유지하는 국소 time warp를 별도 가설로 정의해야 한다. 후자는 구간 접합 artifact를 다시 통제해야 한다.

### 실제 필요한 배율의 클래스·도메인 차이

'''+rate_table+'''
전체 고유 발화 metadata 전수 계산이며 sampler 노출 가중은 하지 않았다. 학습 정상 2,580개·spoof22,800개와 전체 dev/target의 UID·분모·cache SHA를 확인했고 학습 sample 합은 별도의 C input audit와 일치했다. rate<0.5는 원래2.01875초 미만을 두 배 넘게 늘리는 경우다. DF spoof 전체 범위는0.099–7.260배로, 약0.4초를10.1배 늘리거나29.312초를7.26배 빠르게 해야 하는 극단도 있다. 길이 상관이 변환 배율·artifact 강도 상관으로 옮겨갈 수 있다.

![두 배 초과 늘림이 필요한 클래스별 비율](figures/speed_duration_feasibility_20261109/extreme_slowdown_class_share.png)

### 본문까지 확인한 선행연구

| 연구 | 본문에서 확인한 방법 | 우리 설계와 다른 점·해석 한계 |
|---|---|---|
| [Wojcicki, IEEE Access2026](https://doi.org/10.1109/ACCESS.2026.3682478), III-C·IV-E | phase-vocoder로1초 길이 통일, CNN/RNN/Transformer ensemble | 고정길이 전처리의 선행. LJSpeech/WaveFake; EBM 속도쌍 손실이나 ASV21 검증 아님. 전체 EER0.22%를 stretch 단독 효과로 인용하지 않음 |
| [CLAD](https://arxiv.org/pdf/2404.15854) | 같은 음성의 manipulation views를 MoCo positive로 연결, time stretch 포함 | 입력은64600샘플 repeat/clip. 출력길이 배율0.9·0.95·1.05·1.1, FFT128. mild 변형이며 모든 원길이→고정길이 정규화 아님. 정상끼리 all-pairs EBM BPL과도 다름 |
| [DeePen v3](https://arxiv.org/html/2502.20427v3), TableI·II | 정상·spoof 모두 librosa rate0.8–1.2 검사 | W2V2 정상 정확도 ASV19 100→3%, MLAAD93→2%; spoof는99→99%,87→100%. 변환에 따른 정상 오인의 직접 선행이며 EER가 아님 |
| [ParallelChain2024](https://www.isca-archive.org/asvspoof_2024/tran24_asvspoof.pdf), 2.1.3–4 | resampling speed0.9–1.1 증강, 긴 음성의 여러4초 crop 점수 평균 | pitch도 변하는 방식이고 여러 증강 묶음이라 speed 단독 효과 아님 |
| [Schäfer et al., EUSIPCO2025](https://eurasip.org/Proceedings/Eusipco/Eusipco2025/pdfs/0000586.pdf), TableII | spoof-only pitch-preserving stretch 공격 | SSL ITW EER0.73→1.78/2.17%(minimum-rate0.8/0.6). 정상도 같은 변형을 받는 우리 전처리와 다름 |

CLAD는 stretch한 spoof가 더 잘 잡히는 현상의 원인 후보로 FFT/iFFT artifact를 제시한다. spoof 탐지율 개선만으로 불변성을 인정하면 안 된다. CLAD의 length loss는 embedding norm이며 음성 길이 정규화 손실이 아니다. 위 연구들로 정확히 같은 제안 전체의 효과나 신규성이 입증되는 것은 아니다. 배율 정의(출력길이 비율 vs speed-rate)를 혼용하지 않았다.

### 재학습 없이 수행한 512개 진단

기존240/B/C 모두50epoch로 고정했다. dev·19LA·21LA·21DF 각각 정상64/spoof64, 사전 길이4구간×각16개를 metadata UID hash/화자 round-robin으로 선택했다. 각 class의 최단·최장 sentinel도 결과를 보기 전에 포함했다. 입력512개×6조건×3모델=9,216개 점수이며 모델·원음·SSL hidden은 밖으로 내보내지 않았다. 기존 컨테이너/CUDA_VISIBLE_DEVICES=0,1에서 smoke와 본실행 모두 exit0, 본실행 GPU rank별256개 완료. pooling noise는 진단용0으로 고정했다.

조건은 (1)앞64600 cap·비반복 (2)원길이 (3)동일 cap 내용을0.8배 감속 (4)동일 cap 내용을1.25배 가속 (5)전체 원음을4.0375초로 정규화 (6)cap 내용을0.8배 감속한 뒤 원래 길이로 되돌리기다. librosa0.11.0, phase-vocoder FFT2048/hop512이며 길이 보정을 위한 추가 crop/pad는 하지 않고 출력 길이를 검증했다. rate1은 정확한 copy다. Roundtrip은 길이를 복원하지만 DSP 손실까지 원복하는 대조군은 아니다.

**다음 EER는 각 집합128개 균형·길이층화 표본의 탐색적 진단치다. 앞의 전체 benchmark EER와 직접 비교하지 않는다.** 단일 class 오류1개의 해상도는1.5625%p이며 이 표로 체크포인트·배율을 선택하지 않았다.

'''+probe_table+'''
![B 모델의 변환 전후 진단 EER](figures/speed_duration_feasibility_20261109/phase_vocoder_B_diagnostic.png)

B의 전체음성→4.0375초 조건에서 원래 full-dev 임계값을 그대로 적용하면 정상 거부가 dev63/64,19LA64/64,21LA62/64,21DF64/64였다. 원래 길이로 되돌린 roundtrip도 거의 같은 실패를 보였다. EER 자체도 악화돼 단순 임계값 이동만으로 설명되지 않는다. 같은 조건의 dev 표본으로 재보정한 오류도 별도 CSV에 남겼지만 표본이 작아 배포 임계값으로 쓰지 않는다. 현재 결과는 이 phase-vocoder·동결 모델 조합의 실패이며 speed augmentation 재학습이나 다른 알고리즘의 실패를 뜻하지 않는다.

### EBM 목적식과 view 선택

현재 BPL은 정상12개 Long×정상12개 Short의144쌍을 모두 당긴다. 같은 발화쌍은 대각12개이고 나머지132개는 다른 정상 발화다. Long에만 분류 손실이 적용되며 spoof에는 BPL이 없다. 입력을 slow/fast로만 바꾸면 두 class의 같은 발화 속도 일관성을 학습한다는 목적과 다르다. 모든 spoof를 all-pairs로 당기는 것도 여러 공격을 한 군집으로 강제하므로 권하지 않는다.

검토할 목적식은 두 view의 OC 분류 손실 평균 + 같은 발화의 속도 일관성 항 + 기존 정상 all-pairs BPL이다. 각 항을 분리하고 같은 발화 항은 정상·spoof 모두에서 평가한다. 기존 BPL 유지/제거, score 일관성/embedding 일관성은 별도 ablation이다. centroid의 두 번째 업데이트에 어떤 view를 쓸지도 대칭으로 명시해야 한다. 강한 embedding 일치는 탐지에 유효한 시간 단서까지 지울 수 있으므로 처음에는 score 일관성과 class margin 보존부터 확인한다.

기본 후보는 원본 anchor를 포함한 완만한 양방향 변형이다. 원본–slow만 쓰면 긴 음성 정규화에 필요한 fast를 다루지 못하고, slow–fast만 쓰면 두 view가 공유하는 DSP 흔적에 과적합할 수 있다. 원본–slow/원본–fast/slow–fast를 같은 perturbation budget·손실 수·원음으로 비교한다. 예로0.8/1.25는 log-rate 대칭이지만 안전성이 보장된 값은 아니며 현재 phase-vocoder 진단에서는 모두 실패했다. extreme rate까지 불변성을 한 번에 강제하지 않는다.

### 다음 실험과 성공·중단 기준

| 단계 | 비교 | 성공으로 볼 결과 | 해석·중단 조건 |
|---|---|---|---|
| 전처리 진단 | 원본/완만한 slow·fast/roundtrip, 두 알고리즘 | 정상 보존과 spoof 분리가 함께 유지, 변환 강도별 오류 급증 없음 | spoof만 잘 잡히고 정상까지 spoof로 밀리면 전처리 재설계 |
| 학습 ablation | B control / mild augmentation만 / same-utterance consistency추가 / duration-normalization만 | consistency의 추가 이득이 clean·robustness 모두에서 재현 | 전체 표현 collapse나 clean EER 악화로 민감도만 줄면 실패 |
| view ablation | original–slow / original–fast / slow–fast / original anchor+양방향 | 같은 비용에서 unseen-rate·unseen-algorithm 일반화가 좋음 | 동일T 재정규화로 두view가 같아지는 설계는 사용하지 않음 |
| 길이 가설 분리 | class별 길이분포 맞춤 / pooling 길이 의존성 변경 / time stretch | native 짧은 정상 오류와 길이조건부 score차이가 줄고 attack별 탐지가 유지 | 길이평균만 같거나 전체 EER만 좋아서는 원인 규명 아님 |
| 최종 검증 | 보지 않은 배율·알고리즘·데이터, 복수seed | clean EER·고정dev임계값의정상거부/spoof수락 모두 허용범위내, 속도 민감도 감소 | 허용 성능 저하량은 평가 전 정해야 하며 수치 근거 없이 임의로 성공 선언하지 않음 |

권장 순서는 전처리 알고리즘 확인 → mild augmentation 단독 → 원본 anchor+same-utterance consistency → 강한 고정길이 정규화의 추가 가치를 분리 검증하는 것이다. 긴 내용을 전부 쓰는 목적에는 여러4초 window의 score 집계가 시간 압축 없이 가능한 경쟁안이다. 짧은 음성에는 원본 길이 입력과 mask/개별처리를 유지하면서 class별 길이 분포를 맞추는 경쟁안이 있다. 고정64600샘플로 압축해도 모든 acoustic 정보가 손실 없이 보존되는 것은 아니며, 현재 XLS-R가 동결돼 강한 변형에서 frontend가 잃은 정보는 backend만으로 복원하기 어렵다.

분석 코드·프로토콜·전체 점수·검산: `artifacts/speed_duration_feasibility_20261109/`. ROC는 독립 동점그룹 누적 방식으로 재계산해 일치했고 UID·분모·6조건 coverage·SHA를 확인했다. 서버 source 전체 hash와 세 checkpoint hash 불변·Git clean을 확인했다. 추가 학습·optimizer 갱신·예약 재활성화는 하지 않았다.
'''
marker='## 속도·길이 정규화 및 속도 일관성 학습의 타당성 점검 — 2026-11-09'
text=REPORT.read_text(encoding='utf-8')
assert marker not in text, 'Inspect existing section rather than duplicate'
with REPORT.open('a',encoding='utf-8') as f:f.write(section)
print(REPORT)
