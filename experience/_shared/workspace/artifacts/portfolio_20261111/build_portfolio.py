"""Create a research-role portfolio from local, already completed experiment evidence."""
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DEST = ROOT / 'docs/portfolio'
FIG = DEST / 'figures'
DEST.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
REPORT = DEST / 'audio_deepfake_research_portfolio_20261111.md'
SOURCES = {}

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def remember(p):
    q = ROOT / p
    SOURCES[p] = {'path': p, 'sha256': digest(q)}
    return q

def j(p):
    return json.loads(remember(p).read_text(encoding='utf-8-sig'))

def c(p):
    with remember(p).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def link(p, label):
    remember(p)
    return f'[{label}](../../{p})'

def save(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def table(headers, rows):
    return '\n'.join(['|'+'|'.join(headers)+'|', '|'+'|'.join(['---']*len(headers))+'|'] + ['|'+'|'.join(map(str,row))+'|' for row in rows])

P_COMP = 'artifacts/ebm_coverage_e20_eval_20261110/comparison.json'
P_BC = 'artifacts/ebm_B_C_checkpoints_20261109/metrics.json'
P_MIN = 'artifacts/ebm_B_C_checkpoints_20261109/completed_B_C_dataset_minima.json'
P_EER = 'artifacts/repetition_shortcut_20261111/reviewed_analysis/independent_full_eer.csv'
P_DIR = 'artifacts/paired_score_direction_20261111/results.json'
P_CROP = 'artifacts/crop_repetition_eer_20261111/requested_comparison.csv'
P_WITHIN = 'artifacts/crop_repetition_eer_20261111/within_model_contrasts.csv'
P_SWAP = 'artifacts/geometry_cache/local_analysis/bonafide_cross_recheck_20261106_v1/class_replacement_eer.csv'
P_SWAP_CI = 'artifacts/geometry_cache/local_analysis/bonafide_cross_recheck_20261106_v1/paired_contrasts.csv'
P_IO = 'artifacts/train_crop_io_activity_20261110/results/io_benchmark.json'
P_COVER = 'artifacts/source_coverage_20261110/state.json'
P_NORM = 'artifacts/geometry_cache/local_analysis/frozen_all_v1/P3_P4/B/21LA_pilot/metrics.csv'
P_NORM_CI = 'artifacts/geometry_cache/local_analysis/frozen_all_v1/P3_P4/B/21LA_pilot/paired_effects.json'
P_INITIAL = 'artifacts/analysis/ebm-performance-diagnosis-20261104.json'
P_SHORT = 'artifacts/short32_train_20261111_v2/stop25_eval_status.json'
comp, bc, minima = j(P_COMP), j(P_BC), j(P_MIN)
full_eer, direction, crop, within = c(P_EER), j(P_DIR), c(P_CROP), c(P_WITHIN)
swap, swap_ci, io, coverage = c(P_SWAP), c(P_SWAP_CI), j(P_IO), j(P_COVER)
norm, norm_ci, initial, short = c(P_NORM), j(P_NORM_CI), j(P_INITIAL), j(P_SHORT)
bc_complete = j('artifacts/ebm_B_C_checkpoints_20261109/complete.json')
bc_state = j('artifacts/ebm_B_C_checkpoints_20261109/state.json')
c_state = j('artifacts/ebm_C_20261108/state.json')
c_meta = j('artifacts/ebm_C_20261108/metadata_validation.json')
train_lengths = c('artifacts/repetition_shortcut_20261111/training_original_lengths.csv')
eval_lengths = c('artifacts/repetition_shortcut_20261111/original_length_distribution.csv')

def get_full(model, policy):
    return float(next(r['eer'] for r in full_eer if r['model']==model and r['phase']=='21LA' and r['condition']==policy))

fixed = {r['phase']:r for r in comp['fixed50_reference']}
relative_df = 100*(1-fixed['21DF']['B_EER_percent']/fixed['21DF']['legacy240_EER_percent'])
fixed_table = table(['공식 평가 자료','기존 모델','반복 제거 모델 B','차이(B−기존)'], [[d, f"{fixed[d]['legacy240_EER_percent']:.4f}%",f"{fixed[d]['B_EER_percent']:.4f}%",f"{fixed[d]['B_EER_percent']-fixed[d]['legacy240_EER_percent']:+.4f}%p"] for d in ['19LA','21LA','21DF']])
minimum_table = table(['자료','B 관측 최소 EER','선택 epoch','C 관측 최소 EER','선택 epoch'], [[d,f"{minima[d]['B']['EER_percent']:.4f}%",minima[d]['B']['epoch'],f"{minima[d]['C']['EER_percent']:.4f}%",minima[d]['C']['epoch']] for d in ['19LA','21LA','21DF']])
paired_table = table(['고정 50epoch 모델','실제 길이 EER','약 4초 반복 EER'], [[name,f"{get_full(m,'native'):.4f}%",f"{get_full(m,'wrap4'):.4f}%"] for m,name in [('legacy240','기존 반복 학습'),('B','반복 제거 학습 B')]])
crop_table = table(['남긴 음성 내용','B·실제 길이 EER','약 4초 대비 변화','기존·약 4초 반복 EER','약 4초 대비 변화'], [[r['content_seconds']+'초',f"{float(r['B_native_EER_percent']):.4f}%",f"{float(r['B_change_from4_pp']):+.4f}%p",f"{float(r['original_repeat4_EER_percent']):.4f}%",f"{float(r['original_change_from4_pp']):+.4f}%p"] for r in crop])

def swap_value(bona, spoof):
    return float(next(r['eer_percent'] for r in swap if r['baseline']=='19LA_eval' and r['bona_dataset']==bona and r['spoof_dataset']==spoof))

swap_matrix = [[swap_value(b,s) for s in ['19LA_eval','DF_full']] for b in ['19LA_eval','DF_full']]
swap_table = table(['정상 음성 집합','19LA 위조 고정','DF 위조 고정'], [[label]+[f'{v:.4f}%' for v in row] for label,row in zip(['19LA 정상','DF 정상'],swap_matrix)])
swap_effect = swap_matrix[1][0]-swap_matrix[0][0]
ci = next(r for r in swap_ci if r['baseline']=='19LA_eval' and r['contrast']=='normal_swap_with_baseline_spoof')

def norm_mean(variant, metric):
    return mean(float(r[metric]) for r in norm if r['variant']==variant)

norm_table = table(['추가 분류기 입력','균형 로그 손실 ↓','EER ↓'], [[name,f"{norm_mean(v,'balanced_logloss'):.6f}",f"{norm_mean(v,'eer_percent'):.4f}%"] for v,name in [('direction_linear','코사인·최종 크기·잔여 방향'),('direction_noise2','같은 입력 + 무작위 숫자 2개'),('direction_hstat','같은 입력 + 정규화 전 평균·크기 통계')]])
norm_effect = next(r for r in norm_ci if r['baseline']=='direction_linear' and r['candidate']=='direction_hstat')
same20_table = table(['자료','B·20epoch','C·20epoch','길이·발화량 균형 학습·20epoch'], [[r['phase']]+[f"{r[k+'_EER_percent']:.4f}%" for k in ['B','C','coverage']] for r in comp['same_epoch_comparison']])

length_rows=[]
t={int(r['label']):r for r in train_lengths if r['threshold_samples']=='32000'}
length_rows.append(['2019 LA 학습',f"{float(t[1]['le_percent']):.2f}%（{t[1]['le']}/{t[1]['n']}）",f"{float(t[0]['le_percent']):.2f}%（{t[0]['le']}/{t[0]['n']}）"])
t={int(r['label']):r for r in eval_lengths if r['phase']=='21LA'}
length_rows.append(['2021 LA 평가',f"{float(t[1]['le2_percent']):.2f}%（{t[1]['le2']}/{t[1]['n']}）",f"{float(t[0]['le2_percent']):.2f}%（{t[0]['le2']}/{t[0]['n']}）"])
length_table=table(['자료','정상 중 2초 이하 비중','위조 중 2초 이하 비중'],length_rows)

# Standalone figures: Korean labels, exact source bindings, zero-based magnitude axes.
font_manager.findfont('Malgun Gothic',fallback_to_default=False)
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,
                     'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.labelcolor':'#334155','text.color':'#172b45','axes.titleweight':'bold',
                     'savefig.facecolor':'white','figure.facecolor':'white'})
COLORS={'legacy240':'#63758d','B':'#087f8c','C':'#bb8244'}
chart_data={}

fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained')
x=np.arange(3); width=.24
for k,offset,name in [('legacy240',-width,'기존 모델'),('B',0,'반복 제거 B'),('C',width,'긴 입력 상한 제거 C')]:
    vals=[fixed[d][k+'_EER_percent'] for d in ['19LA','21LA','21DF']]
    ax.bar(x+offset,vals,width,color=COLORS[k],label=name)
    for xx,v in zip(x+offset,vals):ax.text(xx,v+.12,f'{v:.2f}',ha='center',fontsize=10)
    chart_data['fixed50_'+k]=vals
ax.set(xticks=x,xticklabels=['2019 LA','2021 LA','2021 DF'],ylim=(0,10.7),ylabel='EER (%) · 낮을수록 좋음',title='동일한 비반복 입력에서 비교한 고정 50epoch 모델')
ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
ax.legend(loc='upper left',frameon=False,ncol=3,fontsize=10)
fig.savefig(FIG/'01_common_input_eer.png',dpi=180);plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
for m,name in [('legacy240','기존 반복 학습'),('B','반복 제거 B')]:
    vals=[get_full(m,p) for p in ['native','wrap4']]
    axs[0].plot([0,1],vals,marker='o',linestyle='--' if m=='legacy240' else '-',label=name,color=COLORS[m],linewidth=2.5,markersize=7)
    for xx,v in enumerate(vals):axs[0].annotate(f'{v:.2f}%',(xx,v),xytext=(8,7),textcoords='offset points',fontsize=10)
    chart_data['paired21LA_'+m]=vals
axs[0].set(xticks=[0,1],xticklabels=['실제 길이','약 4초로 반복'],xlim=(-.15,1.35),ylim=(0,9),ylabel='EER (%)',title='같은 전체 21LA · 입력 처리만 변경')
for key,name,m in [('original_repeat4_EER_percent','기존 모델 · 반복','legacy240'),('B_native_EER_percent','B · 반복 없음','B')]:
    vals=[float(r[key]) for r in crop]
    axs[1].plot([0,1,2],vals,marker='o',linestyle='--' if m=='legacy240' else '-',label=name,color=COLORS[m],linewidth=2.5,markersize=7)
    for xx,v in enumerate(vals):axs[1].annotate(f'{v:.2f}%',(xx,v),xytext=(5,7),textcoords='offset points',fontsize=10)
    chart_data['head_crop_'+m]=vals
axs[1].set(xticks=[0,1,2],xticklabels=['4.0375초','2초','1초'],xlim=(-.18,2.5),ylim=(0,18.5),ylabel='EER (%)',title='같은 긴 원음 336개 · 남긴 내용 감소')
for ax in axs:ax.grid(axis='y',alpha=.15);ax.legend(loc='upper left',frameon=False,fontsize=9);ax.set_axisbelow(True)
fig.savefig(FIG/'02_repetition_and_content.png',dpi=180);plt.close(fig)

fig,ax=plt.subplots(figsize=(7.2,4.8),layout='constrained')
matrix=np.array(swap_matrix)
im=ax.imshow(matrix,cmap='YlGnBu',vmin=0,vmax=6.5,aspect='auto')
for row in range(2):
    for col in range(2):ax.text(col,row,f'{matrix[row,col]:.4f}%',ha='center',va='center',fontsize=20,fontweight='bold',color='white' if matrix[row,col]>4.5 else '#172b45')
ax.set(xticks=[0,1],xticklabels=['19LA 위조','DF 위조'],yticks=[0,1],yticklabels=['19LA 정상','DF 정상'],xlabel='위조 점수 집합',ylabel='정상 점수 집합',title='정상·위조 집합을 한쪽씩 교체한 EER')
fig.colorbar(im,ax=ax,label='EER (%) · 낮을수록 좋음',shrink=.85)
fig.savefig(FIG/'03_class_swap.png',dpi=180);plt.close(fig)
chart_data['class_swap_matrix']=swap_matrix
save('chart_data.json',chart_data)

text=f'''# 음성 딥페이크 탐지 모델의 일반화 문제를 분석하고 개선한 EBM 연구

### 음성 길이 전처리·정상 음성 분포·표현 공간을 연결한 통제 실험과 학습 시스템 개발

**같은 모델이 평가 환경에 따라 다른 성능을 내는 이유를 추적하고, 그 원인을 검증할 수 있는 실험을 설계했습니다.** 기존 EBM을 기반으로 가변 길이 학습과 평가 체계를 구현하고, 반복 전처리와 짧은 음성 내용의 영향, 정상 음성의 분포 변화가 탐지 성능에 미치는 영향을 나누어 분석했습니다.

|항목|내용|
|---|---|
|지원 분야|음성·딥러닝 연구|
|프로젝트 형태|공개 연구 모델 EBM을 기반으로 한 음성 위조 탐지 연구·모델 확장|
|기간|2026년 11월 4일 이전의 초기 모델 기록부터 11월 11일까지의 검증 자료 기준|
|담당 역할|아이디어 제안, 연구 질문 정의, 실험 계획, 실험 수행·검증, 결과 해석 주도|
|협업 방식|대학원생 멘토에게 매주 연구 내용을 발표하고, 신규성과 연구 의의에 대한 피드백을 다음 실험에 반영|
|데이터|ASVspoof 2019 LA 학습 25,380개, 공식 평가 3종 총 753,341개|
|주요 기술|Python, PyTorch, Transformers/XLS-R, DDP, NumPy·Pandas, SoundFile, Docker, Git|
|산출물|학습 모델 B/C, 가변 길이·마스크·재계산 학습 코드, 통제 평가기, 원인 분석 결과, 재현·검증 자료|

## 1. 핵심 성과

- **동일한 비반복 평가 조건에서 DF EER을 8.8775%에서 2.7238%로 낮춘 모델을 구현·검증했습니다.** 고정 50epoch 비교이며 상대 감소율은 {relative_df:.2f}%입니다. 과거 모델의 원래 반복 평가보다 모든 자료에서 좋아졌다는 의미는 아닙니다.
- **정상 음성의 분포 변화가 성능 악화와 연결되는 지점을 분리했습니다.** 위조 집합을 고정하고 정상만 19LA에서 DF로 바꾸면 EER이 0.4352%→5.9991%로 높아졌습니다. 위조만 바꿨을 때는 같은 악화가 나타나지 않았습니다.
- **반복과 짧은 내용의 영향을 구분했습니다.** 반복 없이도 짧아지면 EER이 높아졌고, 기존 모델에서는 같은 짧은 내용을 반복하면 오히려 EER이 낮아졌습니다. 이 결과로 반복이 항상 해롭다는 초기 설명을 수정했습니다.
- **실험 정확성을 유지하면서 데이터 읽기 비용을 줄였습니다.** 구간 직접 읽기로 첫 epoch의 조각 길이 비중을 반영한 읽기 속도를 {io['epoch1_weighted_speedup']:.3f}배로 높였고, 읽기 시간을 {(1-1/io['epoch1_weighted_speedup'])*100:.1f}% 줄였습니다. 원시 구간 512쌍과 증강 후 64쌍의 값 일치를 검증했습니다.

## 2. 목표: 좋은 EER 한 번보다 평가 환경이 바뀌어도 설명 가능한 모델

음성 위조 탐지 모델은 정상 발화와 합성·음성 변환으로 만들어진 발화를 구분합니다. 이 프로젝트에서 마주한 문제는 학습한 위조 종류뿐 아니라 발화 길이와 정상 음성의 출처가 달라질 때도 성능이 크게 달라진다는 점이었습니다.

제가 세운 연구 목표는 세 가지였습니다.

1. **길이·반복·출처 같은 조건에 대한 의존성을 확인한다.** 실제 정답과 관련된 신호를 읽는지, 데이터 구성과 전처리에 민감한지 구분합니다.
2. **기존 분류 목적을 보존하면서 모델을 확장한다.** 입력 길이를 바꾸는 실험에서 손실 함수·배치·평가까지 동시에 바뀌지 않도록 통제합니다.
3. **주장과 증거가 연결되는 결과를 남긴다.** 전체 EER, 정상 거부, 위조 통과, 길이별 결과를 함께 보고, 성능 개선과 원인 분석을 구분합니다.

EER은 정상 거부율과 위조 통과율이 같아지는 지점의 오류율이며 낮을수록 좋습니다. 정상 점수가 올라가는 현상, 예측 확률의 품질이 좋아지는 현상, EER이 낮아지는 현상은 별도로 평가했습니다.

## 3. 기반 모델과 제가 확장한 범위

### 3.1 기반 연구

EBM과 Bridge Module의 기본 구조, 사전학습 XLS-R, 기존 정상 쌍 손실과 정상 중심 분류 목적은 선행 연구의 구현을 사용했습니다. 본 프로젝트의 기여는 그 모델의 실패 조건을 분석하고, 새로운 입력 처리·학습·평가 경로와 통제 실험을 구축한 부분입니다. [기반 구현과 원 논문 안내](../../EBM/README.md)

```mermaid
flowchart LR
    A[16 kHz 음성] --> B[길이 처리와 학습 증강]
    B --> C[고정된 XLS-R 음성 특징 추출기]
    C --> D[Bridge Module: 층별 정보 통합]
    D --> E[시간 정보 요약과 160차원 특징]
    E --> F[LayerNorm]
    F --> G[학습된 정상 기준과 코사인 비교]
    G --> H[정상·위조 점수]
```

XLS-R은 고정하고 Bridge 이후 분류 모듈과 정상 기준을 학습합니다. 학습은 두 단계입니다. 먼저 정상·위조 분류와 정상 특징 정렬로 분류 모듈을 갱신하고, 갱신된 모듈의 정상 특징으로 정상 기준 벡터를 다시 갱신합니다. 배치 240에서는 GPU당 정상 12개·위조 108개이며, 정상의 긴 입력과 짧은 입력을 **12×12 전체 교차쌍**으로 정렬합니다. 같은 원음끼리의 12쌍만 사용하는 손실이 아닙니다.

### 3.2 확장한 구성 요소

|영역|구체적인 구현·검증|
|---|---|
|입력 처리|원음 길이·구간 길이·반복 길이를 분리하고, 반복 없는 가변 길이 입력 지원|
|마스크|음성 샘플의 유효 구간을 XLS-R 특징 프레임까지 연결하고, 시간 요약에서 빈 부분 제외|
|메모리 제한 학습|작은 물리 묶음으로 계산하면서 전체 정상 쌍의 손실과 두 단계 갱신 유지|
|평가|입력 상한·마스크·점수 정의를 통일하고 UID·라벨·표본 수·중복을 검증|
|실험 진단|동일 원음 반복 개입, 구간 축소, 정상·위조 집합 교체, 표현 정보 추가 대조|
|학습 운영|저장 후 평가, 체크포인트 재로딩·해시 확인, 단계별 성공·실패 기록|

## 4. 기간별 모델 개발과 판단의 변화

날짜는 Git 기록과 실행·평가 기록을 기준으로 했습니다. 초기 모델의 개별 시작일은 확정하지 않고 11월 4일 이전 기록으로 표시했습니다.

|기간|모델·개발 단계|그때의 질문과 실행|결과와 다음 판단|
|---|---|---|---|
|11/04 이전·초기 기록|배치 80·160·240 기준 모델, GPU 간 정상 특징 공유형|배치와 정상 쌍을 늘리면 일반화가 좋아지는가?|자료별 결과가 엇갈려 규모 증가만으로 설명하지 않고, 입력·표현·평가 조건을 분석하기로 전환|
|11/05~11/06|고정 모델의 임베딩 분석과 정상·위조 교체 진단|코사인 한 값이 놓치는 정보와, 평가 분포 변화의 취약점은 무엇인가?|정상 분포 변화와 길이 처리의 영향을 후속 핵심 질문으로 구체화|
|11/06~11/07|반복 제거 모델 B 구현·50epoch 학습|약 4초·3초 상한을 유지하면서 반복을 없애면 어떻게 달라지는가?|가변 길이 학습 완료. 비반복 평가 적합성은 좋아졌지만 모든 평가 조건에서 우수한 것은 아님|
|11/08~11/09|긴 입력 상한 제거 모델 C·50epoch 학습|B와 비교해 긴 원음 전체를 학습하는 것이 이득인가?|B/C 각각 저장 모델 10개를 공통 입력으로 평가. 관측 최소 EER은 세 자료 모두 B가 낮았음|
|11/10|길이·발화량 균형 정책, 원음 전수 노출, 구간 읽기 최적화|길이 분포만 맞추면 충분한가? 계산·데이터 사용을 개선할 수 있는가?|입력 읽기·노출 정책은 검증했으나 복합 학습의 LA 성능은 개선되지 않아 원인을 다시 분리|
|11/11|전체 반복 진단·짧은 구간 대조·조건부 3초/2초 정렬 후속 모델|짧은 내용과 반복을 구분하고, 기존 학습 변경을 줄일 수 있는가?|반복 진단 완료. 후속 모델은 보존 기록상 25epoch까지 학습 후 중단, 최종 타깃 성과는 본 포트폴리오에서 확정하지 않음|

B는 11월 6일 21:36부터 11월 7일 16:22까지, C는 11월 8일 06:14부터 11월 9일 06:36까지 각각 50epoch 학습을 완료했습니다. B/C의 20개 저장 모델을 대상으로 한 공통 평가도 11월 9일 종료 기록을 확인했습니다. 시작 문서에 남아 있는 과거 진행 중 문구보다 종료 기록과 실제 지표를 우선했습니다.

근거: {link('docs/experiments/ebm_B_final_results_20261108.md','B 학습 완료 기록')}, {link('artifacts/ebm_C_20261108/state.json','C 학습 종료 기록')}, {link('artifacts/ebm_B_C_checkpoints_20261109/state.json','공통 평가 종료 기록')}, {link(P_SHORT,'후속 모델의 보존된 중단·평가 상태')}.

## 5. 사례 1 — 반복 제거 모델을 만들고 평가 조건의 영향을 분리하다

### 문제와 고민

기존 모델은 짧은 원음을 반복해 약 4초 입력으로 만들었습니다. 일부 정상 음성의 오탐이 반복과 연결돼 보여 반복을 제거하려 했습니다. 그러나 학습과 평가에서 반복 여부가 동시에 바뀌면 새 모델이 좋아진 것인지, 입력 조건이 달라진 것인지 구분하기 어려웠습니다.

### 해결 접근

모델 B는 기존 긴 입력 4.0375초와 짧은 입력 3.0275초의 상한을 유지하고 부족한 음성을 반복하지 않도록 구현했습니다. 전체 배치 240과 기존 정상 쌍 손실을 유지하면서, 가변 길이 입력과 마스크를 연결했습니다. 이후 기존 모델과 B를 같은 비반복 평가 정책에서 비교했습니다.

### 결과: 동일한 비반복 평가의 고정 50epoch 비교

{fixed_table}

![동일 비반복 입력의 고정 50epoch EER](figures/01_common_input_eer.png)

DF에서는 EER이 8.8775%→2.7238%로 낮아졌습니다. 다만 이 표는 **동일 비반복 입력에서의 모델 비교**입니다. 기존 모델의 원래 반복 평가 DF EER은 3.4770%였고, 21LA는 1.0528%였습니다. B의 비반복 21LA 4.7921%보다 기존 반복 처리가 더 좋았습니다. 따라서 반복 제거로 모든 환경을 개선했다고 해석하지 않았습니다.

또한 과거 기준 학습과 B는 샘플러·계산 구현 등이 달라 이 차이를 반복 제거 학습 하나의 순수 효과로 쓰지 않았습니다. 이 구분은 다음의 동일 모델 입력 개입 실험으로 이어졌습니다.

근거: {link(P_COMP,'동일 입력·고정 epoch 비교와 과거 반복 평가')}, {link('docs/experiments/ebm_control_B_20261106.md','B 변경 범위와 통제 조건')}.

### 추가 질문: 4초 상한이 문제라면 긴 입력 전체를 쓰는 편이 좋은가?

모델 C에서는 학습용 긴 입력의 상한만 제거했습니다. 짧은 입력 상한과 공통 평가 입력은 B와 같게 유지했고, 50epoch 전체 원음 순서도 대조했습니다. 두 모델 모두 5epoch 간격의 10개 저장 모델에서 관측 최소 EER을 계산했습니다.

{minimum_table}

**관측 최소값은 세 자료 모두 상한을 유지한 B가 낮았습니다.** 단일 초기값의 결과이며 자료별 최적 epoch이 다릅니다. 고정 50epoch에서 19LA는 C가 더 좋았으므로 약 4초가 보편적인 최적 길이라고 주장하지 않았습니다. 이 결과를 바탕으로 상한을 없애는 것보다 반복·짧은 내용·자료 구성의 영향을 먼저 분리하기로 판단했습니다.

근거: {link(P_BC,'B/C 전체 저장 epoch의 실제 지표')}, {link('artifacts/ebm_C_20261108/metadata_validation.json','B/C 입력·원음 순서·평가 통제 검증')}.

## 6. 사례 2 — 짧아서 어려운 것과 반복해서 달라지는 것을 구분하다

### 문제와 가설

짧은 원음을 반복하면 정상 점수가 위조 방향으로 이동하는 사례가 있었습니다. 처음에는 반복이 탐지 성능을 해치는 단서일 가능성에 주목했습니다. 이를 확인하려면 **같은 원음의 반복 여부**, **같은 내용의 반복량**, **남겨 둔 고유 내용의 길이**를 구분해야 했습니다.

### 실험 A: 같은 전체 평가 자료에서 반복 여부만 바꾸기

기존 모델과 B를 각각 50epoch로 고정하고 21LA 148,176개 전체를 반복·비반복 양쪽으로 평가했습니다. 고정 모델에 대한 입력 개입이므로 조건마다 유리한 epoch을 새로 고르지 않았습니다. 추론 추가 잡음은 0이었습니다.

{paired_table}

기존 모델은 반복 후 정상과 위조가 평균적으로 위조 방향으로 이동했고, 특히 위조의 이동이 컸습니다. B에서는 두 클래스가 평균적으로 정상 방향으로 이동했습니다. 비반복 개발 임계값을 고정한 진단에서 기존 모델은 정상 608개가 새로 거부됐지만, 위조 12,862개의 잘못된 정상 판정이 해결됐습니다. B는 위조 9,561개가 새로 정상 판정을 받았습니다. 이 판정 전환 개수는 조건별 임계값을 사용하는 EER의 직접적인 분해값은 아닙니다. **정상 오탐에 대한 관찰과 전체 EER 변화가 같은 결론을 뜻하지 않는다는 점**을 확인했습니다.

### 실험 B: 동일 원음에서 남겨 둔 내용을 줄이기

긴 원음 336개, 정상 128개·위조 208개에서 앞 4.0375초→2초→1초를 사용했습니다. B에는 실제 길이로, 기존 모델에는 같은 짧은 내용을 반복해 약 4초로 넣었습니다. 앞·뒤 점수 평균으로 여러 구간의 정보를 합치지 않고 앞부분 한 구간만 주 결과에 사용했습니다.

{crop_table}

![전체 반복 개입과 짧은 내용 대조](figures/02_repetition_and_content.png)

두 모델 모두 짧아지면 EER이 높아졌습니다. 그러나 서로 다른 모델의 악화량 차이를 반복의 인과 효과로 해석하지 않고, **기존 모델 안에서 같은 1초 내용을 그대로 넣은 29.3269%와 반복한 15.3846%를 비교**했습니다. 반복은 오히려 EER을 13.9423%p 낮췄습니다. 뒷부분 대조에서도 기존 모델은 같은 방향이었습니다.

화자 66개를 묶어 2,000회 재표집한 개별 탐색적 구간에서도 B의 구간 축소 손해와 기존 모델의 반복 이득은 같은 방향이었습니다. 이 결과는 자연적으로 짧은 모든 음성의 대표값이 아니라 긴 원음을 자른 진단 표본의 결과입니다.

### 데이터를 함께 확인한 이유

{length_table}

원음 길이와 정답의 연관은 실제로 존재했습니다. 하지만 학습 자료에서 단순히 반복이 필요한 원음의 비율은 정상 77.67%, 위조 72.83%였습니다. **반복 여부와 반복 정도를 구분하지 않으면 짧은 위조가 많다는 통계를 잘못 해석할 수 있었습니다.**

### 연구 판단

반복은 항상 해롭다는 설명을 버리고, **짧은 내용의 분리력과 학습 방식에 따른 반복 반응을 함께 다루는 문제**로 수정했습니다. 데이터 길이 차이가 반복 의존성을 만들었다는 학습 인과관계는 아직 확정하지 않았습니다. 접합부를 완화해도 이동 대부분이 남아, 경계만 부드럽게 하는 해법도 충분하지 않았습니다.

근거: {link(P_EER,'전체 21LA 반복 개입')}, {link(P_DIR,'클래스별 이동과 실제 판정 전환')}, {link('docs/experiments/ebm_shortening_vs_repetition_eer_20261111.md','같은 내용의 길이·반복 대조와 불확실성')}, {link('docs/experiments/ebm_duration_bias_research_20261111.md','길이 분포와 접합 대조')}.

## 7. 사례 3 — 새로운 위조보다 정상 분포의 변화가 병목인지 확인하다

### 문제와 고민

새로운 평가 자료의 EER이 높으면 새로운 합성 공격을 잘 잡지 못한다고 생각하기 쉽습니다. 저는 **정상 음성의 분포가 달라져 생기는 문제도 따로 확인해야 한다**고 보고, 정상과 위조 집합을 한쪽씩 교체했습니다.

### 실험 설계

여기서는 앞의 배치 240·B/C와 구분되는 **배치 160·50epoch 고정 모델**의 저장 점수를 사용했습니다. 위조를 고정하고 정상만 바꾸거나, 정상을 고정하고 위조만 바꿔 EER을 다시 계산했습니다. DF 전체 정상 14,869개·위조 519,059개, 19LA 정상 7,355개·위조 63,882개를 사용했습니다.

{swap_table}

![정상·위조 집합의 교차 평가](figures/03_class_swap.png)

정상만 DF로 바꾸면 EER은 **0.4352%→5.9991%, +{swap_effect:.4f}%p** 악화됐습니다. 반면 19LA 정상을 유지하고 위조만 DF로 바꾸면 **0.2855%**로 낮아졌습니다. 출처별 화자를 묶은 2,000회 재표집의 보정된 98.75% 구간도 정상 교체 변화에 대해 **[{float(ci['ci98_75_lower']):+.4f}, {float(ci['ci98_75_upper']):+.4f}]%p**였습니다.

21LA를 기준으로 바꿔도 같은 방향이었으며, DF의 추가 코덱 없는 부분이나 클래스별 길이 비중을 맞춘 대조에서도 정상 교체의 악화는 남았습니다. 두 보완 대조는 각각 수행한 것이며 모든 차이를 동시에 제거한 실험은 아닙니다.

### 연구 판단

이 모델과 자료 조합에서는 정상 분포 변화가 중요한 취약점이었습니다. 이후 연구 질문을 새로운 위조 종류 추가에만 두지 않고, **정상 발화의 출처·길이·녹음 조건이 충분히 대표되는지**로 확장했습니다. 다만 정상 학습 자료를 늘리면 위조 학습 자료를 늘리는 것보다 성능이 더 좋아진다는 학습 효과까지 검증한 것은 아닙니다.

근거: {link(P_SWAP,'클래스 교체 실제 EER')}, {link(P_SWAP_CI,'교체 차이와 화자 재표집 구간')}, {link('docs/experiments/ebm_bonafide_diversity_research_20261111.md','길이·코덱·출처 대조를 포함한 연구 기록')}.

## 8. 사례 4 — 코사인 점수 밖의 정보와 실제 전이 개선을 구분하다

### 문제와 접근

최종 코사인은 정상 기준과 이루는 각도 한 값으로 음성을 평가합니다. 같은 각도에 있는 특징들은 다른 방향 구조나 정규화 전 크기가 달라도 같은 점수를 받을 수 있습니다. 저는 저장 임베딩에 추가로 읽을 수 있는 정보가 있는지, 그것이 새로운 자료의 탐지 성능으로 연결되는지를 나누어 조사했습니다.

길이를 1로 정규화한 특징을 x, 같은 방식으로 정규화한 정상 기준을 c, 코사인 점수를 s라 하면 정상 기준에 수직인 잔여 성분은 `r=x−s·c`입니다. 여기서 `‖r‖=√(1−s²)`이므로 잔여 크기를 코사인과 독립적인 새 정보라고 해석하지 않았습니다. 별도의 방향과 정규화 전 통계를 대조했습니다.

### 결과를 읽는 기준

배치 160·50epoch 모델의 저장 표현에서, 21LA 20,000개를 출처·화자 그룹 기준의 바깥 5분할·안쪽 3분할로 나누어 추가 분류기를 평가했습니다. 규제와 학습 횟수는 안쪽 검증에서만 선택했고 표준화도 학습 부분에서만 계산했습니다. 아래는 추가 분류기 초기값 3개의 지표 평균입니다.

{norm_table}

정규화 전 평균·크기 통계 추가의 균형 로그 손실 감소는 **{norm_effect['mean_balanced_LL_reduction']:.6f}**, 개별 95% 구간은 **[{norm_effect['LL_CI95'][0]:.6f}, {norm_effect['LL_CI95'][1]:.6f}]**였습니다. 입력 수가 두 개 늘어난 효과와 구분하기 위해 무작위 숫자 대조도 포함했습니다.

**예측 확률의 품질은 개선됐지만 EER은 0.8%로 같았습니다.** 또한 이 결과는 해당 평가 자료 내부에서 추가 분류기를 학습·검증한 진단이며, 2019 LA에서 학습해 미지 자료로 전이한 개선이 아닙니다. 별도의 잔여 방향 전이 실험에서도 로그 손실의 소폭 개선과 EER 무개선을 분리해 보고했습니다.

### 연구 판단

표현에 추가 정보가 있다는 사실만으로 새 탐지기가 좋아졌다고 선언하지 않았습니다. **정보의 존재, 분류기가 그 정보를 읽는 능력, 외부 자료로의 전이**를 각각 검증해야 한다는 기준을 세웠습니다. 이 때문에 norm·잔여 방향 분석은 후속 설계의 근거로 사용하고, 검증된 전체 EER 개선과는 분리했습니다.

근거: {link(P_NORM,'초기값별 내부 진단 지표')}, {link(P_NORM_CI,'통계 추가의 효과와 구간')}, {link('docs/experiments/ebm_norm_information_research_20261111.md','크기 정보의 검증 범위')}, {link('docs/experiments/ebm_frozen_all_results_20261105.md','잔여 방향의 내부 평가와 전이 평가')}.

## 9. 연구를 지속할 수 있게 만든 구현과 검증

### 9.1 메모리 때문에 학습 목적이 바뀌지 않도록 설계

가변 길이 음성을 전체 논리 배치 그대로 계산하면 메모리 부담이 커집니다. 하지만 묶음을 작게 나누고 각 묶음의 정상 쌍 손실을 따로 평균하면 서로 다른 묶음 사이의 쌍이 사라집니다.

이를 피하기 위해 작은 물리 묶음에서 임베딩을 모으고, 전체 논리 배치의 손실이 각 임베딩에 주는 gradient를 계산한 뒤, 같은 dropout·잡음 난수를 복원해 본체를 다시 계산했습니다. 이 방식으로 GPU당 정상 **12×12 전체 쌍**을 유지했습니다. 본체 갱신 뒤 정상 기준을 갱신할 때는 새 본체로 특징을 다시 계산해 이전 특징이 섞이지 않게 했습니다.

C의 최장 입력 스트레스 검사에서는 GPU당 실제 120개 논리 배치, 물리 묶음 8개, 두 단계 Adam 갱신을 확인했습니다. 단독·혼합 길이 임베딩의 최대 차이는 두 rank에서 약 1.44×10⁻⁶ 미만이었습니다. 이 수치는 특정 검증 배치의 결과이며 전체 학습의 모든 수치 오차 상한은 아닙니다.

근거: {link('EBM/native_steps.py','임베딩 gradient 재계산과 두 단계 갱신')}, {link('artifacts/ebm_C_20261108/verification_rank0.json','GPU 검증 rank 0')}, {link('artifacts/ebm_C_20261108/verification_rank1.json','GPU 검증 rank 1')}.

### 9.2 필요한 음성 구간만 읽어 데이터 준비 비용 개선

전체 파일을 해독한 뒤 자르던 경로를 파일 헤더 확인·구간 이동·필요 구간 해독으로 바꿨습니다. 값이 바뀌지 않았는지 원시 음성 구간 512쌍, 증강 적용 64쌍을 대조했습니다.

|측정 항목|결과|
|---|---|
|원시·증강 구간 값 비교|512쌍·64쌍 정확히 일치|
|warm-cache 반복 측정|3회|
|첫 epoch 길이 비중 가중 평균 읽기 시간|{io['epoch1_weighted_mean_ms']['full']:.4f}ms→{io['epoch1_weighted_mean_ms']['partial']:.4f}ms|
|가중 읽기 속도|{io['epoch1_weighted_speedup']:.3f}배|
|가중 읽기 시간 감소|{(1-1/io['epoch1_weighted_speedup'])*100:.1f}%|

측정 범위는 CPU 파일 열기·헤더 검사·해독·구간 선택·형변환입니다. 증강과 GPU 계산을 포함한 전체 학습 시간이 같은 비율로 줄었다고 주장하지 않았습니다.

근거: {link(P_IO,'읽기 정확성·반복 측정·길이별 시간 원자료')}.

### 9.3 모든 원음의 노출을 보장하는 길이 균형 샘플러

가변 길이 정책에서 원음을 확률적으로 고르면 특정 epoch에 일부 원음이 빠졌습니다. 모든 원음을 최소 한 번 선택한 뒤 남는 선택 횟수의 확률을 보정해, 기대 선택 비율과 학습량을 유지하는 정책을 구현했습니다.

정책 재생 검사에서 50epoch의 원음 선택을 확인했고, 정상 2,580개·위조 22,800개가 매 epoch 포함됐습니다. 첫 epoch 정상의 분류·쌍 정렬 합산 원음 시간 사용률은 **61.19%→71.70%**로 높아졌습니다. 이는 후속 가변 길이 정책 안에서의 비교이며 기존 고정 길이 EBM보다 원음 수를 더 많이 본다는 주장이 아닙니다. 파일을 한 번 썼다는 것과 그 파일의 모든 시간 구간을 썼다는 것도 구분했습니다.

근거: {link(P_COVER,'원음 전수 노출·기대 확률·시간 사용률 검증')}.

### 9.4 평가 실패로 학습 결과가 사라지는 문제를 저장 순서로 해결

후속 실험에서 평가 작업의 모듈 import 오류로 저장 이전에 흐름이 멈춘 문제가 있었습니다. 이를 계기로 체크포인트를 **먼저 저장·검증하고 평가를 나중에 실행**하도록 수명 주기를 분리했습니다.

모델과 학습 상태를 임시 파일에 기록하고 flush·fsync 후 파일을 교체합니다. 두 파일을 다시 읽어 epoch과 tensor를 대조하고 해시를 확인한 뒤 완료 표식을 만듭니다. Adam·scheduler·rank별 Python/NumPy/Torch/CUDA 난수도 보존합니다. 파일 하나의 원자적 쓰기와 두 파일을 확인한 완료 표식을 구분해 불완전한 저장을 평가에 사용하지 않도록 했습니다.

평가 실패는 해당 단계의 실패로 기록하고, 학습 성공으로 덮어쓰지 않습니다. 저장 모듈 import, 부분 배치, 짧은 정상 0개인 배치, rank마다 대상 수가 다른 경우를 검증 범위에 포함했습니다. 이 운영 개선은 탐지 EER 개선과는 별개의 결과입니다.

근거: {link('EBM/short32_lifecycle.py','저장·재로딩·완료 표식·단계별 오류 처리')}, {link('EBM/SHORT32_ORIGINAL240.md','원본 학습 유지와 저장·평가 계약')}, {link('artifacts/short32_train_20261111_v2/container_validation/validation.json','보존된 CPU 검증 결과')}.

## 10. 기대와 달랐던 결과를 다음 설계로 연결한 과정

### 배치·정상 쌍 확대가 모든 평가에서 좋아지지는 않았다

배치 160에서 두 GPU의 정상 특징을 공유하면 21LA 관측 최소는 1.0160%→0.9928%로 낮아졌지만, 19LA는 0.2311%→0.2552%로 높아졌습니다. 초기 배치 80·160 비교는 학습률과 epoch도 달라 배치 크기 하나의 효과로 해석하지 않았습니다. 배치 320은 첫 배치 메모리 부족으로 학습을 완료하지 못해 성능 순위에서 제외했습니다.

이 경험으로 더 큰 배치·더 많은 정상 쌍보다, 어떤 입력·분포 조건에서 달라지는지를 먼저 확인하는 방향으로 연구를 전환했습니다. {link(P_INITIAL,'초기 모델 설정·최소값·비교 한계')}

### 길이·발화량 균형을 맞춘 복합 학습은 LA 성능을 회복하지 못했다

후속 학습은 0.5~4초의 입력을 사용하되 목표 선택의 80%가 2초 이하가 되도록 했고, 화자·공격별 길이, 음성 활동, 원음 노출, 정상 정렬 상대를 함께 바꿨습니다. 20epoch까지 완료한 뒤 중단했고, 21LA·DF는 지정한 20epoch만 평가했습니다.

{same20_table}

이 표는 **동일 20epoch의 보조 비교이며 여러 epoch 중 최소 EER 비교가 아닙니다.** LA는 악화했고 DF는 B/C 사이였습니다. 최대 길이 4초는 유지했으므로 이 악화를 4초 상한 제거 탓으로 돌리지 않았습니다. 여러 변경이 섞여 짧은 조각의 정보 손실이 주원인이라는 설명도 확정하지 않았습니다.

그다음 설계에서는 원래 데이터로더와 학습 함수를 복원하고, 원음 2초 이하 정상에만 3초/2초 정렬을 적용하도록 변경 범위를 줄였습니다. 다만 이 구성도 대상의 기존 정렬 가중치 감소와 새 항 추가가 함께 일어나므로 두 효과를 분리하는 후속 대조가 필요합니다. 보존 기록상 이 모델은 25epoch 후 중단됐으며, 최종 21LA·DF 개선은 본 포트폴리오의 성과로 사용하지 않았습니다.

근거: {link(P_COMP,'20epoch 학습의 전체 평가와 비교')}, {link('EBM/SHORT32_ORIGINAL240.md','변경 범위를 줄인 조건부 정렬 구현')}, {link(P_SHORT,'후속 모델의 저장·중단 상태')}.

## 11. 이 프로젝트에서 보여 준 연구 역량

|역량|프로젝트에서의 행동과 결과|
|---|---|
|연구 질문 구체화|길이가 짧아서 나쁜지, 반복이 문제인지, 정상 분포가 달라서인지 각각 검정 가능한 대조로 전환|
|통제 실험 설계|같은 원음·같은 고정 모델의 입력 개입, 같은 위조 집합의 정상 교체, 입력 수를 맞춘 무작위 대조 사용|
|모델 개발|기존 EBM의 손실과 두 단계 갱신을 이해하고 가변 길이·마스크·메모리 제한 학습 경로 구현|
|통계·검증|UID·라벨·분모·동점 ROC를 검산하고, 화자 그룹 재표집과 내부/외부 평가 구분|
|결과 해석|고정 epoch와 관측 최소, 점수 이동과 오류, 로그 손실과 EER, 표현 정보와 전이 성능을 구분|
|연구 협업|주간 멘토 발표에서 신규성과 연구 의의를 검토하고 후속 실험의 질문과 범위를 수정|

## 12. 다음 연구에서 검증할 것

1. **반복 학습과 평가 입력의 효과 분리:** 같은 코드·원음 순서·증강·손실에서 반복만 바꿔 학습하고, 두 모델을 반복·비반복으로 교차 평가합니다. 과거 기준 모델과 B의 구현 차이를 제거하기 위한 비교입니다.
2. **정상 분포 대표성의 학습 효과:** 정상의 길이·출처 조건을 바꾸는 효과와 위조 구성 변경을 별도 학습 대조로 나눕니다. 현재 정상 교체 진단을 실제 학습 개선으로 연결하는 단계입니다.
3. **제한적인 길이 정렬의 추가 효과:** 기존 손실 그대로, 대상 기존 항의 가중치만 감소, 가중치 감소+새 정렬 항을 구분합니다. 이후 여러 학습 초기값과 새로운 평가 자료에서 방향이 재현되는지 확인합니다.

이 항목은 후속 계획이며 완료 성과에 합산하지 않았습니다. 연구의 다음 성공 기준은 반복 민감도 감소만이 아니라, 동일 조건의 EER 개선과 짧은 정상·위조 오류의 균형입니다.

## 부록 — 결과를 읽는 기준과 증거 자료

### 평가 기준

- 공식 평가 자료는 19LA 71,237개, 21LA 148,176개, 21DF 533,928개입니다. 개발 자료 24,844개는 별도로 구분합니다.
- B/C 최소값은 각각 저장 모델 10개의 **사후 관측 최소**입니다. 자료별 최적 epoch을 합친 하나의 배포 모델 성능으로 표현하지 않았습니다.
- 같은 모델 입력 개입은 50epoch를 고정했습니다. B/C 공통 평가의 작은 pooling 잡음 1e−5와 반복 진단의 잡음 0도 구분했습니다.
- 정상·위조 집합 교체와 표현 진단은 배치 160 모델, 반복·길이 재학습 비교는 배치 240 계열입니다. 서로 다른 모델의 숫자를 이어서 개선량으로 계산하지 않았습니다.
- 평가 자료는 연구 과정에서 이미 관찰한 자료입니다. 새로운 미관찰 평가 자료에서 검증한 일반적 개선으로 확대하지 않았습니다.

### 구현과 실험 자료

|자료|확인할 수 있는 내용|
|---|---|
|{link('EBM/model/b_linear.py','모델 구현')} · {link('EBM/model/loss.py','손실 구현')}|기반 구조, 마스크 연결, 정상 전체 쌍 손실, 코사인 점수|
|{link('EBM/native_steps.py','가변 길이 학습')}|논리 배치 유지, 임베딩 gradient 재계산, 두 단계 갱신|
|{link('EBM/tools/evaluate_B_C_checkpoints.py','공통 평가기')}|특징 재사용으로 20개 head 평가, UID와 점수 병합|
|{link('docs/experiments/ebm_duration_bias_research_20261111.md','길이 편향 연구 기록')}|같은 원음 개입·데이터 분포·모델 B/C 결과|
|{link('docs/experiments/ebm_bonafide_diversity_research_20261111.md','정상 분포 변화 연구')}|클래스 교체·길이·코덱 대조와 해석|
|{link('docs/experiments/ebm_norm_information_research_20261111.md','표현 크기 정보 연구')}|내부 진단·크기 정보·지표별 한계|
|{link('docs/experiments/ebm_completed_evidence_interpretation_20261111.md','전체 실험 원장')}|학습·진단·검증·미실행 설계를 포함한 110개 항목의 방법과 상태|
|[포트폴리오 수치·출처 검증](../../artifacts/portfolio_20261111/validation.json)|원자료 결합, 그림 수치, 역할 표기, 원본 보존 검사|

작성 기준: **2026년 11월 11일**. 신규 모델 학습이나 음성 추론 없이 보존된 실험 자료와 실제 구현을 확인해 작성했습니다. 역할과 협업 방식은 본인의 설명을 반영했습니다.
'''

REPORT.write_text(text,encoding='utf-8')
save('sources.json',list(SOURCES.values()))
save('metrics_snapshot.json',{
    'fixed50':comp['fixed50_reference'],'B_C_minima':minima,'relative_DF_reduction_percent':relative_df,
    'full21LA_paired':[r for r in full_eer if r['phase']=='21LA'],
    'crop_requested':crop,'original_1sec_within_model':next(r for r in within if r['position']=='head' and r['model']=='legacy240' and r['content_seconds']=='1'),
    'class_swap':swap_matrix,'class_swap_normal_effect_pp':swap_effect,'class_swap_normal_CI98_75':[float(ci['ci98_75_lower']),float(ci['ci98_75_upper'])],
    'io_speedup':io['epoch1_weighted_speedup'],'io_time_reduction_percent':(1-1/io['epoch1_weighted_speedup'])*100,
    'norm_21LA':{v:{k:norm_mean(v,k) for k in ['eer_percent','balanced_logloss']} for v in ['direction_linear','direction_noise2','direction_hstat']},
    'norm_effect':norm_effect,'coverage_policy_status':coverage['status'],'short32_archived_status':short,
    'main_training_seed_count':1,'probe_seed_count':3,'B_C_checkpoint_count':20,
    'official_evaluation_samples':sum(bc_complete['phase_counts'][d] for d in ['19LA','21LA','21DF']),
    'role_evidence':'User confirmed research leadership, ideas/planning/execution/validation/interpretation and weekly graduate-mentor presentations in this conversation.'
})
print(json.dumps({'report':str(REPORT),'bytes':REPORT.stat().st_size,'sources':len(SOURCES),'figures':3,'relative_DF_reduction_percent':relative_df}))
