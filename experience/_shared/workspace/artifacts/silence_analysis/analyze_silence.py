from pathlib import Path
import sys, json, hashlib, wave, struct
import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / '.analysis_deps'))
import webrtcvad

plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False,
                     'font.size': 12, 'figure.facecolor': '#f5f7fb',
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.facecolor': '#f5f7fb'})
SPEECH = '#2761a8'
QUIET = '#f5ce84'

def segments(mask, sr, step, n):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return [(a*step/sr, min(b*step,n)/sr) for a,b in
            zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))]

def analyze(path, mode=2):
    sr, pcm = wavfile.read(path)
    assert pcm.dtype == np.int16 and pcm.ndim == 1 and sr == 16000
    step = sr//100
    padded = np.pad(pcm, (0,(-len(pcm)) % step))
    frames = padded.reshape(-1,step)
    vad = webrtcvad.Vad(mode)
    speech = np.array([vad.is_speech(f.astype('<i2').tobytes(),sr) for f in frames])
    weights = np.minimum(step,np.maximum(0,len(pcm)-np.arange(len(frames))*step))
    intervals = segments(~speech,sr,step,len(pcm))
    duration = len(pcm)/sr
    leading = intervals[0][1] if intervals and intervals[0][0] == 0 else 0
    trailing = duration-intervals[-1][0] if intervals and intervals[-1][1] == duration else 0
    quiet_samples = np.repeat(~speech,step)[:len(pcm)]
    x = pcm.astype(float)/32768
    quiet = x[quiet_samples]
    rms = np.sqrt(np.mean((frames.astype(float)/32768)**2,axis=1))
    chunks=[]
    data=path.read_bytes()
    pos=12
    while pos+8<=len(data):
        tag=data[pos:pos+4].decode('ascii',errors='replace')
        length=struct.unpack('<I',data[pos+4:pos+8])[0]
        chunks.append({'id':tag,'bytes':length})
        pos+=8+length+(length%2)
    stats={'file':path.name,'sample_rate':sr,'channels':1,'sample_format':'signed PCM16',
           'samples':len(pcm),'duration_s':duration,'vad':'WebRTC VAD, mode 2, 10 ms frames',
           'nonspeech_s':float(weights[~speech].sum()/sr),
           'nonspeech_pct':float(weights[~speech].sum()/len(pcm)*100),
           'leading_nonspeech_s':leading,'trailing_nonspeech_s':trailing,
           'exact_zero_pct_all':float(np.mean(pcm==0)*100),
           'exact_zero_pct_nonspeech':float(np.mean(quiet==0)*100) if len(quiet) else None,
           'nonspeech_rms_dbfs':float(20*np.log10(max(np.sqrt(np.mean(quiet**2)),1e-12))) if len(quiet) else None,
           'nonspeech_intervals_s':intervals,'riff_chunks':chunks,
           'sha256':hashlib.sha256(data).hexdigest()}
    return dict(stats=stats,sr=sr,pcm=pcm,x=x,speech=speech,rms=rms,intervals=intervals)

def shade(ax, intervals):
    for a,b in intervals: ax.axvspan(a,b,color=QUIET,alpha=.65,lw=0,zorder=0)

files=['bonafide_p226_247.wav','tts_A16_p226_c0022.wav',
       'bonafide_source_p281_394.wav','vc_A19_p226_p281_394.wav']
items=[analyze(ROOT/f) for f in files]
for d in items:
    d['stats']['vad_mode_sensitivity_pct']={str(m):analyze(ROOT/d['stats']['file'],m)['stats']['nonspeech_pct'] for m in [0,1,2,3]}
stats=[d['stats'] for d in items]
(ROOT/'measurements.json').write_text(json.dumps(stats,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(stats,indent=2,ensure_ascii=False))

d=items[0]; s=d['stats']; x=d['x']; sr=d['sr']; t=np.arange(len(x))/sr
fig=plt.figure(figsize=(14,12),layout='constrained')
gs=fig.add_gridspec(3,2,height_ratios=[1,1.1,1])
ax=fig.add_subplot(gs[0,:])
shade(ax,d['intervals']); ax.plot(t,x,color=SPEECH,lw=.5)
ax.set(xlim=(0,s['duration_s']),ylabel='진폭 (정규화 PCM)',xlabel='시간 (초)',
       title=f"① 전체 파형 — 노란색은 VAD가 비발화로 분류한 구간 ({s['nonspeech_pct']:.1f}%)")
ax.legend(handles=[Patch(facecolor=QUIET,label='비발화 추정 구간'),Patch(facecolor=SPEECH,label='원본 파형')],loc='upper right',fontsize=10)
ax=fig.add_subplot(gs[1,:])
freq,tt,power=spectrogram(x,sr,nperseg=400,noverlap=240,nfft=512,scaling='spectrum')
db=10*np.log10(np.maximum(power,1e-14))
im=ax.pcolormesh(tt,freq/1000,db,shading='auto',cmap='magma',vmin=-110,vmax=-20)
ax.set(xlim=(0,s['duration_s']),ylim=(0,8),ylabel='주파수 (kHz)',xlabel='시간 (초)',
       title='② 스펙트로그램 — 어두운 구간도 반드시 디지털 0은 아닙니다')
fig.colorbar(im,ax=ax,label='스펙트럼 전력 (dB, 고정 기준)',pad=.01)

longest=max(d['intervals'],key=lambda z:z[1]-z[0])
za=longest[0]+min(.04,(longest[1]-longest[0])/4)
zb=min(za+.05,longest[1])
i,j=int(za*sr),int(zb*sr)
z=d['pcm'][i:j]
ax=fig.add_subplot(gs[2,0])
ax.set_facecolor('#fff6e5')
ax.plot(np.arange(i,j)/sr,z,color='#925b12',lw=.8)
ax.axhline(0,color='#748092',lw=.7,ls='--')
ax.set(xlabel='원본 파일의 시간 (초)',ylabel='PCM 정수값 (16 bit)',
       title=f'③ 비발화 구간 {za:.3f}–{zb:.3f}초를 확대')
ax.ticklabel_format(axis='x',style='plain',useOffset=False)
ax=fig.add_subplot(gs[2,1]); ax.axis('off')
text=(f"④ 파일을 열면 무엇이 들어 있나?\n\n"
      f"WAV / mono / 16,000 Hz / signed PCM 16 bit\n"
      f"총 {s['samples']:,}개 샘플 = {s['duration_s']:.3f}초\n\n"
      f"앞 비발화: {s['leading_nonspeech_s']:.2f}초\n"
      f"뒤 비발화: {s['trailing_nonspeech_s']:.3f}초\n"
      f"전체 비발화: {s['nonspeech_s']:.3f}초 ({s['nonspeech_pct']:.1f}%)\n\n"
      f"확대 구간에서 0이 아닌 샘플: {np.mean(z!=0)*100:.1f}%\n"
      f"확대 구간 PCM 범위: {int(z.min())} ~ {int(z.max())}\n"
      f"실제 샘플 12개: {z[:12].tolist()}")
ax.text(0,1,text,va='top',fontsize=12,linespacing=1.5)
fig.suptitle('실제 bona fide 파일 안의 무음\np226_247.wav · ASVspoof 2019 LA 공개 청취 예제',fontsize=20,fontweight='bold')
fig.savefig(ROOT/'bonafide_silence_detail.png',dpi=155)
plt.close(fig)

fig,axes=plt.subplots(4,1,figsize=(14,10),layout='constrained',sharex=True)
labels=['Bona fide · p226_247.wav','TTS A16 · p226_c0022.wav',
        'Bona fide 원본 · p281_394.wav','VC A19 · p226_p281_394.wav']
for ax,d,label in zip(axes,items,labels):
    s=d['stats']; shade(ax,d['intervals'])
    ax.plot(np.arange(len(d['x']))/d['sr'],d['x'],color=SPEECH,lw=.45)
    ax.set(ylabel='진폭',ylim=(-1,1),xlim=(0,5.05),title=f"{label}  |  길이 {s['duration_s']:.3f}초 · 비발화 추정 {s['nonspeech_pct']:.1f}%")
    ax.axvline(s['duration_s'],color='#697586',ls='--',lw=.7)
    if s['duration_s']<4.7:
        ax.text(s['duration_s']+.08,0,'파일 끝\n(이후는 무음이 아님)',fontsize=10,color='#697586',va='center')
axes[-1].set_xlabel('시간 (초) — 모든 행에 같은 시간축 적용')
fig.suptitle('무음은 bona fide에만 존재하지 않습니다\n공개 예제 4개 비교 · 노란색 = WebRTC VAD 비발화 추정',fontsize=20,fontweight='bold')
fig.supxlabel('예제 비교이며 데이터셋 전체 통계가 아닙니다. 위 두 행은 대응 발화 여부를 검증하지 않았습니다.\n아래 두 행은 공개 페이지의 source / VC 예제입니다. VAD mode 2 · 10 ms · 원본 샘플 진폭 유지',fontsize=10,color='#526079')
fig.savefig(ROOT/'silence_comparison.png',dpi=155)
plt.close(fig)

with (ROOT/'vad_intervals.csv').open('w',encoding='utf-8') as f:
    f.write('file,start_s,end_s,duration_s\n')
    for d in items:
        for a,b in d['intervals']:
            f.write(f"{d['stats']['file']},{a:.6f},{b:.6f},{b-a:.6f}\n")

(ROOT/'zoom_measurements.json').write_text(json.dumps({'file':files[0], 'start_s':za,'end_s':zb,
    'sample_count':len(z),'nonzero_pct':float(np.mean(z!=0)*100),'min':int(z.min()),'max':int(z.max()),
    'first_12_samples':z[:12].tolist()},indent=2),encoding='utf-8')
