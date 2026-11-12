"""Server CPU-only acoustic descriptors; no model inference or waveform export.

Uses only numpy/scipy/soundfile to avoid changing the server environment.
No silence-duration/proportion features are measured or used as predictors.
"""
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[name]='1'
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import soundfile as sf
from scipy.fft import dct


def save(path,obj):
    tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(path)


def frames(x,n,hop):
    if len(x)<n:x=np.pad(x,(0,n-len(x)),mode='reflect')
    return np.lib.stride_tricks.sliding_window_view(x,n)[::hop].copy()


def descriptors(audio,sr=16000):
    assert sr==16000 and audio.ndim==1 and np.isfinite(audio).all() and len(audio)>0
    # Read precisely the original first crop, without repeat-padding. This
    # describes the available speech rather than periodicity added by padding.
    x=np.asarray(audio[:64600],dtype=float);eps=1e-20
    a=frames(x,400,160);rms=np.sqrt(np.mean(a*a,axis=1))
    # Relative-energy gate for stable acoustic measurement only. Its count or
    # proportion is deliberately not exported as an explanatory feature.
    active=rms>=max(rms.max()*10**(-25/20),1e-10)
    if not active.any():raise ValueError('No measurable signal')
    aa=a[active];ar=rms[active]
    out={'rms_active_db':float(20*np.log10(np.sqrt(np.mean(ar**2))+1e-12)),
         'rms_frame_range_db':float(20*np.log10((np.quantile(ar,.9)+1e-12)/(np.quantile(ar,.1)+1e-12))),
         'crest_active_db':float(20*np.log10((np.quantile(np.abs(aa),.999)+1e-12)/(np.sqrt(np.mean(aa**2))+1e-12))),
         'clipping_fraction':float(np.mean(np.abs(x)>=.9999)),
         'dc_to_rms':float(abs(np.mean(x))/(np.sqrt(np.mean(x*x))+1e-12)),
         'zcr_active':float(np.mean(np.diff(np.signbit(aa),axis=1)))}
    centered=aa-aa.mean(axis=1,keepdims=True)
    power=np.abs(np.fft.rfft(centered*np.hanning(400),n=512,axis=1))**2+eps
    freq=np.fft.rfftfreq(512,1/sr);valid=freq>=80;norm=power[:,valid]/power[:,valid].sum(axis=1,keepdims=True)
    fv=freq[valid];mean=norm.mean(axis=0)
    out['spectral_centroid_hz']=float(np.median(norm@fv))
    for q in [.85,.95]:
        ind=np.argmax(np.cumsum(norm,axis=1)>=q,axis=1)
        out['spectral_rolloff'+str(int(q*100))+'_hz']=float(np.median(fv[ind]))
    out['spectral_flatness_db']=float(10*np.log10(np.median(np.exp(np.mean(np.log(power[:,valid]),axis=1))/np.mean(power[:,valid],axis=1))))
    for lo,hi in [(80,500),(500,1000),(1000,2000),(2000,4000),(4000,6000),(6000,8001)]:
        ratio=power[:,(freq>=lo)&(freq<hi)].sum(axis=1)/power[:,valid].sum(axis=1)
        out['band_%d_%d_db'%(lo,hi)]=float(10*np.log10(np.median(ratio)+eps))
    sel=(fv>=300)&(fv<=7000);lf=np.log2(fv[sel]/1000);db=10*np.log10(mean[sel]);lf-=lf.mean()
    out['spectral_slope_db_per_octave']=float(np.dot(lf,db-db.mean())/np.dot(lf,lf))
    # Fixed log-mel spectrum and cepstral coefficients, all built explicitly.
    mel=lambda h:2595*np.log10(1+h/700)
    hz=lambda m:700*(10**(m/2595)-1)
    edges=hz(np.linspace(mel(80),mel(8000),26));filters=[]
    for i in range(24):filters.append(np.maximum(0,np.minimum((freq-edges[i])/(edges[i+1]-edges[i]),(edges[i+2]-freq)/(edges[i+2]-edges[i+1]))))
    melp=power@np.asarray(filters).T
    logmel=10*np.log10(melp+eps);cep=dct(logmel,type=2,axis=1,norm='ortho')[:,1:14]
    for j in range(13):
        out['mfcc_%02d_mean'%(j+1)]=float(np.mean(cep[:,j]))
        out['mfcc_%02d_sd'%(j+1)]=float(np.std(cep[:,j]))
    # Time variation uses consecutive raw frames; no pause counts or ratios.
    allp=np.abs(np.fft.rfft((a-a.mean(1,keepdims=True))*np.hanning(400),n=512,axis=1))**2+eps
    allnorm=allp[:,valid]/allp[:,valid].sum(1,keepdims=True)
    consecutive=active[1:]&active[:-1]
    flux=np.sum((np.sqrt(allnorm[1:])-np.sqrt(allnorm[:-1]))**2,axis=1)
    out['spectral_flux_active']=float(np.median(flux[consecutive])) if consecutive.any() else 0.
    # ACF-derived cumulative-normalized difference pitch. Select the first
    # local trough below .15, avoiding harmonic/subharmonic global maxima.
    # This is a diagnostic estimate, not a phonetic ground truth label.
    p=frames(x,1024,320);pr=np.sqrt(np.mean(p*p,axis=1));p=p[pr>=max(pr.max()*10**(-25/20),1e-10)]
    p-=p.mean(1,keepdims=True)
    fft=np.fft.rfft(p,n=2048,axis=1);acf=np.fft.irfft(np.abs(fft)**2,n=2048,axis=1)[:,:1024]
    all_lags=np.arange(1,268);cs=np.c_[np.zeros(len(p)),np.cumsum(p*p,axis=1)]
    diff=np.maximum(0,(cs[:,1024-all_lags]+cs[:,[-1]]-cs[:,all_lags]-2*acf[:,all_lags])/(1024-all_lags))
    nd=diff*all_lags/np.maximum(np.cumsum(diff,axis=1),eps)
    lags=np.arange(32,267);cand=nd[:,lags-1]
    local=(cand<=nd[:,lags-2])&(cand<=nd[:,lags])&(cand<.15)
    ix=np.where(local.any(1),local.argmax(1),cand.argmin(1))
    strength=np.clip(1-cand[np.arange(len(p)),ix],0,1);pitch=sr/lags[ix]
    use=strength>=.75
    out['periodicity_score']=float(np.median(strength))
    if use.sum()>=3:
        vals=pitch[use]
        out['pitch_estimate_hz']=float(np.median(vals));out['pitch_range_semitones']=float(12*np.log2(np.quantile(vals,.9)/np.quantile(vals,.1)))
        out['pitch_available']=1
    else:
        out['pitch_estimate_hz']=None;out['pitch_range_semitones']=None;out['pitch_available']=0
    return out


def path_for(row,base):
    uid=row['utterance_id']
    if row['audio_dataset']=='DF':
        return base/'ASVspoof2021_DF/ASVspoof2021_DF_eval/flac'/(uid+'.wav')
    return base/'ASVspoof2019/LA/ASVspoof2019_LA_train/flac'/(uid+'.flac')


def run(root):
    try:os.nice(10)
    except OSError:pass
    source=root/'selected_ids.csv';lock=json.loads((root/'protocol.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest()==lock['selected_ids_sha256']
    assert os.readlink('/proc/352729/ns/mnt')==lock['container_mount_namespace']
    base=Path('/proc/352729/root/data');rows=list(csv.DictReader(source.open()))
    assert not (root/'features.csv').exists() and not (root/'complete.json').exists()
    # Consequential feature checks on signals with a known spectrum/period.
    t=np.arange(16000*2)/16000
    tests={}
    for f0 in [100,200,300]:
        f=descriptors(.1*np.sin(2*np.pi*f0*t));tests[str(f0)]=f['pitch_estimate_hz']
        assert abs(f['pitch_estimate_hz']/f0-1)<.04
    level1=descriptors(.1*np.sin(2*np.pi*200*t));level2=descriptors(.2*np.sin(2*np.pi*200*t))
    assert abs((level2['rms_active_db']-level1['rms_active_db'])-20*np.log10(2))<1e-8
    assert abs(level2['spectral_centroid_hz']-level1['spectral_centroid_hz'])<1e-7
    save(root/'synthetic_QC.json',dict(status='pass',pitch_estimates=tests,level_and_spectral_scale_invariance=True))
    start=time.monotonic();fail=[];results=[]
    with (root/'features.partial.jsonl').open('x') as log:
        for i,row in enumerate(rows):
            if i%100==0:
                assert os.readlink('/proc/352729/ns/mnt')==lock['container_mount_namespace']
            p=path_for(row,base)
            try:
                audio,sr=sf.read(str(p),dtype='float32');info=sf.info(str(p))
                f=descriptors(audio,sr)
                result=dict(case_id=row['case_id'],utterance_id=row['utterance_id'],sample_rate=sr,
                    channels=info.channels,source_frames=info.frames,subtype=info.subtype,**f)
                results.append(result);log.write(json.dumps(result,allow_nan=False)+'\n')
            except Exception as e:
                fail.append(dict(case_id=row['case_id'],error=type(e).__name__+': '+str(e)[:120]))
            if (i+1)%50==0 or i+1==len(rows):
                log.flush();save(root/'progress.json',dict(completed=i+1,total=len(rows),success=len(results),failures=len(fail),elapsed_seconds=time.monotonic()-start))
    if fail:save(root/'failures.json',fail);raise RuntimeError('Acoustic feature failures; partial output retained')
    fields=list(results[0]);assert all(list(r)==fields for r in results)
    with (root/'features.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(results)
    save(root/'complete.json',dict(status='pass',rows=len(results),seconds=time.monotonic()-start,
        source_ids_sha256=lock['selected_ids_sha256'],features_sha256=hashlib.sha256((root/'features.csv').read_bytes()).hexdigest(),
        original_audio_exported=False,GPU_used=False,silence_ratio_features=False))
    print('completed',len(results),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
