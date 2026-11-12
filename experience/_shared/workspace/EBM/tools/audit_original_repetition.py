"""CPU-only original-audio repetition probe; prints derived measurements only.

Execute on the server with its existing container's read-only /proc data view.
No model imports, audio exports, resampling, padding or source modifications.
"""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '1'
import hashlib
import json
import datetime
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve


def measures(pcm, sr):
    x = pcm.astype(np.float64) / 2147483648
    x -= x.mean()
    n = len(x)
    # Whole-file lag correlation: long copied loops should match at their lag.
    low, high = int(.25 * sr), n - int(.5 * sr)
    corr, lag = None, None
    if high >= low:
        ac = fftconvolve(x, x[::-1], mode='full')[n-1:]
        cs = np.r_[0., np.cumsum(x*x)]
        lags = np.arange(low, high+1)
        e1, e2 = cs[n-lags], cs[n]-cs[lags]
        valid = (e1 > 1e-12) & (e2 > 1e-12)
        cc = np.where(valid, ac[lags]/np.maximum(np.sqrt(e1*e2), 1e-30), -1.)
        j = int(cc.argmax())
        corr, lag = float(cc[j]), float(lags[j]/sr)
    # Exact nonoverlapping >=200ms copies, at arbitrary destination sample.
    # Seeds every100ms; silence/near-silence excluded by energy and variation.
    size, hop = int(.2*sr), int(.1*sr)
    raw = pcm.astype('<i4', copy=False).tobytes()
    exact = None
    for start in range(0, n-size+1, hop):
        a = pcm[start:start+size].astype(float) / 2147483648
        if np.sqrt(np.mean(a*a)) < 10**(-45/20) or a.std() < 10**(-45/20):
            continue
        pos = raw.find(raw[start*4:(start+size)*4], (start+size)*4)
        while pos >= 0 and pos % 4:
            pos = raw.find(raw[start*4:(start+size)*4], pos+1)
        if pos >= 0:
            exact = [start/sr, pos/4/sr]
            break
    return {'duration_s': n/sr, 'max_long_lag_cosine': corr,
            'best_lag_s': lag, 'exact_200ms_copy_at_s': exact}


def main():
    os.nice(15)
    rng = np.random.default_rng(20261108)
    a = rng.integers(-2**25, 2**25, 16000, dtype=np.int32)
    loop = measures(np.tile(a, 3), 16000)
    native = measures(rng.integers(-2**25, 2**25, 48000, dtype=np.int32), 16000)
    silence = measures(np.zeros(48000, dtype=np.int32), 16000)
    assert loop['exact_200ms_copy_at_s'] is not None and loop['max_long_lag_cosine'] > .999999
    assert native['exact_200ms_copy_at_s'] is None and native['max_long_lag_cosine'] < .1
    assert silence['exact_200ms_copy_at_s'] is None and silence['max_long_lag_cosine'] < .1
    base = Path('/proc/352729/root/data')
    specs = [
        ('19LA_train', base/'ASVspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt', base/'ASVspoof2019/LA/ASVspoof2019_LA_train/flac', '.flac', 4),
        ('19LA_eval', base/'ASVspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt', base/'ASVspoof2019/LA/ASVspoof2019_LA_eval/flac', '.flac', 4),
        ('21LA_eval', base/'ASVspoof2021_LA_eval/keys/LA/CM/trial_metadata.txt', base/'ASVspoof2021_LA_eval/flac', '.flac', 4),
        ('21DF_eval', base/'ASVspoof2021_DF/keys/DF/CM/trial_metadata.txt', base/'ASVspoof2021_DF/ASVspoof2021_DF_eval/flac', '.wav', 5),
    ]
    rows, summaries, errors, selection = [], [], [], []
    for corpus, protocol, folder, ext, label_col in specs:
        lines = [s.split() for s in protocol.read_text().splitlines()]
        if corpus.startswith('21'):
            lines = [s for s in lines if s[7] == 'eval']
        for label in ['bonafide', 'spoof']:
            group = [s for s in lines if ('bonafide' if s[label_col] == 'bonafide' else 'spoof') == label]
            group.sort(key=lambda s: hashlib.sha256(('original-repeat-v1|'+corpus+'|'+s[1]).encode()).digest())
            selected = group[:64]
            assert len(selected) == 64, (corpus, label, len(group))
            selection.extend([{'corpus':corpus,'class':label,'uid':s[1]} for s in selected])
            part = []
            for s in selected:
                try:
                    pcm, sr = sf.read(folder/(s[1]+ext), dtype='int32')
                    assert sr == 16000 and pcm.ndim == 1
                    row = dict(corpus=corpus, **{'class':label}, uid=s[1], **measures(pcm,sr))
                    rows.append(row); part.append(row)
                except Exception as e:
                    errors.append({'corpus':corpus,'uid':s[1],'error':str(e)})
            summaries.append({'corpus':corpus,'class':label,'population':len(group),'selected':len(selected),'measured':len(part),
                              'exact_200ms_copies':sum(q['exact_200ms_copy_at_s'] is not None for q in part),
                              'long_lag_cosine_ge_095':sum(q['max_long_lag_cosine'] is not None and q['max_long_lag_cosine'] >= .95 for q in part),
                              'long_lag_cosine_ge_099':sum(q['max_long_lag_cosine'] is not None and q['max_long_lag_cosine'] >= .99 for q in part)})
            print(json.dumps({'progress':summaries[-1]}), flush=True)
    print(json.dumps({'complete':True,'checked_at':datetime.datetime.now().astimezone().isoformat(),'synthetic_checks':'pass','selection':'64 per corpus/class by salted UID SHA256; no score-based selection',
                      'scope':'full original files; no loader padding; lag>=250ms overlap>=500ms; exact200ms seed stride100ms; energy and std>=-45dBFS',
                      'limitations':['Sample only, not a population census','Exact test can miss copies shorter than200ms, unseeded offsets, gain changes or codecs','Global lag cosine can reflect sustained sounds; not proof of a copied passage','Pitch-scale repetition below250ms and linguistic repetition are not tested'],
                      'summary':summaries,'errors':errors,'selection':selection,'rows':rows}), flush=True)


if __name__ == '__main__':
    main()
