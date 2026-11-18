"""Materialize all Train-only variants once, entirely on the existing server."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import soundfile as sf

ROOT = Path(sys.argv[1])
sys.path.insert(0,str(ROOT/'code'))
from mild_duration import tempo, transform_parameters
from tools.train_coverage_experiment import save
from dataset.ASVspoof2019_la_train import load_train_items_only


def main():
    config = json.loads((ROOT/'experiment.json').read_text())
    for name, expected in config['source_sha256'].items():
        assert hashlib.sha256((ROOT/'code'/name).read_bytes()).hexdigest() == expected, name
    raw = (ROOT/'duration_policy.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == config['duration_policy_sha256']
    policy = json.loads(raw)
    assert not (ROOT/'cache_complete.json').exists()
    cache = ROOT/'audio_cache'; cache.mkdir(exist_ok=False)
    needed = sum(v['output_samples']*4+128 for row in policy['originals'] for v in row['variants'])
    assert needed < 20*1024**3 and shutil.disk_usage(ROOT).free > needed+20*1024**3
    items = load_train_items_only('/data/ASVspoof2019')
    by_uid = {Path(item.path).stem:item for item in items}
    assert len(by_uid) == 25380 and set(by_uid) == {r['uid'] for r in policy['originals']}
    version = subprocess.run(['sox','--version'],capture_output=True,text=True,check=True).stdout.strip()
    toy = (.1*np.sin(2*np.pi*440*np.arange(48000)/16000)).astype(np.float32)
    probes = []
    for rate in [.8,.912345,1.,1.12345,1.2]:
        out = tempo(toy,rate)
        peak = np.argmax(abs(np.fft.rfft(out)))*16000/len(out)
        assert abs(peak-440) < 2
        if rate == 1: assert np.array_equal(out,toy)
        probes.append({'rate':rate,'samples':len(out),'peak_Hz':float(peak)})
    started = time.time(); hashes = {}
    def prepare(row):
        item = by_uid[row['uid']]
        assert int(item.label) == row['label']
        wave,sr = sf.read(item.path,dtype='float32')
        assert sr == 16000 and wave.ndim == 1 and len(wave) == row['samples'] and np.isfinite(wave).all()
        values = {}
        for v in row['variants']:
            kept,rate,expected = transform_parameters(len(wave),v['target_samples'])
            assert kept == v['kept_samples'] and rate == v['rate'] and expected == v['output_samples']
            assert 0 <= v['start'] <= len(wave)-kept and kept >= .8*len(wave)
            out = tempo(wave[v['start']:v['start']+kept],rate)
            assert len(out) == expected and len(out) >= 720
            path = cache/v['file']; assert not path.exists()
            temporary = path.with_suffix('.partial.npy')
            np.save(temporary,out,allow_pickle=False); temporary.replace(path)
            values[v['file']] = hashlib.sha256(out.tobytes()).hexdigest()
        return values
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for number, values in enumerate(pool.map(prepare,policy['originals']),1):
            assert not hashes.keys() & values.keys(); hashes.update(values)
            if number % 100 == 0:
                save(ROOT/'cache_progress.json',{'originals':number,'total':25380,'variants':len(hashes),
                    'seconds':time.time()-started,'updated_unix':time.time()})
    save(ROOT/'cache_index.json',hashes)
    save(ROOT/'cache_complete.json',{'status':'passed','originals':25380,'variants':len(hashes),
        'policy_sha256':config['duration_policy_sha256'],
        'cache_index_sha256':hashlib.sha256((ROOT/'cache_index.json').read_bytes()).hexdigest(),
        'bytes':needed,'seconds':time.time()-started,'SoX':version,'synthetic_pitch_tests':probes,
        'maximum_crop_fraction':.2,'no_manual_output_repeat_pad_trim':True,
        'all_measured_lengths_equal_policy':True,'optimizer_updates':0})
    print(json.dumps({'status':'passed','variants':len(hashes),'seconds':time.time()-started}))


if __name__ == '__main__': main()
