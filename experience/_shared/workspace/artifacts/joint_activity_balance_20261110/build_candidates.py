"""Enumerate raw Train crop candidates; no dev/eval or learning."""
from collections import Counter
import hashlib,json,multiprocessing as mp,os
from pathlib import Path
import sys,time
from types import SimpleNamespace
import numpy as np
import soundfile as sf
import arguments
from positive_bpl import MatchedTrainSet

DS=None


def one(index):
    import webrtcvad
    item=DS.items[index];path=Path(item.path)
    x,sr=sf.read(path);assert sr==16000 and x.ndim==1 and len(x)==DS.lengths[index]
    # Same float32 pre-augmentation waveform seen by the model/previous audit.
    x=x.astype(np.float32).astype(np.float64)
    pcm=np.clip(np.rint(x*32768),-32768,32767).astype('<i2')
    rows=[];counts=[]
    for length in DS.targets:
        length=int(length);tiles=len(x)//length
        starts=Counter([t*length for t in range(tiles)]+[len(x)-(t+1)*length for t in range(tiles)])
        counts.append(len(starts))
        for start,multiplicity in sorted(starts.items()):
            frames=pcm[start:start+length].reshape(-1,320)
            vad=webrtcvad.Vad(2)
            speech=sum(vad.is_speech(frame.tobytes(),16000) for frame in frames)
            rows.append((start,multiplicity,speech))
    stat=path.stat()
    return counts,rows,stat.st_size,stat.st_mtime_ns,hashlib.sha256(path.read_bytes()).hexdigest()


if __name__=='__main__':
    root=Path(sys.argv[1]);os.nice(10);start=time.monotonic()
    protocol=Path('/data/ASVspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt')
    audio_root=Path('/data/ASVspoof2019/LA/ASVspoof2019_LA_train/flac');items=[]
    for line in protocol.read_text().splitlines():
        speaker,uid,_,attack,kind=line.split();label=int(kind=='bonafide')
        items.append(SimpleNamespace(path=str(audio_root/(uid+'.flac')),speaker=speaker,label=label,type='bonafide' if label else attack))
    args,_,_=arguments.get_args();args['duration_policy_path']=str(root/'duration_policy.json')
    DS=MatchedTrainSet(args,items)
    assert len(DS)==25380 and len(DS.profiles)==140
    offsets=[0];starts=[];mult=[];voice=[];sizes=[];mtimes=[];hashes=[]
    with mp.get_context('fork').Pool(4) as pool:
        for index,(counts,rows,size,mtime,sha) in enumerate(pool.imap(one,range(len(DS)),chunksize=8)):
            for count in counts:offsets.append(offsets[-1]+count)
            for a,b,c in rows:starts.append(a);mult.append(b);voice.append(c)
            sizes.append(size);mtimes.append(mtime);hashes.append(sha)
            if (index+1)%500==0:
                (root/'candidate_progress.json').write_text(json.dumps({'originals_done':index+1,'candidates':len(starts),'elapsed_seconds':time.monotonic()-start}))
    dest=root/'raw_candidates.npz'
    np.savez_compressed(dest,context_offsets=np.asarray(offsets,dtype=np.int64),starts=np.asarray(starts,dtype=np.int32),
        multiplicity=np.asarray(mult,dtype=np.uint8),speech_frames=np.asarray(voice,dtype=np.uint16),
        source_sizes=np.asarray(sizes,dtype=np.int64),source_mtime_ns=np.asarray(mtimes,dtype=np.int64),
        source_sha256=np.asarray(hashes),source_uids=np.asarray([Path(v.path).stem for v in items]))
    metadata={'schema_version':1,'scope':'Train only, all front/back tile candidates, before RawBoost',
        'source_fingerprint':DS.source_fingerprint,'duration_policy_sha256':DS.policy_sha256,
        'train_protocol_sha256':hashlib.sha256(protocol.read_bytes()).hexdigest(),
        'originals':len(DS),'groups':len(DS.profiles),'candidates':len(starts),'target_samples':DS.targets.tolist(),
        'vad':{'implementation':'webrtcvad-wheels2.0.14.post1','mode':2,'sample_rate':16000,'frame_samples':320,'reset_each_crop':True},
        'candidate_npz_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
        'dev_used':False,'eval_used':False,'model_training':False,'elapsed_seconds':time.monotonic()-start}
    (root/'candidate_metadata.json').write_text(json.dumps(metadata,indent=2))
    (root/'candidate_progress.json').write_text(json.dumps({'stage':'complete',**metadata}))
