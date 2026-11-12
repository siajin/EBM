"""Server-only frozen EBM expansion. Reuse the QC-verified single-shape worker.

No original sources, checkpoints or audio are changed/copied. Each dataset is a
new fail-if-existing output. The old DF pilot is excluded with all seed fields
unchanged, so its exact h/z can later be joined to the remaining official eval.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

LA_DIR=Path('/home/mentee/analysis/embedding_geometry/20261105_21la_full_03')
CORE_DIR=Path('/home/mentee/analysis/embedding_geometry/20261105_df_fast_01')
sys.path[:0]=[str(LA_DIR),str(CORE_DIR)]
import la21_extract_v1 as la
core=la.core
REFERENCE=Path('/home/mentee/results/CM/LA/EBM_e50_bs160_lr1e5')
TV=Path('/home/mentee/analysis/embedding_geometry/20261105_fast_p0_01')

def prepare(root, dataset):
    import pandas as pd
    archived,args=core.settings(REFERENCE)
    if dataset=='Q':
        allrows=pd.read_parquet(TV/'manifest.parquet')
        frame=allrows[allrows.split=='Q'].copy()
        protocols={'TV_manifest':core.digest(TV/'manifest.parquet')}
        assert len(frame)==9861
    elif dataset=='19LA_eval':
        base=Path(args['path_19LA'])/'LA'
        protocol=base/'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt'
        records=[]
        for line in protocol.read_text().splitlines():
            speaker,uid,_,attack,label=line.split()
            assert label in {'bonafide','spoof'}
            records.append(dict(dataset='ASVspoof2019_LA',split='R19LA',utterance_id=uid,family_id=uid,
                family_resolution='utterance_only_no_source_derivation_map',speaker_group=speaker,dependency_group=speaker,
                y=int(label=='bonafide'),attack_id='bonafide' if label=='bonafide' else attack,
                path=str(base/'ASVspoof2019_LA_eval/flac'/(uid+'.flac')),condition_id='clean',crop_id='native_first'))
        frame=pd.DataFrame(records);protocols={'19LA_eval':core.digest(protocol)}
    else:
        assert dataset=='DF_remaining'
        base=Path(args['path_21DF']);protocol=base/'keys/DF/CM/trial_metadata.txt'
        pilot=pd.read_parquet(CORE_DIR/'manifest.parquet');existing=set(pilot.utterance_id)
        records=[]
        for line in protocol.read_text().splitlines():
            a=line.split();assert len(a)==13
            if a[7]!='eval' or a[1] in existing:continue
            assert a[5] in {'bonafide','spoof'}
            sp=a[0] if a[0] not in {'-','unknown',''} else 'unknown_shared'
            records.append(dict(dataset='ASVspoof2021_DF',split='R21DF_remaining',utterance_id=a[1],family_id=a[1],
                family_resolution='source_family_unresolved_speaker_group_only',speaker_group=a[3]+'::'+sp,
                dependency_group=a[3]+'::'+sp,speaker_id_known=sp!='unknown_shared',y=int(a[5]=='bonafide'),
                attack_id='bonafide' if a[5]=='bonafide' else a[4],codec=a[2],source_corpus=a[3],trim=a[6],vocoder=a[8],
                path=str(base/'ASVspoof2021_DF_eval/flac'/(a[1]+'.wav')),condition_id='native_DF_'+a[2],crop_id='native_first'))
        frame=pd.DataFrame(records);protocols={'DF':core.digest(protocol),'reused_pilot_manifest':core.digest(CORE_DIR/'manifest.parquet')}
        assert len(frame)==513928 and not set(frame.utterance_id)&existing
    frame=frame.sort_values('utterance_id',kind='stable').reset_index(drop=True)
    assert not frame.utterance_id.duplicated().any() and set(frame.y)=={0,1}
    assert all(Path(p).is_file() for p in frame.path)
    frame.to_parquet(root/'manifest.parquet',index=False)
    audit=dict(status='pass',dataset=dataset,checkpoint_sha256=core.digest(REFERENCE/'model/BestCLS_50_21DF.pt'),
        manifest_sha256=core.digest(root/'manifest.parquet'),script_sha256=core.digest(__file__),
        reused_worker_sha256=core.digest(la.__file__),core_extractor_sha256=core.digest(core.__file__),
        archive_code_sha256={str(p.relative_to(archived)):core.digest(p) for p in archived.rglob('*.py')},
        protocol_sha256=protocols,extraction_splits=sorted(frame.split.unique()),sample_count=len(frame),
        class_counts=frame.y.value_counts().to_dict(),feature_dimension=160,crop_samples=64600,
        SSL_frontend_batch_size=1,classifier_microbatch_size=1,frontend_execution_mode='single_utterance_CUDA_graph',
        Q_features_not_extracted=dataset!='Q',per_utterance_seed_base=20261105,new_EBM_training=False,
        original_sources_modified=False,preprocessing_seed_fields_unchanged=True)
    assert audit['checkpoint_sha256']==la.EXPECTED_HASH
    core.save_json(root/'input_audit.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if k!='archive_code_sha256'}),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','qc','extract']);p.add_argument('--root',required=True)
    p.add_argument('--dataset',required=True,choices=['Q','19LA_eval','DF_remaining']);o=p.parse_args()
    assert Path('/.dockerenv').is_file() and os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1'
    root=Path(o.root)
    if o.stage=='prepare':
        assert not (root/'manifest.parquet').exists();prepare(root,o.dataset);return
    import torch
    import torch.multiprocessing as mp
    assert torch.cuda.device_count()==2
    audit=json.loads((root/'input_audit.json').read_text())
    assert audit['script_sha256']==core.digest(__file__) and audit['reused_worker_sha256']==core.digest(la.__file__)
    assert core.digest(root/'manifest.parquet')==audit['manifest_sha256']
    assert core.digest(REFERENCE/'model/BestCLS_50_21DF.pt')==la.EXPECTED_HASH
    if o.stage=='extract':
        qc=json.loads((root/'qc/summary.json').read_text());assert qc['status']=='pass' and qc['single_utterance_reference_verified']
        assert (root/'qc.exit').read_text().strip()=='0'
    phase=root/o.stage;phase.mkdir(exist_ok=False)
    core.save_json(phase/'run.json',dict(stage=o.stage,start_epoch=time.time(),script_sha256=core.digest(__file__),manifest_sha256=audit['manifest_sha256']))
    mp.spawn(la.worker,args=(str(root),str(REFERENCE),o.stage,o.stage,1,True),nprocs=2,join=True)
    core.finalize(root,o.stage,o.stage)
    summary=json.loads((phase/'summary.json').read_text())
    if o.stage=='qc':
        models=[json.loads((phase/f'rank{r}_model.json').read_text()) for r in [0,1]]
        err={k:max(m['single_utterance_reference_comparison'][k] for m in models) for k in ['z','h','score']}
        assert max(err.values())<=1e-5
        summary.update(single_utterance_reference_verified=True,single_utterance_reference_max_errors=err)
    summary.update(dataset=o.dataset,checkpoint_sha256=la.EXPECTED_HASH,original_EBM_frozen=True)
    core.save_json(phase/'summary.json',summary)
    archived,_=core.settings(REFERENCE)
    assert audit['archive_code_sha256']=={str(p.relative_to(archived)):core.digest(p) for p in archived.rglob('*.py')}

if __name__=='__main__':main()
