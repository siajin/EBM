import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

TOOLS=Path(__file__).resolve().parents[1]/'tools'
sys.path.insert(0,str(TOOLS))
import probe_repetition_shortcut as probe
from summarize_repetition_shortcut import main as summarize
from analyze_repetition_shortcut import eer

class RepetitionControls(unittest.TestCase):
    def test_waveform_controls(self):
        x=np.random.default_rng(12).normal(size=100000).astype(np.float32)
        views=dict(probe.dose_views(x));self.assertEqual(len(views),31)
        np.testing.assert_array_equal(views['head_2s_native'],views['head_2s_repeat2'])
        np.testing.assert_array_equal(views['tail_1s_native'],x[-16000:])
        np.testing.assert_array_equal(views['head_1s_repeat4'][:16000],x[:16000])
        wrap=views['head_1s_repeat4'];smooth=views['head_1s_smooth_seams4']
        changed=np.flatnonzero(wrap!=smooth)
        self.assertTrue(len(changed)>0)
        self.assertTrue(all(min(abs(int(i)-p) for p in [16000,32000,48000,64000])<=80 for i in changed))
        for _,a in views.items():self.assertTrue(np.isfinite(a).all());self.assertEqual(a.dtype,np.float32)

    def test_weighted_eer_class_prior_invariant_and_ties(self):
        self.assertAlmostEqual(eer([0,1],[0,1]),0)
        self.assertAlmostEqual(eer([0,1],[.5,.5]),50)
        self.assertAlmostEqual(eer([0,0,1],[.5,.5,.5]),50)
        self.assertAlmostEqual(eer([0,1],[.5,.5],[10,1]),50)

    def test_derived_summary_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);records=[]
            for phase in ['dev','21LA']:
                for y in [0,1]:
                    for i,n in enumerate([30000,80000]):
                        records.append(dict(phase=phase,index=y*2+i,uid=f'{phase}_{y}_{i}',speaker=f's{y}_{i}',label=y,original_samples=n,attack='bonafide' if y else 'A10'))
            selected=pd.DataFrame(records);selected.to_csv(root/'selected.csv',index=False)
            subset=selected[(selected.phase=='21LA')&(selected.original_samples>64600)];subset.to_csv(root/'probe_selected.csv',index=False)
            p={'selection_hashes':{f:probe.sha(root/f) for f in ['selected.csv','probe_selected.csv']},'limits':['synthetic test fixture']}
            (root/'protocol.json').write_text(json.dumps(p))
            meta=[' '.join([r['speaker'],r['uid'],'nocodec','x',r['attack'],'bonafide' if r['label'] else 'spoof','x','eval']) for r in records if r['phase']=='21LA']
            (root/'trial_metadata_21LA.txt').write_text('\n'.join(meta))
            for kind,frame in [('full',selected),('probe',subset)]:
                for rank in [0,1]:
                    rows=[]
                    for r in frame.iloc[rank::2].to_dict('records'):
                        conditions=[name for name,_ in probe.dose_views(np.ones(80000,dtype=np.float32))] if kind=='probe' else ['native','wrap4']
                        for condition in conditions:
                            row=r.copy();row.update(condition=condition,used_samples=min(r['original_samples'],64600),wave_sha256=r['uid'],gpu_rank=rank,score_legacy240=float(r['label'])*1.6-.8,score_B=float(r['label'])*1.2-.6);rows.append(row)
                    file=root/f'{kind}_rank{rank}.csv';pd.DataFrame(rows,columns=probe.FIELDS).to_csv(file,index=False)
                    (root/f'{kind}_done_rank{rank}.json').write_text(json.dumps({'sha256':probe.sha(file),'rows':len(rows)}))
            probe.validate(root,False,False);probe.validate(root,True,False);summarize(root)
            result=json.loads((root/'analysis_complete.json').read_text())
            self.assertEqual(result['status'],'analysis_complete')
            self.assertTrue(all(m['eer']==0 for m in result['full_metrics']))
            self.assertTrue(all(m['delta']==0 for m in result['paired_effects']))
            # Deliberate score-file corruption must be rejected.
            with (root/'full_rank0.csv').open('a') as f:f.write('\n')
            with self.assertRaises(AssertionError):probe.validate(root,False,False)

if __name__=='__main__':unittest.main()
