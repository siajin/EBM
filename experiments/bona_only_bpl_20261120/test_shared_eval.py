import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import torch
from shared_eval import independent_eer, probe_steps, score_heads, summarize
from role_lifecycle import calculate_eer


class FakeHead(torch.nn.Module):
    def __init__(self,scale):super().__init__();self.scale=scale
    def forward(self,x,phase):return {'score_bona':x.sum(1)*self.scale+1e-5*torch.randn(len(x))}


class SharedTests(unittest.TestCase):
    def test_tied_independent_ROC(self):
        torch.manual_seed(7)
        for n in [8,101,1000]:
            labels=[i%2 for i in range(n)];scores=torch.randint(-3,4,(n,)).tolist()
            self.assertAlmostEqual(independent_eer(labels,scores),calculate_eer(scores,labels),places=10)

    def test_actual_partial_batches_and_class_probe(self):
        trials=[SimpleNamespace(label=int(i in [8,17,34])) for i in range(35)]
        for rank in [0,1]:
            steps=probe_steps(trials,rank);ids=list(range(rank,35,2));selected=[i for s in steps for i in ids[s*8:s*8+8]]
            self.assertEqual({trials[i].label for i in selected},{0,1});self.assertEqual(steps[-1],(len(ids)-1)//8)
            self.assertEqual(len(selected),len(set(selected)))

    def test_head_order_and_RNG_are_isolated(self):
        torch.manual_seed(99);state=torch.get_rng_state().clone();x=torch.ones(8,4)
        a=score_heads({'e5':FakeHead(1),'e10':FakeHead(2)},x,'cpu',1,3,'21DF')
        b=score_heads({'e10':FakeHead(2),'e5':FakeHead(1)},x,'cpu',1,3,'21DF')
        self.assertEqual(a,b);self.assertTrue(torch.equal(state,torch.get_rng_state()))
        torch.testing.assert_close(torch.tensor([v-4 for v in a['e5']]),torch.tensor([v-8 for v in a['e10']]),atol=5e-7,rtol=0)

    def test_scores_merge_and_missing_duplicate_rejection(self):
        trials=[SimpleNamespace(path=f'/x/id{i}.wav',label=i%2) for i in range(12)]
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            def write(bad=False):
                for rank in [0,1]:
                    with (p/f'19LA_rank{rank}.csv').open('w',newline='') as f:
                        w=csv.writer(f);w.writerow(['index','uid','label','e5'])
                        for i in range(rank,12,2):w.writerow([i,'id0' if bad and i==3 else 'id'+str(i),i%2,float(i%2)])
            write();m=summarize(p,'19LA',{'e5':None},trials,{'e5':'hash'},True,list(range(12)))
            self.assertEqual(m['metrics']['e5']['EER_percent'],0.);self.assertEqual(m['rows'],12)
            write(True)
            with self.assertRaises(AssertionError):summarize(p,'19LA',{'e5':None},trials,{'e5':'hash'},True,list(range(12)))


if __name__=='__main__':unittest.main(verbosity=2)
