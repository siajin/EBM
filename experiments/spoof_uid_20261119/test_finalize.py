import json,math,tempfile,unittest
from pathlib import Path
from finalize import independent_eer,prepare_shared
from role_lifecycle import calculate_eer
from role_objectives import RoleConfig
import torch

class Checks(unittest.TestCase):
    def test_independent_ROC_with_ties(self):
        torch.manual_seed(12)
        for n in [8,101,1000]:
            y=[i%2 for i in range(n)];s=torch.randint(-3,4,(n,)).tolist()
            self.assertAlmostEqual(independent_eer(y,s),calculate_eer(s,y),places=10)
    def test_shared_prepare_rejects_training_not_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);(p/'stages').mkdir();(p/'stages/training_complete.json').write_text(json.dumps(dict(status='running',saved_epochs=[5],results=[])))
            with self.assertRaises(AssertionError):prepare_shared(p/'analysis',p)
            self.assertFalse((p/'analysis/shared_eval').exists())

if __name__=='__main__':unittest.main(verbosity=2)
