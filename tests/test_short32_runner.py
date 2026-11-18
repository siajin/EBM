"""Launch wiring only; subprocess/GPU training are never executed here."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('short32_runner',ROOT/'tools/run_short32_experiment.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def test_actual_cli_reaches_spawn_once_and_refuses_duplicate_claim(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'code').mkdir();source=root/'code/source.py';source.write_text('pass\n')
            (root/'experiment.json').write_text(json.dumps({'source_sha256':{'source.py':hashlib.sha256(source.read_bytes()).hexdigest()}}))
            original=Path.is_file
            with patch.object(sys,'argv',['runner',folder,'--mode','smoke']),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0,1'}),patch.object(Path,'is_file',lambda p:True if str(p)=='/.dockerenv' or p.name=='.dockerenv' else original(p)),patch.object(torch.multiprocessing,'spawn') as spawn,patch.object(torch.multiprocessing,'set_sharing_strategy') as sharing:
                runner.cli()
                spawn.assert_called_once();sharing.assert_called_once_with('file_system')
                self.assertEqual(spawn.call_args.kwargs['nprocs'],2)
                self.assertTrue((root/'smoke.claim').is_dir());self.assertFalse((root/'train.claim').exists())
                with self.assertRaises(FileExistsError):runner.cli()
                spawn.assert_called_once()

    def test_smoke_uses_original_sampler_with_zero_full_and_asymmetric_k(self):
        items=[SimpleNamespace(path=f'short{i}',label=1) for i in range(109)]
        items += [SimpleNamespace(path=f'long{i}',label=1) for i in range(2471)]
        items += [SimpleNamespace(path=f'spoof{i}',label=0) for i in range(22800)]
        before=torch.get_rng_state().clone()
        with patch('soundfile.info',side_effect=lambda p:SimpleNamespace(frames=16000 if p.startswith('short') else 64000)):
            selected=runner.smoke_items(items)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertEqual(len(selected),396)
        for rank,expected in [(0,[0,12,2]),(1,[0,3,0])]:
            s=torch.utils.data.DistributedSampler(range(60),num_replicas=2,rank=rank,shuffle=True);s.set_epoch(1)
            ids=list(s);counts=[]
            for a,b in [(0,12),(12,24),(24,30)]:counts.append(sum(selected[i].path.startswith('short') for i in ids[a:b]))
            self.assertEqual(counts,expected)


if __name__=='__main__':unittest.main()
