import ast
from pathlib import Path
import shutil
import hashlib
import tempfile
import unittest
import numpy as np
from mild_duration import MildTrainSet, MildSpec, tempo, transform_parameters


class MildDurationTests(unittest.TestCase):
    def test_integer_crop_and_speed_bounds(self):
        rng=np.random.default_rng(1729)
        for samples in rng.integers(8000,480000,200):
            for ratio in [2/3,.75,.95,1.,1.1,1.25]:
                kept,rate,output=transform_parameters(int(samples),float(samples*ratio))
                self.assertGreaterEqual(kept,.8*samples)
                self.assertLessEqual(kept,samples)
                self.assertTrue(.8<=rate<=1.2)
                self.assertLessEqual(abs(output-samples*ratio),2)

    def test_outside_constraints_fails(self):
        for target in [0,float('nan'),100,200000]:
            with self.assertRaises(ValueError): transform_parameters(48000,target)

    def test_cached_waveform_digest_and_random_target(self):
        with tempfile.TemporaryDirectory() as directory:
            ds=MildTrainSet.__new__(MildTrainSet)
            ds.cache=Path(directory)
            wave=np.arange(8000,dtype=np.float32)/8000
            np.save(ds.cache/'test.npy',wave)
            ds.wave_hashes={'test.npy':hashlib.sha256(wave.tobytes()).hexdigest()}
            ds.records=[{'variants':[{'file':'test.npy','output_samples':8000,'probability':.25},
                                    {'file':'test.npy','output_samples':8000,'probability':.75}]}]
            self.assertTrue(np.array_equal(ds.read_crop(MildSpec(0,0,1),False).numpy(),wave))
            choices=[ds.crop_spec(0,[1,epoch,0]).variant for epoch in range(2000)]
            self.assertLess(abs(np.mean(choices)-.75),.04)
            self.assertEqual(ds.crop_spec(0,[1,4,0]),ds.crop_spec(0,[1,4,0]))
            np.save(ds.cache/'test.npy',wave+1)
            with self.assertRaises(ValueError): ds.read_crop(MildSpec(0,0,1),False)

    @unittest.skipUnless(shutil.which('sox'),'SoX available only on execution server')
    def test_sox_duration_pitch_and_identity(self):
        wave=(.1*np.sin(2*np.pi*440*np.arange(48000)/16000)).astype(np.float32)
        for rate in [.8,.912345,1.,1.12345,1.2]:
            out=tempo(wave,rate)
            self.assertEqual(len(out),int(np.floor(len(wave)/rate+.5)))
            peak=np.argmax(abs(np.fft.rfft(out)))*16000/len(out)
            self.assertLess(abs(peak-440),2)
            if rate==1: self.assertTrue(np.array_equal(out,wave))

    def test_actual_runner_uses_19la_and_fixed20(self):
        root=Path(__file__).resolve().parents[1]
        tree=ast.parse((root/'tools/train_mild_duration.py').read_text())
        trainer=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='AuditedTrainer')
        method=next(n for n in trainer.body if isinstance(n,ast.FunctionDef) and n.name=='evaluation')
        ns={}; exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),'dispatch','exec'),ns)
        calls=[]
        class Probe:
            def test(self,phase,epoch): calls.append((phase,epoch))
        for epoch in [5,10,15,20]: ns['evaluation'](Probe(),epoch)
        self.assertEqual(calls,[('19LA',e) for e in [5,10,15,20]])


if __name__=='__main__': unittest.main()
