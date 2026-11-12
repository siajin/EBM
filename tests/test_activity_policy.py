"""Synthetic activity policy tests; no real dev/eval or audio is accessed."""
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import unittest
import numpy as np
import soundfile as sf
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import test_positive_bpl as base_tests
from positive_bpl import MatchedTrainSet,MatchedBatchSampler,validate_options
from activity_policy import ActivityPolicy,activity_bins
from dataset.ASVspoof2019_la_train import load_train_items_only
from tools.fit_activity_policy import fit_flow


class ActivityTests(unittest.TestCase):
    def setUp(self):
        self.fixture=base_tests.MatchedTests();self.fixture.setUp();self.addCleanup(self.fixture.temp.cleanup)
        self.root=Path(self.fixture.temp.name)
        for i,item in enumerate(self.fixture.items):
            length=self.fixture.lengths[item.path]
            path=self.root/f'source_{i}.wav'
            sf.write(path,.1*np.sin(np.arange(length)*.013),16000)
            item.path=str(path)
        self.args=self.fixture.args
        self.ds=MatchedTrainSet(self.args,self.fixture.items)
        offsets=[0];starts=[];voice=[];prob=[]
        for i,item in enumerate(self.ds.items):
            for length in self.ds.targets:
                s=sorted(set([t*int(length) for t in range(int(self.ds.lengths[i]//length))]
                    +[int(self.ds.lengths[i]-(t+1)*length) for t in range(int(self.ds.lengths[i]//length))]))
                offsets.append(offsets[-1]+len(s));starts.extend(s)
                voice.extend([int(length//320)]*len(s));prob.extend([1/len(s)]*len(s) if s else [])
        self.data={'context_offsets':np.array(offsets,dtype=np.int64),'starts':np.array(starts),
            'speech_frames':np.array(voice),'position_probabilities':np.array(prob),
            'length_probabilities':np.stack([self.ds.tables[self.ds.keys[i]][self.ds.buckets[i]-1] for i in range(len(self.ds))]),
            'source_uids':np.array([Path(v.path).stem for v in self.ds.items]),
            'source_sizes':np.array([Path(v.path).stat().st_size for v in self.ds.items]),
            'source_mtime_ns':np.array([Path(v.path).stat().st_mtime_ns for v in self.ds.items])}
        self.path=self.root/'activity.json';self.write()

    def write(self):
        array=self.root/'activity.npz';np.savez_compressed(array,**self.data)
        self.meta={'schema_version':1,'reference':'raw_before_RawBoost',
                   'activity_matching_scope':'class_global_with_source_marginals',
                   'duration_policy_sha256':self.ds.policy_sha256,'source_fingerprint':self.ds.source_fingerprint,
                   'arrays_file':array.name,'arrays_sha256':hashlib.sha256(array.read_bytes()).hexdigest(),
                   'activity_bins':['zero','(0,.2]','(.2,.4]','(.4,.6]','(.6,.8]','(.8,1]']}
        self.path.write_text(json.dumps(self.meta))

    def test_valid_cache_keeps_sources_and_augmentation_streams(self):
        joint=MatchedTrainSet(dict(self.args,activity_policy_path=str(self.path)),self.fixture.items)
        np.testing.assert_array_equal(self.ds.source_probability,joint.source_probability)
        self.assertLess(joint.activity_policy.statistics['max_conditional_bin_error'],1e-12)
        a=MatchedBatchSampler(self.ds,self.args);b=MatchedBatchSampler(joint,self.args)
        a.set_epoch(4);b.set_epoch(4)
        for before,after in zip(itertools.islice(a,3),itertools.islice(b,3)):
            self.assertEqual([v.index for v in before],[v.index for v in after])
            self.assertEqual([v.augmentation_seed for v in before],[v.augmentation_seed for v in after])
            self.assertEqual([v.stop-v.start for v in before],[v.stop-v.start for v in after])
            for spec in after:self.assertLessEqual(spec.stop,joint.lengths[spec.index])

    def test_cache_integrity_and_scope_are_required(self):
        self.meta['arrays_sha256']='0'*64;self.path.write_text(json.dumps(self.meta))
        with self.assertRaisesRegex(ValueError,'SHA mismatch'):ActivityPolicy(self.path,self.ds)
        self.write();self.meta['source_fingerprint']='other';self.path.write_text(json.dumps(self.meta))
        with self.assertRaisesRegex(ValueError,'differs from Train'):ActivityPolicy(self.path,self.ds)
        with self.assertRaises(ValueError):validate_options(dict(self.args,bpl_pairing='legacy_long_short',activity_policy_path=str(self.path)))

    def test_original_and_group_length_marginals_are_checked(self):
        self.data['length_probabilities'][0]*=0;self.write()
        with self.assertRaisesRegex(ValueError,'normalize for every original'):ActivityPolicy(self.path,self.ds)
        self.data['length_probabilities'][0]=[1.,0.]
        self.data['length_probabilities'][1]=[.2,.8];self.write()
        with self.assertRaisesRegex(ValueError,'duration targets'):ActivityPolicy(self.path,self.ds)

    def test_unbalanced_activity_is_rejected(self):
        self.data['speech_frames'][0]=0;self.write()
        with self.assertRaisesRegex(ValueError,'distributions/means'):ActivityPolicy(self.path,self.ds)

    def test_global_activity_can_match_without_claiming_within_speaker_equality(self):
        k=len(self.ds.targets)
        for i,item in enumerate(self.ds.items):
            full=(item.speaker=='s0') == bool(item.label)
            for length_id,length in enumerate(self.ds.targets):
                a,b=self.data['context_offsets'][i*k+length_id:i*k+length_id+2]
                self.data['speech_frames'][a:b]=int(length//320) if full else 0
        self.write();policy=ActivityPolicy(self.path,self.ds)
        self.assertLess(policy.statistics['max_conditional_bin_error'],1e-12)
        self.assertGreater(policy.statistics['max_within_speaker_bin_difference'],.9)

    def test_changed_source_is_rejected_without_reading_all_audio(self):
        path=Path(self.ds.items[0].path);stat=path.stat()
        os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns+1000000))
        with self.assertRaisesRegex(ValueError,'files changed'):ActivityPolicy(self.path,self.ds)

    def test_bins_and_Train_only_protocol_reader(self):
        np.testing.assert_array_equal(activity_bins(np.array([0,1,5,6,10,15,20,25]),25),[0,1,1,2,2,3,4,5])
        folder=self.root/'dataset/LA/ASVspoof2019_LA_cm_protocols';folder.mkdir(parents=True)
        (folder/'ASVspoof2019.LA.cm.train.trn.txt').write_text('s u1 - - bonafide\ns u2 - A01 spoof\n')
        # No dev protocol exists: a reader touching dev would fail here.
        items=load_train_items_only(str(self.root/'dataset'))
        self.assertEqual([(v.label,v.type) for v in items],[(1,'bonafide'),(0,'A01')])

    def test_probability_fit_matches_bins_and_means_with_originals_retained(self):
        base=np.array([.5,.5,.05,.45,.25,.25]);row=np.array([0,1,2,2,3,3])
        target=np.full(4,.5);labels=np.array([1,1,0,0,0,0]);length=np.zeros(6,dtype=int)
        voice=np.array([.25,.75,.25,.75,.25,.75]);bins=activity_bins((voice*100).astype(int),100)
        flow,report=fit_flow(base,row,target,None,None,labels,length,bins,voice,np.array([1.]),1,maxiter=500)
        np.testing.assert_allclose(np.bincount(row,weights=flow),target,atol=1e-8)
        self.assertLess(max(report['errors'].values()),2e-8)
        self.assertTrue((flow>0).all())

    def test_infeasible_support_never_drops_an_original_silently(self):
        with self.assertRaises(ValueError):
            fit_flow(np.ones(2),np.array([0,1]),np.ones(2),None,None,np.array([1,0]),
                     np.zeros(2,dtype=int),np.array([1,5]),np.array([.1,1.]),np.array([1.]),1,maxiter=20)


if __name__=='__main__':unittest.main(verbosity=2)
