"""Coverage schedules retain priors, quotas, independent streams and DDP sharding."""
import itertools
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import test_positive_bpl as base_tests
from positive_bpl import CoverageSourcePlan,MatchedBatchSampler,validate_options


class SourceCoverageTests(unittest.TestCase):
    def test_full_coverage_and_marginals_at_random_slots(self):
        plan=CoverageSourcePlan([11,22,33],[.24,.26,.50],5)
        np.testing.assert_allclose(plan.expected_probability,[.24,.26,.50],atol=1e-14)
        rng=np.random.default_rng(17);counts=np.zeros(3);first=np.zeros(3)
        for _ in range(6000):
            schedule=plan.draw(rng)
            self.assertEqual(len(schedule),5)
            self.assertEqual(set(schedule),{11,22,33})
            counts+=np.array([(schedule==index).sum() for index in plan.indices])
            first+=np.array([schedule[0]==index for index in plan.indices])
        np.testing.assert_allclose(counts/30000,plan.probabilities,atol=.01,rtol=0)
        np.testing.assert_allclose(first/6000,plan.probabilities,atol=.02,rtol=0)

    def test_infeasible_budgets_fail_without_changing_priors(self):
        for probabilities,total in [([.2,.3,.5],4),([.2,.3,.5],3),([1/3]*3,2)]:
            with self.assertRaisesRegex(ValueError,'cannot retain source priors'):
                CoverageSourcePlan([0,1,2],probabilities,total)
        plan=CoverageSourcePlan([0,1,2],[1/3]*3,3)
        self.assertEqual(plan.extra,0)
        self.assertEqual(set(plan.draw(np.random.default_rng(1))),{0,1,2})
        with self.assertRaises(ValueError):CoverageSourcePlan([0,0],[.5,.5],4)
        with self.assertRaises(ValueError):validate_options({'matched_source_sampling':'coverage'})

    def test_rank_union_covers_every_original_and_OC_control_streams_match(self):
        fixture=base_tests.MatchedTests();fixture.setUp();self.addCleanup(fixture.temp.cleanup)
        ds=fixture.dataset();args=dict(fixture.args,matched_source_sampling='coverage',bpl_steps_per_epoch=2)
        roles=[set(),set(),set()];epoch_indices=[]
        for rank in range(2):
            independent=MatchedBatchSampler(ds,dict(args,rank=rank))
            same=MatchedBatchSampler(ds,dict(args,rank=rank,bpl_pairing='matched_same_source'))
            independent.set_epoch(9);same.set_epoch(9)
            batches=list(independent);self.assertEqual(batches,list(independent))
            epoch_indices.extend([v.index for batch in batches for v in batch])
            for a,b in zip(batches,same):
                self.assertEqual(a[:120],b[:120])
                self.assertEqual([v.index for v in b[:12]],[v.index for v in b[120:]])
                self.assertEqual([v.augmentation_seed for v in a],[v.augmentation_seed for v in b])
                self.assertEqual([ds.items[v.index].label for v in a],[1]*12+[0]*108+[1]*12)
                for target,part in zip(roles,[a[:12],a[12:120],a[120:]]):target.update(v.index for v in part)
            independent.set_epoch(10)
            self.assertNotEqual(batches,list(independent))
        self.assertEqual(roles,[set(ds.class_indices[1]),set(ds.class_indices[0]),set(ds.class_indices[1])])
        self.assertEqual(len(epoch_indices),2*2*132)

    def test_replacement_default_is_unchanged(self):
        fixture=base_tests.MatchedTests();fixture.setUp();self.addCleanup(fixture.temp.cleanup)
        ds=fixture.dataset()
        a=MatchedBatchSampler(ds,fixture.args)
        b=MatchedBatchSampler(ds,dict(fixture.args,matched_source_sampling='replacement'))
        self.assertEqual(list(itertools.islice(a,3)),list(itertools.islice(b,3)))


if __name__=='__main__':unittest.main(verbosity=2)
