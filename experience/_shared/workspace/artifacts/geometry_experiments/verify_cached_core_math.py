"""Meaningful boundary checks for the new masking, matching and uncertainty math."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from fast_analyze import eer, balanced_ll
from cached_core_diagnostics import (masked_cosine, match_pairs, stable_noise,
                                     weighted_eer_setup, weighted_eer, ll_intervals)


def main():
    checks = []
    rng = np.random.default_rng(42)
    z = rng.normal(size=(40, 7))
    c = rng.normal(size=7)
    c /= np.linalg.norm(c)
    original = (z@c)/np.linalg.norm(z, axis=1)
    assert np.allclose(masked_cosine(z, c, []), original, atol=1e-14)
    # Explicit independent zero-mask formula catches omission of centroid renorm.
    zm, cm = z.copy(), c.copy()
    zm[:, [1, 3]] = 0
    cm[[1, 3]] = 0
    expected = np.sum(zm*cm, axis=1)/(np.sqrt(np.sum(zm*zm, axis=1))*np.sqrt(np.sum(cm*cm)))
    assert np.allclose(masked_cosine(z, c, [1, 3]), expected, atol=1e-14)
    for indices in [list(range(7))]:
        try:
            masked_cosine(z, c, indices)
        except ValueError:
            pass
        else:
            raise AssertionError("Undefined masked cosine accepted")
    checks.append("identity and explicit joint-mask renormalization; zero norm refusal")
    # Ties, inverse ordering, missing-class bootstrap and explicit replicated rows.
    for j in range(50):
        y = np.r_[np.ones(7), np.zeros(9)].astype(int)
        scores = rng.integers(-3, 4, len(y)).astype(float)
        groups = np.arange(len(y)) % 4
        counts = rng.integers(1, 5, 4)
        prepared = weighted_eer_setup(y, scores, groups)
        repeated = np.repeat(np.arange(len(y)), counts[groups])
        assert abs(weighted_eer(prepared, counts)-eer(y[repeated], scores[repeated])) < 1e-10
    y = np.array([0, 0, 1, 1])
    prepared = weighted_eer_setup(y, np.array([0., 1., 2., 3.]), np.arange(4))
    assert np.isnan(weighted_eer(prepared, np.array([1, 1, 0, 0])))
    checks.append("50 weighted tied EER cases against explicit sample replication; missing class")
    ids = np.array(["a", "b", "c"])
    noise = stable_noise(ids, c)
    assert np.max(np.abs(noise@c)) < 1e-12
    assert np.allclose(np.linalg.norm(noise, axis=1), 1)
    assert np.array_equal(stable_noise(ids[::-1], c)[::-1], noise)
    checks.append("noise stable by ID and perpendicular to centroid")
    rows = pd.DataFrame({"utterance_id": ["b1", "b2", "s1", "s2", "other"],
                         "y": [1, 1, 0, 0, 0], "source_corpus": ["a"]*4+["b"],
                         "codec": ["x"]*5})
    geom = {"cosine": np.array([.1, .105, .102, .8, .1]),
            "norm": np.array([2., 2., 2., 2., 2.]), "valid": np.ones(5, dtype=bool)}
    matched = match_pairs(SimpleNamespace(rows=rows), geom, ["source_corpus", "codec"], .01, .001)
    assert len(matched) == 1 and matched[0, 1] == 2
    assert match_pairs(SimpleNamespace(rows=rows), geom, ["source_corpus", "codec"], .00001, .001).shape == (0, 2)
    bona_only = SimpleNamespace(rows=rows.iloc[:2].copy())
    assert match_pairs(bona_only, {k: v[:2] for k, v in geom.items()}, ["source_corpus", "codec"], .01, .001).shape == (0, 2)
    checks.append("matching caliper, exact metadata strata, no replacement and empty support")
    y = np.tile([0, 1], 10)
    groups = np.repeat(np.arange(10), 2)
    b = np.where(y, .6, .4)
    a = np.where(y, .8, .2)
    result = ll_intervals(y, {"b": b, "a": a}, groups, [("b", "a")], draws=200)
    exact = balanced_ll(y, b)-balanced_ll(y, a)
    assert np.allclose(result[0]["CI95"], exact)
    assert exact > 0 and result[0]["dropped_replicates"] == 0
    checks.append("balanced paired logloss delta sign and exact constant group effect")
    print(json.dumps({"status": "pass", "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
