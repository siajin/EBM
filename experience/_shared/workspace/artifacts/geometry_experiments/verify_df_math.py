"""Synthetic checks for DF sampling, deterministic controls and paired EER."""

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


folder = Path(__file__).parent
for alias, file in [("fast_analyze_v1", "fast_analyze.py"), ("df_extract_v1", "df_extract.py"), ("df_screen_v1", "df_screen.py")]:
    spec = importlib.util.spec_from_file_location(alias, folder / file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
screen = sys.modules["df_screen_v1"]
checks = []
rng = np.random.default_rng(20261105)
for i in range(60):
    y = np.r_[np.ones(41), np.zeros(129)].astype(int)
    score = rng.normal(size=len(y))
    if i % 3 == 0:
        score = np.round(score, 1)  # Includes ties and repeated bootstrap rows.
    codes = rng.integers(0, 16, size=len(y))
    counts = rng.integers(0, 5, size=16)
    weights = counts[codes]
    expected = screen.eer(np.repeat(y, weights), np.repeat(score, weights))
    prepared = screen.weighted_eer_preparation(y, score, codes)
    actual = screen.weighted_eer(prepared, counts)
    assert abs(expected - actual) < 1e-10
checks.append("60 weighted/tied EER cases match explicit sample replication")
for score, expected in [(y, 0), (-y, 100), (np.zeros(len(y)), 50)]:
    prepared = screen.weighted_eer_preparation(y, score, np.arange(len(y)))
    assert abs(screen.weighted_eer(prepared, np.ones(len(y))) - expected) < 1e-10
checks.append("perfect, reversed and all-tied EER boundaries")
pool = pd.DataFrame({"utterance_id": [f"s{i}" for i in range(19000)] + [f"b{i}" for i in range(2200)],
                     "y": [0]*19000 + [1]*2200})
a = screen.sample_pool(pool)
b = screen.sample_pool(pool.sample(frac=1, random_state=32))
assert a.utterance_id.tolist() == b.utterance_id.tolist()
assert a.y.sum() == 2000 and len(a) == 20000 and not a.utterance_id.duplicated().any()
assert set(a.loc[a.y == 1, "inclusion_probability"]) == {2000/2200}
assert set(a.loc[a.y == 0, "inclusion_probability"]) == {18000/19000}
checks.append("fixed class-stratified sample independent of protocol row order")
noise = screen.id_noise(["a", "b", "c"])
assert np.max(np.abs(noise - screen.id_noise(["c", "a", "b"])[[1, 2, 0]])) == 0
assert np.allclose(np.linalg.norm(noise, axis=1), 1)
checks.append("per-ID unit noise reproducible independent of evaluation order")
z = rng.normal(size=(60, 160))
c = rng.normal(size=160)
c /= np.linalg.norm(c)
cov = np.cov(rng.normal(size=(1000, 160)).T)
for mode in ["diagonal", "full"]:
    metric = screen.covariance_metric(cov, mode, 1.0)
    assert np.max(np.abs(metric-np.eye(160))) < 1e-12
    assert np.max(np.abs(screen.metric_score(z, c, metric) - z / np.linalg.norm(z, axis=1, keepdims=True) @ c)) < 1e-12
checks.append("both gamma=1 covariance metrics reproduce cosine")
print(json.dumps({"status": "pass", "checks": checks}, indent=2))
