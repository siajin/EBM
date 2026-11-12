"""Offline access to saved embeddings and predictions. No SSH or GPU imports."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "geometry_cache"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def resolve_run(dataset, cache_root=DEFAULT_ROOT):
    cache_root = Path(cache_root).resolve()
    catalog = read_json(cache_root / "catalog.json")
    matches = [entry for entry in catalog["runs"].values() if dataset in entry.get("datasets", {})]
    if not matches:
        names = sorted({name for entry in catalog['runs'].values() for name in entry.get('datasets', {})})
        raise KeyError(f"Unknown dataset {dataset}; catalog names are {', '.join(names)}")
    if len(matches) > 1:
        raise ValueError("Multiple model/recipe caches match this dataset; use a catalog with one explicit version")
    entry = matches[0]
    if entry["status"] != "ready" or "local_folder" not in entry:
        raise FileNotFoundError(f"{dataset} is not fully extracted/cached yet")
    folder = (cache_root / entry["local_folder"]).resolve()
    folder.relative_to(cache_root)
    return folder, entry, entry["datasets"][dataset]


@dataclass
class CachedEmbeddings:
    z: np.ndarray
    h: np.ndarray
    rows: pd.DataFrame
    centroid: np.ndarray
    provenance: dict

    def geometry(self):
        norm = np.linalg.norm(self.z.astype("float64"), axis=1)
        unit = self.z / norm[:, None]
        cosine = unit @ self.centroid
        tangent = unit - cosine[:, None]*self.centroid
        tangent_norm = np.linalg.norm(tangent, axis=1)
        valid = tangent_norm >= 1e-4
        direction = np.zeros_like(tangent)
        direction[valid] = tangent[valid] / tangent_norm[valid, None]
        return {"norm": norm, "unit_z": unit, "cosine": cosine,
                "tangent_norm": tangent_norm, "valid": valid, "direction": direction}


def validate_run(dataset, cache_root=DEFAULT_ROOT, verify_hashes=True):
    folder, entry, split = resolve_run(dataset, cache_root)
    audit = read_json(folder / "input_audit.json")
    summary = read_json(folder / "extract/summary.json")
    assert audit["checkpoint_sha256"] == entry["identity"]["checkpoint_sha256"]
    assert audit["manifest_sha256"] == entry["identity"]["manifest_sha256"]
    assert summary["status"] == "pass" and summary["Q_evaluated"] == ('Q' in entry['datasets'])
    for file in entry["files"]:
        path = folder / file["path"]
        assert path.is_file() and path.stat().st_size == file["bytes"]
        if verify_hashes:
            assert _digest(path) == file["sha256"], "Cached file checksum mismatch"
    rows = pd.read_parquet(folder / "extract/scores.parquet")
    assert len(rows) == summary["sample_count"] and not rows.utterance_id.duplicated().any()
    cursor = 0
    for rank in [0, 1]:
        shard = pd.read_parquet(folder / f"extract/rank{rank}_rows.parquet")
        n = len(shard)
        selected = rows.iloc[cursor:cursor+n]
        assert shard.utterance_id.tolist() == selected.utterance_id.tolist()
        assert np.array_equal(selected["rank"], np.full(n, rank))
        assert np.array_equal(selected.embedding_row, np.arange(n))
        c = np.load(folder / f"extract/rank{rank}_centroid.npy", allow_pickle=False)
        assert c.shape == (160,) and abs(np.linalg.norm(c)-1) < 1e-10
        for kind in ["z", "h"]:
            array = np.load(folder / f"extract/rank{rank}_{kind}.npy", mmap_mode="r", allow_pickle=False)
            assert array.shape == (n, 160) and array.dtype == np.dtype("float32")
            assert np.isfinite(array).all()
        z = np.load(folder / f"extract/rank{rank}_z.npy", allow_pickle=False).astype("float64")
        score = z / np.linalg.norm(z, axis=1, keepdims=True) @ c
        assert np.max(np.abs(score-selected.cosine_s.to_numpy())) < 1e-10
        assert np.max(np.abs(score-selected.raw_score.to_numpy())) < 1e-5
        cursor += n
    assert cursor == len(rows)
    return {"status": "pass", "dataset": dataset, "rows": int((rows.split == split).sum()),
            "cache_fingerprint": entry["cache_fingerprint"], "file_hashes_checked": verify_hashes}


def load_embeddings(dataset, cache_root=DEFAULT_ROOT, expected_checkpoint_sha256=None):
    folder, entry, split = resolve_run(dataset, cache_root)
    if expected_checkpoint_sha256 is not None:
        if entry["identity"]["checkpoint_sha256"] != expected_checkpoint_sha256:
            raise ValueError("Checkpoint differs; use a matching cache rather than mixing models")
    rows = pd.read_parquet(folder / "extract/scores.parquet")
    mask = rows.split.to_numpy() == split
    vectors = {kind: np.concatenate([np.load(folder / f"extract/rank{rank}_{kind}.npy", allow_pickle=False)
                                    for rank in [0, 1]])[mask] for kind in ["z", "h"]}
    c = np.load(folder / "extract/rank0_centroid.npy", allow_pickle=False)
    return CachedEmbeddings(vectors["z"], vectors["h"], rows.loc[mask].reset_index(drop=True), c, entry)


def load_predictions(dataset, cache_root=DEFAULT_ROOT):
    folder, entry, split = resolve_run(dataset, cache_root)
    if dataset == "V":
        path = folder / "fast_analysis/C_V_predictions.parquet"
    elif dataset == "DF_pilot":
        path = folder / "DF_evaluation/predictions.parquet"
    else:
        frame = pd.read_parquet(folder / "extract/scores.parquet")
        return frame.loc[frame.split == split].reset_index(drop=True)
    if not path.exists():
        raise FileNotFoundError("Predictions are not complete/cached yet; sync again after evaluation")
    return pd.read_parquet(path)
