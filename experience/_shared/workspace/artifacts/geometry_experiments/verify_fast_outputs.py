"""Read-only numerical checks of completed server-side fast-screen outputs."""

import argparse
import json
import math
from pathlib import Path
import statistics

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    root = Path(parser.parse_args().root)
    assert Path("/.dockerenv").exists()
    for stage in ["extract", "fast_analysis"]:
        assert (root / (stage + ".exit")).read_text().strip() == "0"
        assert (root / stage / "complete").read_text().strip() == "0"
    out = root / "output_verification_v1.json"
    assert not out.exists(), "Preserve previous verification"
    rows = pd.read_parquet(root / "extract/scores.parquet")
    assert set(rows.split) == {"T", "V"}
    assert not rows.utterance_id.duplicated().any()
    z = np.concatenate([np.load(root / "extract" / f"rank{i}_z.npy") for i in [0, 1]]).astype("float64")
    for rank in [0, 1]:
        selected = rows.loc[rows["rank"] == rank, "embedding_row"].to_numpy()
        assert np.array_equal(selected, np.arange(len(selected)))
    c = np.load(root / "extract/rank0_centroid.npy")
    unit = z / np.linalg.norm(z, axis=1, keepdims=True)
    scalar = unit @ c
    cosine_error = float(np.max(np.abs(scalar - rows.cosine_s.to_numpy())))
    euclidean_score = 1 - 0.5 * np.square(unit - c).sum(axis=1)
    identity_error = float(np.max(np.abs(euclidean_score - scalar)))
    assert cosine_error < 1e-10 and identity_error < 1e-10
    val = rows.loc[rows.split == "V"]
    low_bona = float(val.loc[val.y == 1, "cosine_s"].min())
    high_spoof = float(val.loc[val.y == 0, "cosine_s"].max())
    gap = low_bona - high_spoof
    assert gap > 0, "Native EER zero requires a separating threshold here"
    pred = pd.read_parquet(root / "fast_analysis/C_V_predictions.parquet")
    assert pred.utterance_id.tolist() == val.utterance_id.tolist()
    assert np.array_equal(pred.y, val.y)
    summary = json.loads((root / "fast_analysis/summary.json").read_text())
    independent_loss = {}
    for case in summary["C"]:
        name = case["case"]
        positive = np.clip(pred.loc[pred.y == 1, name].to_numpy(), 1e-7, 1-1e-7)
        negative = np.clip(pred.loc[pred.y == 0, name].to_numpy(), 1e-7, 1-1e-7)
        loss = -0.5 * (math.fsum(math.log(float(v)) for v in positive)/len(positive)
                       + math.fsum(math.log1p(-float(v)) for v in negative)/len(negative))
        assert abs(loss - case["balanced_logloss"]) < 1e-12
        independent_loss[name] = loss
    random = [v for v in summary["A"] if v["case"].startswith("random25_")]
    report = {
        "status": "pass", "row_count": len(rows), "Q_rows": 0,
        "native_V_min_bona_cosine": low_bona,
        "native_V_max_spoof_cosine": high_spoof,
        "native_V_separating_gap": gap,
        "native_V_zero_EER_independently_verified": True,
        "row_and_prediction_alignment_verified": True,
        "recomputed_cosine_max_error": cosine_error,
        "unit_Euclidean_cosine_max_error": identity_error,
        "C_independently_recomputed_balanced_logloss": independent_loss,
        "random40_subsets": len(random),
        "random40_zero_EER_count": sum(v["eer_percent"] == 0 for v in random),
        "random40_balanced_logloss_median": statistics.median(v["balanced_logloss"] for v in random),
        "interpretation": "Known-attack V has native-score ceiling; no claim of novel-direction benefit or attention/loss effectiveness."
    }
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
