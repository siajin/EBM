"""Recompute native and saved-probe metrics locally without server/GPU access."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fast_analyze import metrics
from geometry_cache import DEFAULT_ROOT, load_embeddings, load_predictions, read_json, resolve_run, validate_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--datasets", nargs="+")
    args = parser.parse_args()
    if args.datasets is None:
        catalog = read_json(args.cache_root / 'catalog.json')
        args.datasets = [name for entry in catalog['runs'].values()
                         if entry.get('status') == 'ready' and entry.get('local_folder')
                         for name in entry.get('datasets', {})]
    rows, checks, missing = [], [], []
    cached = []
    checkpoints = set()
    for dataset in args.datasets:
        try:
            folder, entry, split = resolve_run(dataset, args.cache_root)
        except (FileNotFoundError, KeyError) as error:
            missing.append({"dataset": dataset, "reason": str(error)})
            continue
        checks.append(validate_run(dataset, args.cache_root))
        data = load_embeddings(dataset, args.cache_root)
        y = data.rows.y.to_numpy()
        geometry = data.geometry()
        assert np.max(np.abs(geometry["cosine"]-data.rows.cosine_s.to_numpy())) < 1e-10
        native = metrics(y, geometry["cosine"])
        rows.append({"dataset": dataset, "case": "native_cosine", "bona": int(y.sum()),
                     "spoof": int((1-y).sum()), **native})
        if dataset == "V":
            reference = read_json(folder / "fast_analysis/summary.json")
            assert abs(native["eer_percent"]-reference["baseline_native_V"]["eer_percent"]) < 1e-9
            prediction_reference = reference["C"]
        elif dataset == "DF_pilot" and entry["analysis_complete"]:
            prediction_reference = read_json(folder / "DF_evaluation/metrics.json")
        else:
            prediction_reference = []
        if dataset == '21LA_eval':
            reference_native = read_json(folder / 'extract/summary.json')['native_metrics']
            assert all(abs(native[key] - value) < 1e-9 for key, value in reference_native.items())
        if prediction_reference:
            predictions = load_predictions(dataset, args.cache_root)
            assert set(predictions.utterance_id) == set(data.rows.utterance_id)
            assert not predictions.utterance_id.duplicated().any()
            for record in prediction_reference:
                name = record["case"]
                probability = predictions[name].to_numpy() if "balanced_logloss" in record else None
                actual = metrics(predictions.y.to_numpy(), predictions[name].to_numpy(), probability)
                for key in actual:
                    assert abs(actual[key]-record[key]) < 1e-9, (dataset, name, key)
                if name != "native_cosine":
                    rows.append({"dataset": dataset, "case": name, "bona": int(y.sum()),
                                 "spoof": int((1-y).sum()), **actual})
        checkpoints.add(entry["identity"]["checkpoint_sha256"])
        cached.append({"dataset": dataset, "fingerprint": entry["cache_fingerprint"],
                       "z_shape": list(data.z.shape), "h_shape": list(data.h.shape),
                       "analysis_complete": entry["analysis_complete"]})
    assert rows, "No completed cached datasets are available"
    assert len(checkpoints) == 1, "Different checkpoint caches must be analyzed separately"
    identity = {"datasets": cached, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    tag = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    out = args.cache_root / "local_analysis" / tag
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    report = {"status": "pass", "offline": True, "GPU_inference_runs": 0, "SSH_required": False,
              "verified_utc": datetime.now(timezone.utc).isoformat(), "cached_datasets": cached,
              "validation": checks, "not_available": missing,
              "metrics_recomputed": len(rows), "metrics_file": str(out / "metrics.csv"),
              "scope": "T train diagnostic; V source-domain development; DF_pilot fixed 20000 sample; 21LA_eval full official eval when cached"}
    (out / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
