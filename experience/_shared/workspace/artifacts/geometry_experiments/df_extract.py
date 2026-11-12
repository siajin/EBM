"""DF extension of the previously QC-checked extractor; data stay on server.

Imports archived EBM modules without importing main.py or constructing its logger.
Stages: prepare, qc, extract. This script never trains the EBM.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time
from types import SimpleNamespace


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def settings(reference):
    archived = reference / "backup/scripts"
    args = runpy.run_path(str(archived / "arguments.py"))["get_args"]()[0]
    return archived, args


def prepare(root, reference):
    import numpy as np
    import pandas as pd

    archived, args = settings(reference)
    records = []
    protocol_dir = Path(args["path_19LA"]) / "LA/ASVspoof2019_LA_cm_protocols"
    protocols = {
        "T": protocol_dir / "ASVspoof2019.LA.cm.train.trn.txt",
        "dev": protocol_dir / "ASVspoof2019.LA.cm.dev.trl.txt",
    }
    for split, path in protocols.items():
        subset = "train" if split == "T" else "dev"
        for line in path.read_text().splitlines():
            columns = line.split()
            if len(columns) != 5 or columns[4] not in {"bonafide", "spoof"}:
                raise ValueError("Protocol format or label mismatch")
            speaker, uid, _, attack, label = columns
            audio_path = Path(args["path_19LA"]) / f"LA/ASVspoof2019_LA_{subset}/flac/{uid}.flac"
            if not audio_path.is_file():
                raise FileNotFoundError("Missing source audio; details remain on server")
            records.append({
                "dataset": "ASVspoof2019_LA", "split": split,
                "utterance_id": uid, "family_id": uid,
                "family_resolution": "utterance_only_no_source_derivation_map",
                "speaker_group": speaker, "dependency_group": speaker,
                "y": int(label == "bonafide"),
                "attack_id": "bonafide" if label == "bonafide" else attack,
                "path": str(audio_path), "condition_id": "clean", "crop_id": "native_first",
            })
    frame = pd.DataFrame(records)
    assert not frame.utterance_id.duplicated().any()
    speakers_t = set(frame.loc[frame.split == "T", "speaker_group"])
    speakers_dev = set(frame.loc[frame.split == "dev", "speaker_group"])
    assert not (speakers_t & speakers_dev)
    order = np.random.default_rng(11).permutation(sorted(speakers_dev))
    selection = set(order[: (len(order) + 1) // 2])
    frame.loc[frame.split == "dev", "split"] = frame.loc[frame.split == "dev", "speaker_group"].map(lambda s: "V" if s in selection else "Q")
    frame = frame.sort_values(["split", "utterance_id"], kind="stable").reset_index(drop=True)
    for split in ["T", "V", "Q"]:
        assert set(frame.loc[frame.split == split, "y"]) == {0, 1}
    manifest = root / "manifest.parquet"
    assert not manifest.exists(), "Do not overwrite an existing manifest"
    frame.to_parquet(manifest, index=False)
    safe = {
        "status": "pass", "manifest_sha256": digest(manifest),
        "protocol_sha256": {k: digest(v) for k, v in protocols.items()},
        "counts": frame.groupby(["split", "y"]).size().rename("count").reset_index().to_dict("records"),
        "speaker_counts": frame.groupby("split").speaker_group.nunique().to_dict(),
        "attack_counts": frame.groupby(["split", "attack_id"]).size().rename("count").reset_index().to_dict("records"),
        "duplicate_utterance_ids": 0, "missing_audio_paths": 0,
        "train_dev_speaker_overlap": 0,
        "unresolved_source_families": "Original-to-synthetic mapping is not present in the official five-column protocol; grouping uses known speaker IDs.",
        "feature_dimension": args["output_size"],
        "crop_samples": args["num_test_frames"] * 160 - 40,
        "extraction_splits": ["T", "V"], "Q_features_not_extracted": True,
        "checkpoint_sha256": digest(reference / "model/BestCLS_50_21DF.pt"),
        "archive_code_sha256": {str(p.relative_to(archived)): digest(p) for p in archived.rglob("*.py")},
        "script_sha256": digest(__file__),
    }
    save_json(root / "input_audit.json", safe)
    print(json.dumps({k: v for k, v in safe.items() if k != "archive_code_sha256"}, ensure_ascii=False), flush=True)


def utterance_seed(row, seed):
    key = "|".join(str(row[k]) for k in ["dataset", "utterance_id", "condition_id", "crop_id"])
    return int.from_bytes(hashlib.sha256((key + "|" + str(seed)).encode()).digest()[:8], "big") % (2**31 - 1)


def worker(rank, root_string, reference_string, stage, phase_name):
    import numpy as np
    import pandas as pd
    import soundfile as sf
    import torch
    from transformers import AutoConfig, Wav2Vec2Model

    root, reference = Path(root_string), Path(reference_string)
    phase = root / phase_name
    torch.set_num_threads(2)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    archived, args = settings(reference)
    sys.path.insert(0, str(archived))
    import model
    from data_loaders import EnrollmentSet

    audit = json.loads((root / "input_audit.json").read_text())
    frame = pd.read_parquet(root / "manifest.parquet")
    frame = frame.loc[frame.split.isin(audit["extraction_splits"])].reset_index(drop=True)
    if stage == "qc":
        # Runtime/format QC only; every DF pilot row is scored in extract.
        chosen = []
        for label, group in frame.groupby("y", sort=True):
            n = 32 if label == 1 else 96
            chosen.append(group.sample(n=min(n, len(group)), random_state=20261105))
        frame = pd.concat(chosen).sort_values("utterance_id").reset_index(drop=True)
    shared = frame.head(8).copy()
    shard = frame.iloc[rank::2].copy().reset_index(drop=True)
    config = AutoConfig.from_pretrained(args["PLM_name"], local_files_only=True)
    plm = Wav2Vec2Model.from_pretrained(args["PLM_name"], config=config, local_files_only=True).to(device).eval()
    classifier = model.B_Linear(
        num_layer=args["num_layers"], hidden_size=args["hidden_size"],
        output_size=args["output_size"], agg_size=args["agg_size"],
        loss=model.OCSoftmax(args["output_size"], r_real=args["r_real"], r_fake=args["r_fake"], alpha=args["alpha"]),
        loss_bpl=model.BPL(alpha=args["alpha_cl"]),
    ).to(device).eval()
    raw_state = torch.load(reference / "model/BestCLS_50_21DF.pt", map_location="cpu", weights_only=True)
    state = {key.removeprefix("module."): value for key, value in raw_state.items()}
    classifier.load_state_dict(state, strict=True)
    for parameter in list(plm.parameters()) + list(classifier.parameters()):
        parameter.requires_grad_(False)
    centroid = classifier.loss.weight.detach().cpu().numpy().reshape(-1).astype("float64")
    centroid /= np.linalg.norm(centroid)
    assert centroid.size == 160
    captured = {}
    def hook(_module, inputs, output):
        captured["h"] = inputs[0].detach().cpu().numpy().copy()[0]
        captured["z"] = output.detach().cpu().numpy().copy()[0]
    handle = classifier.layer_norm.register_forward_hook(hook)

    def infer(row):
        item = SimpleNamespace(path=row.path, type=row.attack_id, label=int(row.y))
        dataset = EnrollmentSet(args, [item])
        audio, _, label = dataset[0]
        info = sf.info(row.path)
        assert info.samplerate == 16000 and info.channels == 1
        assert len(audio) == args["num_test_frames"] * 160 - 40
        x = torch.as_tensor(audio, dtype=torch.float32, device=device).unsqueeze(0)
        seed = utterance_seed(row, 20261105)
        with torch.random.fork_rng(devices=[rank]), torch.inference_mode():
            torch.random.default_generator.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            hidden = plm(x, output_hidden_states=True).hidden_states
            _, native = classifier(torch.stack(hidden, dim=1))
            native_score = float(native.item())
        z, h = captured["z"].copy(), captured["h"].copy()
        assert np.isfinite(z).all() and np.isfinite(h).all() and np.isfinite(native_score)
        norm = float(np.linalg.norm(z.astype("float64")))
        assert norm > 1e-12
        unit_z = z.astype("float64") / norm
        cosine = float(unit_z @ centroid)
        assert abs(cosine - native_score) <= 1e-5
        residual = unit_z - cosine * centroid
        tangent = float(np.linalg.norm(residual))
        assert abs(tangent**2 - (1 - cosine**2)) <= 1e-10
        metadata = {key: row[key] for key in frame.columns if key != "path"}
        metadata.update(raw_score=native_score, cosine_s=cosine, raw_norm=norm, tangent_norm=tangent,
                        direction_valid=int(tangent >= 1e-4), extraction_seed=seed,
                        duration=info.duration, crop_start=0, crop_samples=len(audio),
                        padding_fraction=max(0, len(audio)-info.frames)/len(audio))
        return z, h, metadata

    # Same sample processed on both GPUs and twice on each GPU.
    repeats, shared_rows = [], []
    for _, row in shared.iterrows():
        first = infer(row)
        second = infer(row)
        error = float(np.max(np.abs(first[0] - second[0])))
        assert error <= 1e-5
        repeats.append(error)
        shared_rows.append({"id": row.utterance_id, "z": first[0].tolist(), "s": first[2]["cosine_s"]})
    save_json(phase / f"rank{rank}_reproducibility.json", {"maximum_repeat_z_error": max(repeats), "samples": shared_rows})
    n = len(shard)
    z_path, h_path = phase / f"rank{rank}_z.npy", phase / f"rank{rank}_h.npy"
    assert not z_path.exists() and not h_path.exists()
    zs = np.lib.format.open_memmap(z_path, mode="w+", dtype="float32", shape=(n, 160))
    hs = np.lib.format.open_memmap(h_path, mode="w+", dtype="float32", shape=(n, 160))
    metadata_rows = []
    start = time.monotonic()
    for i, (_, row) in enumerate(shard.iterrows()):
        z, h, metadata = infer(row)
        zs[i], hs[i] = z, h
        metadata.update(rank=rank, embedding_row=i)
        metadata_rows.append(metadata)
        if (i + 1) % 100 == 0 or i == n - 1:
            elapsed = time.monotonic() - start
            save_json(phase / f"rank{rank}_progress.json", {"completed": i + 1, "total": n, "elapsed_seconds": elapsed, "utterances_per_second": (i + 1) / elapsed})
            print(f"rank={rank} {i+1}/{n} seconds={elapsed:.1f}", flush=True)
        if (i + 1) % 500 == 0:
            zs.flush(); hs.flush()
    zs.flush(); hs.flush()
    pd.DataFrame(metadata_rows).to_parquet(phase / f"rank{rank}_rows.parquet", index=False)
    np.save(phase / f"rank{rank}_centroid.npy", centroid)
    save_json(phase / f"rank{rank}_model.json", {"gpu": torch.cuda.get_device_name(rank), "torch": torch.__version__, "frontend_commit": getattr(config, "_commit_hash", None), "strict_state_loading": True, "sample_count": n, "elapsed_seconds": time.monotonic()-start})
    handle.remove()


def finalize(root, phase_name, stage):
    import numpy as np
    import pandas as pd
    phase = root / phase_name
    r0 = json.loads((phase / "rank0_reproducibility.json").read_text())
    r1 = json.loads((phase / "rank1_reproducibility.json").read_text())
    assert [x["id"] for x in r0["samples"]] == [x["id"] for x in r1["samples"]]
    cross = max(float(np.max(np.abs(np.array(a["z"])-np.array(b["z"])))) for a,b in zip(r0["samples"],r1["samples"]))
    assert cross <= 1e-5
    frames = [pd.read_parquet(phase / f"rank{rank}_rows.parquet") for rank in [0,1]]
    rows = pd.concat(frames, ignore_index=True)
    assert not rows.utterance_id.duplicated().any()
    assert np.allclose(np.load(phase / "rank0_centroid.npy"), np.load(phase / "rank1_centroid.npy"), atol=1e-12)
    durations = [json.loads((phase / f"rank{rank}_model.json").read_text())["elapsed_seconds"] for rank in [0,1]]
    result = {
        "status": "pass", "stage": stage, "sample_count": len(rows),
        "max_cross_gpu_z_error": cross,
        "max_repeat_z_error": max(r0["maximum_repeat_z_error"], r1["maximum_repeat_z_error"]),
        "max_native_score_error": float(np.max(np.abs(rows.raw_score-rows.cosine_s))),
        "split_class_counts": rows.groupby(["split","y"]).size().rename("count").reset_index().to_dict("records"),
        "finite_embedding_check": True, "duplicate_utterance_ids": 0,
        "wall_seconds_excluding_model_load": max(durations),
        "aggregate_utterances_per_second": len(rows)/max(durations),
        "Q_evaluated": bool((rows.split == "Q").any()), "no_EBM_training": True,
    }
    if stage == "extract":
        expected = pd.read_parquet(root / "manifest.parquet")
        audit = json.loads((root / "input_audit.json").read_text())
        expected = expected.loc[expected.split.isin(audit["extraction_splits"])]
        assert set(rows.utterance_id) == set(expected.utterance_id)
    rows.to_parquet(phase / "scores.parquet", index=False)
    save_json(phase / "summary.json", result)
    (phase / "complete").write_text("0\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "qc", "extract"])
    parser.add_argument("--root", required=True)
    parser.add_argument("--phase")
    parser.add_argument("--reference", default="/home/mentee/results/CM/LA/EBM_e50_bs160_lr1e5")
    options = parser.parse_args()
    assert Path("/.dockerenv").exists(), "Must run in the existing container"
    root, reference = Path(options.root), Path(options.reference)
    if options.stage == "prepare":
        prepare(root, reference)
        return
    import torch
    import torch.multiprocessing as mp
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0,1"
    assert torch.cuda.device_count() == 2
    audit = json.loads((root / "input_audit.json").read_text())
    assert audit["status"] == "pass"
    assert digest(root / "manifest.parquet") == audit["manifest_sha256"]
    assert digest(reference / "model/BestCLS_50_21DF.pt") == audit["checkpoint_sha256"]
    phase_name = options.phase or options.stage
    phase = root / phase_name
    phase.mkdir(exist_ok=False)
    save_json(phase / "run.json", {"stage": options.stage, "script_sha256": digest(__file__), "manifest_sha256": audit["manifest_sha256"], "start_epoch": time.time()})
    mp.spawn(worker, args=(str(root), str(reference), options.stage, phase_name), nprocs=2, join=True)
    finalize(root, phase_name, options.stage)


if __name__ == "__main__":
    main()
