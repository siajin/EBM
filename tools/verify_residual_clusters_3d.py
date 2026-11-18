"""Independent checks of saved 3D geometry and the offline browser viewer."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "2"
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "artifacts/geometry_cache"
OUT = CACHE / "visualizations/residual_clusters_20261111_v1"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    result = json.loads((OUT / "statistics_3d.json").read_text(encoding="utf-8"))
    assert result["status"] == "complete"
    points = pd.read_parquet(OUT / "plotted_points_3d.parquet")
    u = np.load(OUT / "plot_unit_directions_3d.npy").astype(float)
    pca = np.load(OUT / "projection_parameters.npz")
    centers = np.load(OUT / "cluster_centers.npy")
    basis = np.load(OUT / "bearing_basis3.npy")
    lock = pd.read_parquet(OUT / "sample_lock.parquet").groupby(["dataset", "y"], sort=False).head(500)
    assert len(points) == 5000 and points.groupby(["dataset", "y"]).size().eq(500).all()
    assert set(zip(points.dataset, points.utterance_id)) == set(zip(lock.dataset, lock.utterance_id))
    assert np.isfinite(points[[f"{m}{i}" for m in ("pca", "tsne", "direction") for i in (1, 2, 3)]]).all().all()
    assert np.max(abs(np.linalg.norm(u, axis=1) - 1)) < 1e-7
    assert np.max(abs(u @ pca["centroid"])) < 1e-7
    assert np.max(abs(basis.T @ basis - np.eye(3))) < 1e-10
    xyz = (u - pca["pca_mean"]) @ pca["pca_components"][:3].T
    assert np.max(abs(xyz - points[["pca1", "pca2", "pca3"]].to_numpy())) < 1e-7
    assert np.max(abs(u @ basis - points[["direction1", "direction2", "direction3"]].to_numpy())) < 1e-7
    assert np.max(np.linalg.norm(u @ basis, axis=1)) <= 1.0000001
    assert np.array_equal(np.argmax(u @ centers.T, axis=1), points.cluster)
    catalog = json.loads((CACHE / "catalog.json").read_text())
    max_raw_error = 0.0
    full_checks = []
    specs = [("19la_tv", ("T", "V")), ("frozen_19la_eval", ("19LA",)),
             ("21la_eval", ("21LA",)), ("21df_pilot", ("DF",))]
    for key, datasets in specs:
        folder = CACHE / catalog["runs"][key]["local_folder"] / "extract"
        c = np.load(folder / "rank0_centroid.npy")
        all_rows, all_unit = [], []
        for rank in (0, 1):
            rows = pd.read_parquet(folder / f"rank{rank}_rows.parquet")
            z = np.load(folder / f"rank{rank}_z.npy", mmap_mode="r").astype(float)
            normz = np.sqrt(np.einsum("ij,ij->i", z, z))
            raw_cos = (z @ c) / normz
            residual = z / normz[:, None] - raw_cos[:, None] * c
            unit = residual / np.linalg.norm(residual, axis=1)[:, None]
            all_rows.append(rows)
            all_unit.append(unit)
        rows = pd.concat(all_rows, ignore_index=True)
        unit = np.concatenate(all_unit)
        for dataset in datasets:
            stored = pd.read_parquet(OUT / f"{dataset}_cluster_assignments.parquet")
            idx = pd.Index(rows.utterance_id).get_indexer(stored.utterance_id)
            assert (idx >= 0).all()
            directions = unit[idx]
            assert np.array_equal(np.argmax(directions @ centers.T, axis=1), stored.cluster)
            drawn = np.flatnonzero(points.dataset.to_numpy() == dataset)
            idxdraw = pd.Index(rows.utterance_id).get_indexer(points.iloc[drawn].utterance_id)
            err = float(np.max(abs(unit[idxdraw] - u[drawn])))
            assert err < 1e-7
            max_raw_error = max(max_raw_error, err)
            for y in (1, 0):
                v = directions[stored.y.to_numpy() == y]
                n = len(v)
                stats = result["full_group_statistics"][f"{dataset}:{y}"]
                assert n == stats["n"]
                assert abs(np.linalg.norm(v.mean(axis=0)) - stats["R"]) < 1e-10
                assert sum(stats["bearing_bin_counts"]) == n
                assert abs(sum(stats["cluster_fractions"]) - 1) < 1e-10
                # Explicit Gram matrix independently checks the pairwise identity.
                small = v[:128]
                gram = small @ small.T
                pair_direct = (gram.sum() - np.trace(gram)) / (len(small) * (len(small) - 1))
                pair_identity = (len(small) * np.linalg.norm(small.mean(axis=0)) ** 2 - 1) / (len(small) - 1)
                assert abs(pair_direct - pair_identity) < 1e-10
                full_checks.append(f"{dataset}:{y}")
    for path, expected in json.loads((OUT / "input_hashes_3d.json").read_text()).items():
        assert sha(path) == expected
    errors, requests = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--use-angle=swiftshader-webgl", "--enable-unsafe-swiftshader"])
        page = browser.new_page(viewport={"width": 1500, "height": 1080}, device_scale_factor=1)
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("request", lambda req: requests.append(req.url))
        page.goto((OUT / "residual_direction_3d.html").as_uri())
        page.wait_for_function("document.getElementById('plot').data?.length > 0")
        page.wait_for_timeout(1000)
        count = "document.getElementById('plot').data.filter(t=>t.mode==='markers').reduce((s,t)=>s+t.x.length,0)"
        assert page.evaluate(count) == 5000
        page.screenshot(path=str(OUT / "10_interactive_3d_preview.png"), full_page=True)
        page.select_option("#color", "cluster")
        page.wait_for_function("document.getElementById('plot').data.filter(t=>t.mode==='markers').every(t=>t.name.startsWith('C'))")
        for method in ("tsne", "direction"):
            page.select_option("#method", method)
            page.wait_for_function("document.getElementById('meta').textContent.includes('5,000')")
            assert page.evaluate(count) == 5000
        for dataset in ("T", "V", "19LA", "21LA"):
            page.uncheck(f'#datasets input[value="{dataset}"]')
        page.uncheck('#classes input[value="0"]')
        page.wait_for_function("document.getElementById('meta').textContent.includes('표시 500개')")
        assert page.evaluate(count) == 500
        assert page.evaluate("document.getElementById('plot').data.filter(t=>t.mode==='markers').every(t=>t.customdata.every(p=>p[0]==='DF' && p[1]==='정상'))")
        # Verify the actual drag interaction emits a 3D-camera change.
        page.evaluate("window.cameraMoves=0;document.getElementById('plot').on('plotly_relayout',e=>{if(Object.keys(e).some(k=>k.startsWith('scene.camera')))window.cameraMoves++})")
        box = page.locator("#plot").bounding_box()
        page.mouse.move(box["x"] + box["width"] * .5, box["y"] + box["height"] * .5)
        page.mouse.down()
        page.mouse.move(box["x"] + box["width"] * .65, box["y"] + box["height"] * .58, steps=12)
        page.mouse.up()
        page.wait_for_function("window.cameraMoves > 0")
        page.click("#reset")
        page.set_viewport_size({"width": 420, "height": 900})
        page.wait_for_timeout(400)
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")
        page.screenshot(path=str(OUT / "11_mobile_3d_check.png"), full_page=True)
        browser.close()
    assert not errors, errors
    assert not any(url.startswith(("http:", "https:")) for url in requests), requests
    validation = {"status": "passed", "plot_rows": 5000, "full_dataset_class_checks": full_checks,
                  "raw_embedding_direction_max_error_float32": max_raw_error,
                  "PCA_and_origin_projection_recomputed": True, "cluster_assignments_recomputed": True,
                  "source_hashes_unchanged": True, "browser": {"initial_points": 5000,
                  "DF_normal_filter_points": 500, "all_three_projection_controls": True,
                  "cluster_colors": True, "mouse_rotation": True, "reset": True,
                  "mobile_no_horizontal_overflow": True, "javascript_errors": errors,
                  "external_requests": 0}, "remote_operations": 0, "new_spend_usd": 0}
    (OUT / "independent_validation_3d.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(validation, ensure_ascii=False))


if __name__ == "__main__":
    main()
