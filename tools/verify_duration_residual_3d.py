"""Verify duration analysis, preserved coordinates, and the updated 3D viewer."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "2"
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits
from playwright.sync_api import sync_playwright
import visualize_residual_clusters as shared

ROOT, CACHE = shared.ROOT, shared.CACHE
OLD = shared.OUT
OUT = CACHE / "visualizations/residual_duration_20261111_v1"


def records(html):
    return json.loads(html.split("const points=", 1)[1].split(";\nconst metadata=", 1)[0])


def main():
    result = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    old = pd.read_parquet(OLD / "plotted_points_3d.parquet")
    new = pd.read_parquet(OUT / "plotted_points_duration.parquet")
    pd.testing.assert_frame_equal(old, new[old.columns])
    u = np.load(OLD / "plot_unit_directions_3d.npy").astype(float)
    probe = np.load(OUT / "duration_probe.npz")
    assert np.max(abs(u @ probe["basis3"] - new[["length1", "length2", "length3"]].to_numpy())) < 1e-12
    assert np.max(abs(probe["basis3"].T @ probe["basis3"] - np.eye(3))) < 1e-10
    for fold in result["train_probe"]["folds"]:
        assert set(fold["train_speakers"]).isdisjoint(fold["test_speakers"])
    independent_nested_rho = float(np.corrcoef(rankdata(probe["train_target_log2duration"]), rankdata(probe["train_oof_prediction"]))[0, 1])
    assert abs(independent_nested_rho - result["train_probe"]["nested_CV_spearman"]) < 1e-12
    # Independent original-z reconstruction of two complete non-Train populations.
    catalog = json.loads((CACHE / "catalog.json").read_text())
    direct_errors = []
    for key, dataset in [("19la_tv", "V"), ("21df_pilot", "DF")]:
        folder = CACHE / catalog["runs"][key]["local_folder"] / "extract"
        frames, directions = [], []
        for rank in (0, 1):
            frame = pd.read_parquet(folder / f"rank{rank}_rows.parquet")
            select = frame.split.to_numpy() == "V" if dataset == "V" else np.ones(len(frame), dtype=bool)
            z = np.load(folder / f"rank{rank}_z.npy", mmap_mode="r")[select].astype(float)
            x = z / np.linalg.norm(z, axis=1, keepdims=True)
            r = x - (x @ probe["centroid"])[:, None] * probe["centroid"]
            directions.append(r / np.linalg.norm(r, axis=1, keepdims=True))
            frames.append(frame.loc[select])
        frame = pd.concat(frames, ignore_index=True)
        axis_coordinate = np.concatenate(directions) @ probe["unit_axis"]
        for label in (1, 0):
            mask = frame.y.to_numpy() == label
            direct = float(np.corrcoef(rankdata(frame.duration.to_numpy()[mask]), rankdata(axis_coordinate[mask]))[0, 1])
            saved = next(row for row in result["associations"] if row["dataset"] == dataset and row["class"] == ("normal" if label else "spoof"))
            error = abs(direct - saved["length_axis_spearman"])
            assert error < 1e-12
            direct_errors.append(error)
    viewer = OUT / "residual_direction_duration_3d.html"
    html = viewer.read_text(encoding="utf-8")
    old_points = records((OUT / "original_residual_direction_3d.html").read_text(encoding="utf-8"))
    new_points = records(html)
    for before, after in zip(old_points, new_points):
        assert all(after[key] == value for key, value in before.items())
    errors, requests = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--use-angle=swiftshader-webgl", "--enable-unsafe-swiftshader"])
        page = browser.new_page(viewport={"width": 1500, "height": 1150}, device_scale_factor=1)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("request", lambda request: requests.append(request.url))
        page.goto(viewer.as_uri())
        page.wait_for_function("document.getElementById('plot').data?.length > 0")
        assert page.locator("#color").input_value() == "duration"
        assert page.evaluate("document.getElementById('plot').layout.coloraxis.cmin") == -1
        assert page.evaluate("document.getElementById('plot').layout.coloraxis.cmax") == 4
        assert page.evaluate("document.getElementById('plot').data.filter(t=>t.mode==='markers').every(t=>t.marker.coloraxis==='coloraxis'&&t.marker.color.length===t.x.length)")
        # Preserve filters; inspect the same DF normal cloud on the original PCA.
        for dataset in ("T", "V", "19LA", "21LA"):
            page.uncheck(f'#datasets input[value="{dataset}"]')
        page.uncheck('#classes input[value="0"]')
        page.wait_for_function("document.getElementById('meta').textContent.includes('표시 500개')")
        page.wait_for_timeout(700)
        page.screenshot(path=str(OUT / "duration_3d_preview.png"), full_page=True)
        assert page.evaluate("document.getElementById('plot').layout.coloraxis.cmin") == -1
        page.select_option("#durationScale", "linear")
        assert page.evaluate("document.getElementById('plot').layout.coloraxis.cmin") == .5
        assert page.evaluate("document.getElementById('plot').data.filter(t=>t.mode==='markers').every(t=>t.marker.color.every((v,i)=>Math.abs(v-t.customdata[i][4])<1e-9))")
        page.select_option("#durationScale", "log")
        assert page.evaluate("document.getElementById('plot').data.filter(t=>t.mode==='markers').every(t=>t.marker.color.every((v,i)=>Math.abs(v-Math.log2(t.customdata[i][4]))<1e-9))")
        for method in ("tsne", "direction", "length", "pca"):
            page.select_option("#method", method)
            assert page.evaluate("document.getElementById('plot').data.filter(t=>t.mode==='markers').reduce((s,t)=>s+t.x.length,0)") == 500
        for coloring in ("class", "dataset", "cluster", "duration"):
            page.select_option("#color", coloring)
        page.select_option("#method", "length")
        page.wait_for_timeout(500)
        page.screenshot(path=str(OUT / "length_axis_3d_preview.png"), full_page=True)
        page.evaluate("window.moves=0;document.getElementById('plot').on('plotly_relayout',e=>{if(Object.keys(e).some(k=>k.startsWith('scene.camera')))window.moves++})")
        box = page.locator("#plot").bounding_box()
        page.mouse.move(box["x"] + box["width"] * .45, box["y"] + box["height"] * .5)
        page.mouse.down()
        page.mouse.move(box["x"] + box["width"] * .60, box["y"] + box["height"] * .6, steps=10)
        page.mouse.up()
        page.wait_for_function("window.moves>0")
        page.click("#reset")
        page.set_viewport_size({"width": 420, "height": 900})
        page.wait_for_timeout(300)
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
        page.screenshot(path=str(OUT / "duration_mobile_check.png"), full_page=True)
        browser.close()
    assert not errors, errors
    assert not any(url.startswith(("http:", "https:")) for url in requests)
    for path, expected in json.loads((OUT / "input_hashes.json").read_text()).items():
        assert shared.digest(path) == expected
    authorized_path = OUT / "authorized_visualization_source_edits.json"
    authorized = json.loads(authorized_path.read_text()) if authorized_path.exists() else {}
    for path, expected in json.loads((OUT / "existing_sources_before.json").read_text()).items():
        actual = shared.digest(path)
        if actual != expected:
            assert path in authorized
            assert authorized[path]["before"] == expected and authorized[path]["after"] == actual
    # Only after validation, publish to the user's existing local viewer URL.
    destination = OLD / "residual_direction_3d.html"
    pending = destination.with_suffix(".duration-pending.html")
    shutil.copyfile(viewer, pending)
    pending.replace(destination)
    assert shared.digest(destination) == shared.digest(viewer)
    validation = {"status": "passed", "original_UIDs_labels_metadata_and_3D_coordinates_preserved": True,
                  "original_embedded_JSON_values_preserved_exactly": True, "new_axis_projection_recomputed": True,
                  "nested_CV_independent_rank_correlation": independent_nested_rho,
                  "independent_V_DF_full_group_rho_max_error": max(direct_errors),
                  "browser": {"continuous_duration_colors": True, "common_scale_across_filters": True,
                              "linear_and_log_color_values_verified": True, "four_coordinate_modes": True,
                              "all_old_color_modes_preserved": True, "mouse_rotation": True,
                              "mobile_no_horizontal_overflow": True, "javascript_errors": [], "external_requests": 0},
                  "updated_existing_viewer": str(destination), "training_modified": False, "new_spend_usd": 0}
    (OUT / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    result["status"] = "complete_and_verified"
    result["updated_existing_viewer"] = str(destination)
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(validation, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
