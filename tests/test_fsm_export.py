import csv
import os
import subprocess
import sys

import numpy as np
import pytest

from scripts.fsm_export import export_fsm_maps, load_external_scores
from utils.fsm_map import load_fsm_map, validate_fsm_map


def make_params_file(tmp_path, num_gaussians=6):
    params_path = tmp_path / "params.npz"
    means = np.arange(num_gaussians * 3, dtype=np.float32).reshape(num_gaussians, 3) * 0.1
    log_scales = np.log(np.linspace(0.01, 0.06, num_gaussians, dtype=np.float32)[:, None])
    rotations = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (num_gaussians, 1))
    opacities = np.linspace(-3.0, 3.0, num_gaussians, dtype=np.float32)[:, None]
    np.savez(
        params_path,
        means3D=means,
        log_scales=log_scales,
        unnorm_rotations=rotations,
        logit_opacities=opacities,
        rgb_colors=np.zeros((num_gaussians, 3), dtype=np.float32),
    )
    return params_path


def read_summary(summary_path):
    with open(summary_path, newline="") as f:
        return list(csv.DictReader(f))


def test_export_fsm_maps_writes_maps_and_summary(tmp_path):
    params_path = make_params_file(tmp_path)
    output_dir = tmp_path / "maps"
    rows = export_fsm_maps(
        params_path=str(params_path),
        output_dir=str(output_dir),
        strategies=["full", "opacity"],
        compression_ratios=[0.5, 1.0],
        seed=3,
    )

    assert len(rows) == 4
    summary_rows = read_summary(output_dir / "export_summary.csv")
    assert len(summary_rows) == 4
    assert set(summary_rows[0].keys()) == {
        "strategy",
        "compression_ratio",
        "count",
        "N_full",
        "N_star",
        "map_path",
        "source_params_path",
        "seed",
    }

    for row in summary_rows:
        assert os.path.exists(row["map_path"])
        fsm_map = load_fsm_map(row["map_path"])
        validate_fsm_map(fsm_map)
        assert int(row["N_full"]) == 6
        assert int(row["N_star"]) == fsm_map["means"].shape[0]


def test_export_with_counts(tmp_path):
    params_path = make_params_file(tmp_path)
    output_dir = tmp_path / "maps"
    rows = export_fsm_maps(
        params_path=str(params_path),
        output_dir=str(output_dir),
        strategies=["random"],
        counts=[2, 4],
        seed=10,
    )
    assert [row["N_star"] for row in rows] == ["2", "4"]
    assert (output_dir / "random_count_2" / "fsm_map.npz").exists()
    assert (output_dir / "random_count_4" / "fsm_map.npz").exists()


def test_export_external_scores(tmp_path):
    params_path = make_params_file(tmp_path)
    output_dir = tmp_path / "maps"
    scores = np.array([0.0, 9.0, 2.0, 8.0, 1.0, 3.0], dtype=np.float32)
    rows = export_fsm_maps(
        params_path=str(params_path),
        output_dir=str(output_dir),
        strategies=["external_scores"],
        counts=[2],
        external_scores=scores,
    )
    fsm_map = load_fsm_map(rows[0]["map_path"])
    np.testing.assert_array_equal(fsm_map["indices"], np.array([1, 3]))


def test_export_fisher_hybrid_strategy_uses_external_scores(tmp_path):
    params_path = make_params_file(tmp_path)
    output_dir = tmp_path / "maps"
    fisher_scores = np.array([1.0, 6.0, 2.0, 5.0, 3.0, 4.0], dtype=np.float32)
    rows = export_fsm_maps(
        params_path=str(params_path),
        output_dir=str(output_dir),
        strategies=["fisher_score", "fisher_x_scale"],
        counts=[2],
        external_scores=fisher_scores,
    )
    assert len(rows) == 2
    fisher_map = load_fsm_map(rows[0]["map_path"])
    np.testing.assert_array_equal(fisher_map["indices"], np.array([1, 3]))
    assert (output_dir / "fisher_x_scale_count_2" / "fsm_map.npz").exists()


def test_load_external_scores_from_npz_and_npy(tmp_path):
    scores = np.array([1.0, 2.0, 3.0])
    npz_path = tmp_path / "scores.npz"
    npy_path = tmp_path / "scores.npy"
    np.savez(npz_path, scores=scores)
    np.save(npy_path, scores)
    np.testing.assert_allclose(load_external_scores(str(npz_path)), scores)
    np.testing.assert_allclose(load_external_scores(str(npy_path)), scores)


def test_export_rejects_existing_without_overwrite(tmp_path):
    params_path = make_params_file(tmp_path)
    output_dir = tmp_path / "maps"
    export_fsm_maps(str(params_path), str(output_dir), strategies=["full"], compression_ratios=[1.0])
    with pytest.raises(FileExistsError):
        export_fsm_maps(str(params_path), str(output_dir), strategies=["full"], compression_ratios=[1.0])


def test_export_overwrite_replaces_existing(tmp_path):
    params_path = make_params_file(tmp_path)
    output_dir = tmp_path / "maps"
    export_fsm_maps(str(params_path), str(output_dir), strategies=["full"], compression_ratios=[1.0])
    marker = output_dir / "full_ratio_1" / "marker.txt"
    marker.write_text("old")
    export_fsm_maps(str(params_path), str(output_dir), strategies=["full"], compression_ratios=[1.0], overwrite=True)
    assert not marker.exists()
    assert (output_dir / "full_ratio_1" / "fsm_map.npz").exists()


def test_export_rejects_counts_and_ratios_together(tmp_path):
    params_path = make_params_file(tmp_path)
    with pytest.raises(ValueError):
        export_fsm_maps(
            str(params_path),
            str(tmp_path / "maps"),
            strategies=["full"],
            compression_ratios=[1.0],
            counts=[2],
        )


def test_cli_smoke_exports_maps(tmp_path):
    params_path = make_params_file(tmp_path)
    output_dir = tmp_path / "cli_maps"
    cmd = [
        sys.executable,
        "scripts/fsm_export.py",
        "--params",
        str(params_path),
        "--output-dir",
        str(output_dir),
        "--strategies",
        "full",
        "opacity",
        "--compression-ratios",
        "0.5",
        "--seed",
        "0",
    ]
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(__file__)), check=True, text=True, capture_output=True)
    assert "Exported 2 FSM map variants" in result.stdout
    assert (output_dir / "export_summary.csv").exists()
    assert (output_dir / "full_ratio_0p5" / "fsm_map.npz").exists()
    assert (output_dir / "opacity_ratio_0p5" / "fsm_map.npz").exists()
