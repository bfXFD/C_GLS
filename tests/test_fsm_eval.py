import csv
import os
import subprocess
import sys

import numpy as np

from scripts.fsm_eval import SUMMARY_ALL_FIELDS, TASK_PLAN_FIELDS, build_task_plan, run_eval, variant_name


def make_config_file(tmp_path):
    config_path = tmp_path / "fsm_config.py"
    config_path.write_text(
        "config = {\n"
        "    'seed': 7,\n"
        "    'dataset': {'dataset_name': 'fake', 'sequence': 'seq0'},\n"
        "    'registration': {\n"
        "        'backend': 'numpy_point_to_point',\n"
        "        'voxel_size': None,\n"
        "        'max_correspondence_distance': 0.1,\n"
        "        'init_mode': 'identity',\n"
        "    },\n"
        "    'fsm': {\n"
        "        'strategies': ['full', 'random'],\n"
        "        'compression_ratios': [1.0, 0.5],\n"
        "    },\n"
        "}\n"
    )
    return config_path


def make_params_file(tmp_path):
    params_path = tmp_path / "run0" / "params.npz"
    params_path.parent.mkdir(parents=True)
    points = np.zeros((4, 3), dtype=np.float32)
    np.savez(
        params_path,
        means3D=points,
        log_scales=np.zeros((4, 1), dtype=np.float32),
        unnorm_rotations=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (4, 1)),
    )
    return params_path


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_variant_name_formats_ratio():
    assert variant_name("full", 1.0) == "full_ratio_1"
    assert variant_name("random", 0.05) == "random_ratio_0p05"


def test_build_task_plan_schema_and_paths(tmp_path):
    rows = build_task_plan(str(tmp_path), ["full", "random"], [1.0, 0.5])
    assert len(rows) == 4
    assert set(rows[0].keys()) == set(TASK_PLAN_FIELDS)
    assert rows[0]["map_path"].endswith("maps/full_ratio_1/fsm_map.npz")
    assert rows[1]["localize_dir"].endswith("localize/full_ratio_0p5")


def test_run_eval_dry_run_writes_plan_and_summary_schema(tmp_path):
    config_path = make_config_file(tmp_path)
    params_path = make_params_file(tmp_path)
    output_dir = tmp_path / "eval"
    result = run_eval(str(config_path), str(params_path), str(output_dir), dry_run=True)

    assert result["dry_run"] is True
    assert len(result["tasks"]) == 4
    assert len(result["summary_rows"]) == 4
    assert (output_dir / "task_plan.csv").exists()
    assert (output_dir / "summary_all.csv").exists()

    task_rows = read_csv(output_dir / "task_plan.csv")
    summary_rows = read_csv(output_dir / "summary_all.csv")
    assert set(task_rows[0].keys()) == set(TASK_PLAN_FIELDS)
    assert set(summary_rows[0].keys()) == set(SUMMARY_ALL_FIELDS)
    assert summary_rows[0]["dataset"] == "fake"
    assert summary_rows[0]["sequence"] == "seq0"
    assert summary_rows[0]["run_name"] == "run0"
    assert summary_rows[0]["backend"] == "numpy_point_to_point"
    assert summary_rows[0]["seed"] == "7"
    assert "score_source" in summary_rows[0]


def test_run_eval_dry_run_cli_overrides_strategies_and_ratios(tmp_path):
    config_path = make_config_file(tmp_path)
    params_path = make_params_file(tmp_path)
    output_dir = tmp_path / "eval"
    result = run_eval(
        str(config_path),
        str(params_path),
        str(output_dir),
        strategies=["opacity"],
        compression_ratios=[0.25],
        dry_run=True,
    )
    assert len(result["tasks"]) == 1
    assert result["tasks"][0]["strategy"] == "opacity"
    assert result["tasks"][0]["compression_ratio"] == "0.25"


def test_run_eval_dry_run_accepts_external_score_args(tmp_path):
    config_path = make_config_file(tmp_path)
    params_path = make_params_file(tmp_path)
    scores_path = tmp_path / "scores.npz"
    np.savez(scores_path, scores=np.arange(4, dtype=np.float32))
    result = run_eval(
        str(config_path),
        str(params_path),
        str(tmp_path / "eval"),
        strategies=["fisher_score"],
        compression_ratios=[0.5],
        dry_run=True,
        external_scores_path=str(scores_path),
    )
    assert len(result["tasks"]) == 1


def test_cli_dry_run_smoke(tmp_path):
    config_path = make_config_file(tmp_path)
    params_path = make_params_file(tmp_path)
    output_dir = tmp_path / "cli_eval"
    cmd = [
        sys.executable,
        "scripts/fsm_eval.py",
        "--config",
        str(config_path),
        "--params",
        str(params_path),
        "--output-dir",
        str(output_dir),
        "--strategies",
        "full",
        "--compression-ratios",
        "1.0",
        "--dry-run",
    ]
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(__file__)), check=True, text=True, capture_output=True)
    assert "FSM eval planned: 1 tasks" in result.stdout
    assert (output_dir / "task_plan.csv").exists()
    assert (output_dir / "summary_all.csv").exists()
