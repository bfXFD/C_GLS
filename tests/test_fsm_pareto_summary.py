import csv

from scripts.fsm_pareto_summary import build_pareto_rows, write_rows


def _write_summary(path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_name",
                "strategy",
                "compression_ratio",
                "ate_rmse",
                "rpe_rmse",
                "latency_mean_ms",
                "map_size_mb",
                "fitness",
                "registration_rmse",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "run_name": "full_ratio_1",
                "strategy": "full",
                "compression_ratio": "1",
                "ate_rmse": "0.01",
                "rpe_rmse": "0.002",
                "latency_mean_ms": "100",
                "map_size_mb": "50",
                "fitness": "1",
                "registration_rmse": "0.01",
            }
        )
        writer.writerow(
            {
                "run_name": "voxel_ratio_0p1",
                "strategy": "voxel_grid_pure",
                "compression_ratio": "0.1",
                "ate_rmse": "0.02",
                "rpe_rmse": "0.003",
                "latency_mean_ms": "25",
                "map_size_mb": "10",
                "fitness": "0.9",
                "registration_rmse": "0.02",
            }
        )


def test_build_pareto_rows_adds_relative_metrics(tmp_path):
    summary = tmp_path / "summary_all.csv"
    _write_summary(summary)

    rows = build_pareto_rows([["room", str(summary)]])

    assert len(rows) == 2
    compressed = rows[1]
    assert compressed["scene"] == "room"
    assert compressed["ate_over_full"] == "2"
    assert compressed["speedup_over_full"] == "4"
    assert compressed["size_reduction_over_full"] == "5"


def test_write_rows_creates_pareto_csv(tmp_path):
    output = tmp_path / "pareto.csv"
    write_rows(output, [])
    assert output.exists()
    assert output.read_text().startswith("scene,run_name,strategy")
