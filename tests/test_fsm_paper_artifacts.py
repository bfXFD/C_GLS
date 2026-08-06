import csv

from scripts.fsm_paper_artifacts import build_ablation_table, build_replica_main_table, build_tum_table, generate_artifacts


def test_build_replica_main_table_selects_key_operating_points():
    rows = [
        {
            "scene": "room0",
            "strategy": "full",
            "compression_ratio": "1",
            "ate_rmse": "0.01",
            "ate_over_full": "1",
            "rpe_rmse": "0.002",
            "speedup_over_full": "1",
            "size_reduction_over_full": "1",
        },
        {
            "scene": "room0",
            "strategy": "voxel_scale_opacity",
            "compression_ratio": "0.2",
            "ate_rmse": "0.011",
            "ate_over_full": "1.1",
            "rpe_rmse": "0.003",
            "speedup_over_full": "4",
            "size_reduction_over_full": "4.2",
        },
    ]

    table = build_replica_main_table(rows)

    assert len(table) == 2
    assert table[1]["strategy"] == "voxel_scale_opacity"
    assert table[1]["mean_speedup_over_full"] == "4"


def test_build_ablation_table_keeps_coverage_and_saliency_methods():
    rows = [
        {"strategy": "random", "compression_ratio": "0.1", "ate_over_full": "2", "speedup_over_full": "7", "size_reduction_over_full": "7"},
        {"strategy": "scale_large", "compression_ratio": "0.1", "ate_over_full": "1.5", "speedup_over_full": "7", "size_reduction_over_full": "7"},
    ]

    table = build_ablation_table(rows)

    assert [row["strategy"] for row in table] == ["random", "scale_large"]


def test_build_tum_table_keeps_full_and_0p1_methods():
    rows = [
        {"scene": "fr1", "strategy": "full", "compression_ratio": "1", "ate_rmse": "0.1"},
        {"scene": "fr1", "strategy": "fps_scale_opacity", "compression_ratio": "0.1", "ate_rmse": "0.05"},
        {"scene": "fr1", "strategy": "fps_scale_opacity", "compression_ratio": "0.2", "ate_rmse": "0.06"},
    ]

    table = build_tum_table(rows)

    assert len(table) == 2
    assert table[1]["compression_ratio"] == "0.1"


def _write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_generate_artifacts_writes_tables(tmp_path):
    replica = tmp_path / "replica.csv"
    tum = tmp_path / "tum.csv"
    fields = ["scene", "strategy", "compression_ratio", "ate_rmse", "ate_over_full", "rpe_rmse", "speedup_over_full", "size_reduction_over_full", "map_size_mb"]
    _write_csv(replica, fields, [{"scene": "room0", "strategy": "full", "compression_ratio": "1", "ate_rmse": "0.01", "ate_over_full": "1", "rpe_rmse": "0.002", "speedup_over_full": "1", "size_reduction_over_full": "1", "map_size_mb": "50"}])
    _write_csv(tum, fields, [{"scene": "fr1", "strategy": "full", "compression_ratio": "1", "ate_rmse": "0.1", "ate_over_full": "1", "rpe_rmse": "0.01", "speedup_over_full": "1", "size_reduction_over_full": "1", "map_size_mb": "30"}])

    outputs = generate_artifacts(str(replica), str(tum), str(tmp_path / "missing.csv"), str(tmp_path / "out"), make_plots=False)

    assert outputs["replica_main"].endswith("table_replica_main.csv")
    assert (tmp_path / "out" / "table_replica_main.md").exists()
