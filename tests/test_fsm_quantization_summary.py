import csv
import json

from scripts.fsm_quantization_summary import build_quantization_rows, write_rows


def _write_base_summary(path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["strategy", "compression_ratio", "ate_rmse", "rpe_rmse", "latency_mean_ms", "map_size_mb"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "strategy": "fps_scale_opacity",
                "compression_ratio": "0.1",
                "ate_rmse": "0.02",
                "rpe_rmse": "0.003",
                "latency_mean_ms": "40",
                "map_size_mb": "20",
            }
        )


def test_build_quantization_rows_compares_base_and_quantized(tmp_path):
    base = tmp_path / "summary_all.csv"
    _write_base_summary(base)
    qdir = tmp_path / "quant" / "fps_scale_opacity_ratio_0p1_fp16_drop_scores"
    (qdir / "localize").mkdir(parents=True)
    (qdir / "fsm_map.npz").write_bytes(b"0" * 10_000_000)
    (qdir / "localize" / "summary.json").write_text(
        json.dumps(
            {
                "ate_rmse": 0.021,
                "rpe_trans_rmse": 0.0031,
                "latency_mean_ms": 42,
                "num_fallbacks": 0,
            }
        )
    )

    rows = build_quantization_rows([["TUM", "fr2", str(base), str(tmp_path / "quant"), "fps_scale_opacity_ratio_0p1"]])

    assert len(rows) == 1
    row = rows[0]
    assert row["strategy"] == "fps_scale_opacity"
    assert row["compression_ratio"] == "0.1"
    assert row["ate_relative"] == "1.05"
    assert row["extra_size_reduction"] == "2"


def test_write_rows_creates_csv(tmp_path):
    output = tmp_path / "summary.csv"
    write_rows(output, [])
    assert output.exists()
    assert output.read_text().startswith("dataset,scene,method")
