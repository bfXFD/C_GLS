import csv
import json

from scripts.fsm_decoupling_summary import build_decoupling_rows, write_rows


def test_build_decoupling_rows_merges_render_and_localization(tmp_path):
    localization_path = tmp_path / "summary_all.csv"
    with open(localization_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "strategy",
                "compression_ratio",
                "ate_rmse",
                "rpe_rmse",
                "fitness",
                "registration_rmse",
                "latency_mean_ms",
                "map_size_mb",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "strategy": "voxel_scale_opacity",
                "compression_ratio": "0.1",
                "ate_rmse": "0.02",
                "rpe_rmse": "0.004",
                "fitness": "0.99",
                "registration_rmse": "0.03",
                "latency_mean_ms": "100",
                "map_size_mb": "50",
            }
        )

    render_path = tmp_path / "render_summary.json"
    render_path.write_text(
        json.dumps(
            {
                "psnr_mean": 18.5,
                "ms_ssim_mean": 0.71,
                "depth_l1_mean": 0.12,
                "num_render_gaussians": 10,
                "num_eval_frames": 5,
            }
        )
    )

    rows = build_decoupling_rows(
        scene="room0_0",
        localization_summary=str(localization_path),
        entries=[["C-GLS-Voxel", "voxel_scale_opacity", "0.1", str(render_path)]],
    )

    assert rows == [
        {
            "scene": "room0_0",
            "method": "C-GLS-Voxel",
            "ratio": "0.1",
            "localization_strategy": "voxel_scale_opacity",
            "localization_ratio": "0.1",
            "ate_rmse": "0.02",
            "rpe_trans_rmse": "0.004",
            "fitness": "0.99",
            "registration_rmse": "0.03",
            "latency_mean_ms": "100",
            "map_size_mb": "50",
            "psnr_mean": 18.5,
            "ms_ssim_mean": 0.71,
            "depth_l1_mean": 0.12,
            "num_render_gaussians": 10,
            "num_eval_frames": 5,
            "render_summary_path": str(render_path),
        }
    ]


def test_write_rows_creates_csv(tmp_path):
    output_path = tmp_path / "decoupling.csv"
    write_rows(output_path, [])
    assert output_path.exists()
    assert output_path.read_text().startswith("scene,method,ratio")
