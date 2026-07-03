import numpy as np

from scripts.fsm_skeleton_analysis import _coverage_stats, _nearest_distance_stats, _trajectory_distance_stats


def test_coverage_stats_counts_voxels_and_entropy():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    stats = _coverage_stats(points, voxel_size=0.5)

    assert stats["occupied_voxels"] == 2
    assert stats["coverage_entropy"] > 0


def test_nearest_distance_stats_uses_nearest_neighbor():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float64)

    stats = _nearest_distance_stats(points)

    assert np.isclose(stats["median_nn_distance"], 1.0)
    assert stats["mean_nn_distance"] > 0


def test_trajectory_distance_stats_uses_nearest_pose():
    points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
    trajectory = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)

    stats = _trajectory_distance_stats(points, trajectory)

    assert np.isclose(stats["trajectory_distance_mean"], 0.5)
    assert np.isclose(stats["trajectory_distance_median"], 0.5)
