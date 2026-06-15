import numpy as np
import pytest

from utils.fsm_dataset import backproject_depth, get_frame, rgbd_to_pointcloud


def intrinsics():
    return np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def test_backproject_depth_constant_depth():
    depth = np.ones((2, 2), dtype=np.float32)
    points = backproject_depth(depth, intrinsics())
    expected = np.array(
        [
            [-1.0, -1.0, 1.0],
            [0.0, -1.0, 1.0],
            [-1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(points, expected)


def test_backproject_accepts_hwc_depth():
    depth = np.ones((2, 2, 1), dtype=np.float32)
    points = backproject_depth(depth, intrinsics())
    assert points.shape == (4, 3)


def test_backproject_mask_and_depth_range():
    depth = np.array([[0.5, 1.0], [2.0, 5.0]], dtype=np.float32)
    mask = np.array([[True, False], [True, True]])
    points = backproject_depth(depth, intrinsics(), mask=mask, depth_min=0.75, depth_max=3.0)
    assert points.shape == (1, 3)
    np.testing.assert_allclose(points[0], np.array([-2.0, 0.0, 2.0], dtype=np.float32))


def test_backproject_rejects_bad_intrinsics():
    with pytest.raises(ValueError):
        backproject_depth(np.ones((2, 2), dtype=np.float32), np.eye(2))


def test_rgbd_to_pointcloud_with_color_uint8():
    depth = np.ones((2, 2), dtype=np.float32)
    color = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )
    pc = rgbd_to_pointcloud(color, depth, intrinsics())
    assert pc["points"].shape == (4, 3)
    assert pc["colors"].shape == (4, 3)
    np.testing.assert_allclose(pc["colors"][0], np.array([1.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(pc["indices"], np.array([0, 1, 2, 3]))


def test_rgbd_to_pointcloud_accepts_chw_color():
    depth = np.ones((2, 2), dtype=np.float32)
    color = np.zeros((3, 2, 2), dtype=np.float32)
    color[0] = 1.0
    pc = rgbd_to_pointcloud(color, depth, intrinsics())
    np.testing.assert_allclose(pc["colors"][:, 0], 1.0)
    np.testing.assert_allclose(pc["colors"][:, 1:], 0.0)


def test_rgbd_to_pointcloud_applies_c2w_transform():
    depth = np.ones((1, 1), dtype=np.float32)
    K = np.eye(3, dtype=np.float32)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 3] = np.array([1.0, 2.0, 3.0])
    pc = rgbd_to_pointcloud(None, depth, K, c2w=c2w)
    np.testing.assert_allclose(pc["points"], np.array([[1.0, 2.0, 4.0]], dtype=np.float32))
    assert pc["colors"] is None
    np.testing.assert_allclose(pc["c2w"], c2w)


def test_rgbd_to_pointcloud_max_points_is_deterministic():
    depth = np.ones((10, 10), dtype=np.float32)
    first = rgbd_to_pointcloud(None, depth, np.eye(3), max_points=10, seed=5)
    second = rgbd_to_pointcloud(None, depth, np.eye(3), max_points=10, seed=5)
    assert first["points"].shape == (10, 3)
    np.testing.assert_array_equal(first["indices"], second["indices"])
    np.testing.assert_allclose(first["points"], second["points"])


def test_rgbd_to_pointcloud_filters_invalid_depths():
    depth = np.array([[1.0, np.nan], [0.0, 2.0]], dtype=np.float32)
    pc = rgbd_to_pointcloud(None, depth, intrinsics(), depth_min=0.0, depth_max=1.5)
    assert pc["points"].shape == (1, 3)
    np.testing.assert_array_equal(pc["indices"], np.array([0]))


class FakeDataset:
    def __getitem__(self, idx):
        color = np.zeros((2, 2, 3), dtype=np.uint8)
        depth = np.ones((2, 2, 1), dtype=np.float32)
        K = np.eye(4, dtype=np.float32)
        pose = np.eye(4, dtype=np.float32)
        pose[0, 3] = float(idx)
        return color, depth, K, pose


def test_get_frame_normalizes_dataset_tuple():
    frame = get_frame(FakeDataset(), 3)
    assert frame["frame_id"] == 3
    assert frame["color"].shape == (2, 2, 3)
    assert frame["depth"].shape == (2, 2, 1)
    assert frame["intrinsics"].shape == (3, 3)
    assert frame["c2w"].shape == (4, 4)
    assert frame["w2c"].shape == (4, 4)
    np.testing.assert_allclose(frame["c2w"][0, 3], 3.0)
    np.testing.assert_allclose(frame["w2c"][0, 3], -3.0)


def test_rgbd_to_pointcloud_rejects_bad_color_shape():
    with pytest.raises(ValueError):
        rgbd_to_pointcloud(np.zeros((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32), intrinsics())


def test_rgbd_to_pointcloud_rejects_bad_c2w_shape():
    with pytest.raises(ValueError):
        rgbd_to_pointcloud(None, np.ones((2, 2), dtype=np.float32), intrinsics(), c2w=np.eye(3))
