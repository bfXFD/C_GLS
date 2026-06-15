import os
import tempfile
import unittest

import numpy as np

from utils.fsm_map import (
    build_fsm_map,
    expand_log_scales,
    gaussians_to_covariances,
    load_fsm_map,
    load_splatam_params,
    normalize_quaternions,
    quaternions_to_rotation_matrices,
    save_fsm_map,
    validate_fsm_map,
)


def make_params(num_gaussians=4, anisotropic=False):
    means = np.arange(num_gaussians * 3, dtype=np.float32).reshape(num_gaussians, 3) * 0.1
    if anisotropic:
        log_scales = np.log(
            np.stack(
                [
                    np.linspace(0.01, 0.04, num_gaussians),
                    np.linspace(0.02, 0.05, num_gaussians),
                    np.linspace(0.03, 0.06, num_gaussians),
                ],
                axis=1,
            )
        ).astype(np.float32)
    else:
        log_scales = np.log(np.linspace(0.01, 0.04, num_gaussians)[:, None]).astype(np.float32)

    rotations = np.tile(np.array([[2.0, 0.0, 0.0, 0.0]], dtype=np.float32), (num_gaussians, 1))
    return {
        "means3D": means,
        "log_scales": log_scales,
        "unnorm_rotations": rotations,
        "rgb_colors": np.zeros((num_gaussians, 3), dtype=np.float32),
        "logit_opacities": np.zeros((num_gaussians, 1), dtype=np.float32),
    }


class TestFsmMap(unittest.TestCase):
    def test_expand_isotropic_log_scales(self):
        log_scales = np.log(np.array([[0.1], [0.2]], dtype=np.float32))
        expanded = expand_log_scales(log_scales)
        self.assertEqual(expanded.shape, (2, 3))
        np.testing.assert_allclose(expanded[:, 0], expanded[:, 1])
        np.testing.assert_allclose(expanded[:, 1], expanded[:, 2])

    def test_expand_anisotropic_log_scales_keeps_channels(self):
        log_scales = np.log(np.array([[0.1, 0.2, 0.3]], dtype=np.float32))
        expanded = expand_log_scales(log_scales)
        self.assertEqual(expanded.shape, (1, 3))
        np.testing.assert_allclose(expanded, log_scales)

    def test_normalize_quaternions(self):
        quats = np.array([[2.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0]])
        normalized = normalize_quaternions(quats)
        np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), 1.0)

    def test_quaternion_to_rotation_identity(self):
        rotations = quaternions_to_rotation_matrices(np.array([[2.0, 0.0, 0.0, 0.0]]))
        np.testing.assert_allclose(rotations[0], np.eye(3), atol=1e-7)

    def test_covariances_are_symmetric_positive_definite(self):
        params = make_params(num_gaussians=3, anisotropic=True)
        covariances = gaussians_to_covariances(params["log_scales"], params["unnorm_rotations"], eps=1e-6)
        self.assertEqual(covariances.shape, (3, 3, 3))
        np.testing.assert_allclose(covariances, np.transpose(covariances, (0, 2, 1)), atol=1e-8)
        self.assertTrue(np.all(np.linalg.eigvalsh(covariances) > 0))

    def test_build_full_fsm_map(self):
        params = make_params(num_gaussians=5)
        fsm_map = build_fsm_map(params, strategy="full", source_params_path="params.npz")
        self.assertEqual(fsm_map["means"].shape, (5, 3))
        self.assertEqual(fsm_map["covariances"].shape, (5, 3, 3))
        self.assertEqual(fsm_map["scales"].shape, (5, 3))
        self.assertEqual(fsm_map["rotations"].shape, (5, 4))
        np.testing.assert_array_equal(fsm_map["indices"], np.arange(5))
        validate_fsm_map(fsm_map)

    def test_build_subset_preserves_index_order(self):
        params = make_params(num_gaussians=5)
        indices = np.array([3, 1, 4], dtype=np.int64)
        scores = np.linspace(0.0, 1.0, 5)
        fsm_map = build_fsm_map(params, indices=indices, scores=scores, strategy="external")
        np.testing.assert_array_equal(fsm_map["indices"], indices)
        np.testing.assert_allclose(fsm_map["means"], params["means3D"][indices])
        self.assertEqual(fsm_map["scores"].shape, (5,))

    def test_save_load_roundtrip(self):
        params = make_params(num_gaussians=4)
        fsm_map = build_fsm_map(params, indices=np.array([0, 2]), strategy="test", lambda_min=1.5, delta=0.25)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "fsm_map.npz")
            save_fsm_map(fsm_map, output_path)
            loaded = load_fsm_map(output_path)
        for key in ["means", "covariances", "scales", "rotations", "indices"]:
            np.testing.assert_allclose(loaded[key], fsm_map[key])
        self.assertEqual(str(loaded["strategy"]), "test")

    def test_load_splatam_params_validates_required_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            params_path = os.path.join(tmpdir, "params.npz")
            np.savez(params_path, means3D=np.zeros((1, 3)), log_scales=np.zeros((1, 1)))
            with self.assertRaises(KeyError):
                load_splatam_params(params_path)

    def test_validate_rejects_missing_field(self):
        params = make_params(num_gaussians=2)
        fsm_map = build_fsm_map(params)
        fsm_map.pop("means")
        with self.assertRaises(KeyError):
            validate_fsm_map(fsm_map)

    def test_build_rejects_duplicate_indices(self):
        params = make_params(num_gaussians=3)
        with self.assertRaises(ValueError):
            build_fsm_map(params, indices=np.array([0, 0]))

    def test_build_rejects_bad_scores_shape(self):
        params = make_params(num_gaussians=3)
        with self.assertRaises(ValueError):
            build_fsm_map(params, scores=np.ones(2))

    def test_save_creates_parent_directory(self):
        params = make_params(num_gaussians=2)
        fsm_map = build_fsm_map(params)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "nested", "fsm_map.npz")
            save_fsm_map(fsm_map, output_path)
            self.assertTrue(os.path.exists(output_path))


if __name__ == "__main__":
    unittest.main()
