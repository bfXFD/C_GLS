import os
import tempfile
import unittest

import numpy as np
from unittest import mock

from utils.fsm_map import (
    ACTIVATION_GEOMETRY_ID_SPACE,
    ACTIVATION_GEOMETRY_SCHEMA_VERSION,
    build_fsm_map,
    expand_log_scales,
    fsm_map_field_sizes,
    export_runtime_reserve_geometry,
    gaussians_to_covariances,
    load_fsm_map,
    load_activation_geometry,
    load_runtime_reserve_geometry,
    load_splatam_params,
    normalize_quaternions,
    quaternions_to_rotation_matrices,
    save_fsm_map,
    selected_ids_sha256,
    validate_activation_geometry,
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


def make_activation_geometry(params, fsm_map):
    selected = np.asarray(fsm_map["indices"], dtype=np.int64)
    baseline = np.array([3, 0, 2], dtype=np.int64)
    preserved = np.array([3], dtype=np.int64)
    replacement = np.array([1, 4], dtype=np.int64)
    removed = np.array([0, 2], dtype=np.int64)
    removed_log_scales = params["log_scales"][removed]
    removed_rotations = normalize_quaternions(params["unnorm_rotations"][removed])
    local_by_global = {int(global_id): local_id for local_id, global_id in enumerate(selected)}
    reserve = np.array([], dtype=np.int64)
    return {
        "schema_version": np.asarray(ACTIVATION_GEOMETRY_SCHEMA_VERSION, dtype=np.int64),
        "id_space": np.asarray(ACTIVATION_GEOMETRY_ID_SPACE),
        "selected_global_ids": selected,
        "baseline_global_ids": baseline,
        "preserved_global_ids": preserved,
        "replacement_global_ids": replacement,
        "removed_global_ids": removed,
        "removed_means": params["means3D"][removed],
        "removed_log_scales": removed_log_scales,
        "removed_rotations": removed_rotations,
        "removed_scales": np.exp(expand_log_scales(removed_log_scales)),
        "removed_covariances": gaussians_to_covariances(removed_log_scales, removed_rotations),
        "removed_scores": np.array([0.2, 0.1], dtype=np.float32),
        "removed_normals": np.tile(np.array([[0.0, 0.0, 1.0]]), (2, 1)),
        "normal_source": np.asarray("params:normals"),
        "source_params_sha256": np.asarray("a" * 64),
        "selection_controls_sha256": np.asarray("b" * 64),
        "map_selected_ids_sha256": np.asarray(selected_ids_sha256(selected)),
        "replacement_selected_local_ids": np.asarray(
            [local_by_global[int(global_id)] for global_id in replacement], dtype=np.int64
        ),
        "reserve_global_ids": reserve,
        "reserve_means": np.empty((0, 3), dtype=np.float32),
        "reserve_normals": np.empty((0, 3), dtype=np.float32),
        "reserve_scores": np.empty(0, dtype=np.float32),
        "reserve_covariances": np.empty((0, 3, 3), dtype=np.float32),
        "reserve_scales": np.empty((0, 3), dtype=np.float32),
        "reserve_rotations": np.empty((0, 4), dtype=np.float32),
        "reserve_count": np.asarray(0, dtype=np.int64),
        "reserve_policy": np.asarray("test"),
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

    def test_epsilon_must_be_finite(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            normalize_quaternions(np.array([[1.0, 0.0, 0.0, 0.0]]), eps=np.nan)
        with self.assertRaisesRegex(ValueError, "finite"):
            gaussians_to_covariances(np.zeros((1, 1)), np.array([[1.0, 0.0, 0.0, 0.0]]), eps=np.inf)

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
        self.assertEqual(fsm_map["scores"].shape, (3,))
        np.testing.assert_allclose(fsm_map["scores"], scores[indices])

    def test_map_optional_provenance_hashes_are_validated(self):
        params = make_params(num_gaussians=3)
        fsm_map = build_fsm_map(
            params, source_params_sha256="a" * 64, selection_controls_sha256="b" * 64
        )
        self.assertEqual(str(fsm_map["source_params_sha256"]), "a" * 64)
        invalid = dict(fsm_map)
        invalid["selection_controls_sha256"] = np.asarray("bad")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            validate_fsm_map(invalid)

    def test_map_provenance_binds_activation_geometry(self):
        params = make_params(num_gaussians=5)
        fsm_map = build_fsm_map(
            params,
            indices=np.array([4, 3, 1], dtype=np.int64),
            source_params_sha256="a" * 64,
            selection_controls_sha256="b" * 64,
        )
        geometry = make_activation_geometry(params, fsm_map)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "activation_geometry.npz")
            np.savez(path, **geometry)
            loaded = load_activation_geometry(
                path,
                fsm_map,
                source_params_sha256=str(fsm_map["source_params_sha256"]),
                selection_controls_sha256=str(fsm_map["selection_controls_sha256"]),
            )
        np.testing.assert_array_equal(loaded["selected_global_ids"], fsm_map["indices"])

    def test_subset_build_only_constructs_retained_covariances(self):
        params = make_params(num_gaussians=10000)
        indices = np.array([9, 500, 9999], dtype=np.int64)
        from utils import fsm_map as module

        original = module.gaussians_to_covariances
        with mock.patch.object(module, "gaussians_to_covariances", wraps=original) as patched:
            fsm_map = build_fsm_map(params, indices=indices)
        self.assertEqual(patched.call_args.args[0].shape[0], indices.size)
        self.assertEqual(fsm_map["covariances"].shape, (indices.size, 3, 3))

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

    def test_load_migrates_legacy_full_length_scores(self):
        params = make_params(num_gaussians=5)
        indices = np.array([3, 1, 4], dtype=np.int64)
        fsm_map = build_fsm_map(params, indices=indices, scores=np.arange(5, dtype=np.float64))
        fsm_map["scores"] = np.arange(5, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "legacy_map.npz")
            np.savez(output_path, **fsm_map)
            loaded = load_fsm_map(output_path)
        np.testing.assert_array_equal(loaded["scores"], np.array([3.0, 1.0, 4.0]))

    def test_field_sizes_report_array_bytes(self):
        fsm_map = build_fsm_map(make_params(num_gaussians=3))
        sizes = fsm_map_field_sizes(fsm_map)
        self.assertEqual(sizes["means"], fsm_map["means"].nbytes)
        self.assertEqual(sizes["scores"], fsm_map["scores"].nbytes)

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

    def test_activation_geometry_maps_replacements_through_reordered_map_indices(self):
        params = make_params(num_gaussians=5)
        fsm_map = build_fsm_map(params, indices=np.array([4, 3, 1], dtype=np.int64))
        geometry = make_activation_geometry(params, fsm_map)
        np.testing.assert_array_equal(geometry["replacement_selected_local_ids"], np.array([2, 0]))
        validate_activation_geometry(geometry, fsm_map)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "activation_geometry.npz")
            np.savez(path, **geometry)
            loaded = load_activation_geometry(
                path, fsm_map, source_params_sha256="a" * 64, selection_controls_sha256="b" * 64
            )
        np.testing.assert_array_equal(loaded["selected_global_ids"], fsm_map["indices"])

    def test_runtime_reserve_export_preserves_geometry_and_rejects_wrong_map(self):
        params = make_params(num_gaussians=5)
        fsm_map = build_fsm_map(params, indices=np.array([4, 3, 1], dtype=np.int64))
        fsm_map.update(source_params_sha256=np.asarray("a" * 64), selection_controls_sha256=np.asarray("b" * 64))
        geometry = make_activation_geometry(params, fsm_map)
        with tempfile.TemporaryDirectory() as tmpdir:
            old, new = (os.path.join(tmpdir, name) for name in ("audit.npz", "runtime.npz"))
            np.savez(old, **geometry)
            export_runtime_reserve_geometry(old, fsm_map, new)
            with self.assertRaises(FileExistsError):
                export_runtime_reserve_geometry(old, fsm_map, new)
            loaded = load_runtime_reserve_geometry(new, fsm_map)
            self.assertNotIn("selected_global_ids", loaded)
            for key in ("reserve_global_ids", "reserve_means", "reserve_normals", "reserve_scores", "reserve_covariances"):
                self.assertEqual(loaded[key].dtype, geometry[key].dtype)
                np.testing.assert_array_equal(loaded[key], geometry[key])
            with self.assertRaisesRegex(ValueError, "indices and order"):
                load_runtime_reserve_geometry(new, dict(fsm_map, indices=np.array([3, 4, 1])))
            with self.assertRaisesRegex(ValueError, "source_params_sha256"):
                load_runtime_reserve_geometry(new, dict(fsm_map, source_params_sha256=np.asarray("c" * 64)))
            np.savez(new, **dict(loaded, reserve_global_ids=np.array([4], dtype=np.int64)))
            with self.assertRaisesRegex(ValueError, "overlaps"):
                load_runtime_reserve_geometry(new, fsm_map)

    def test_activation_geometry_rejects_id_and_hash_mismatches(self):
        params = make_params(num_gaussians=5)
        fsm_map = build_fsm_map(params, indices=np.array([4, 3, 1], dtype=np.int64))
        geometry = make_activation_geometry(params, fsm_map)
        for field, value, message in (
            ("selected_global_ids", np.array([3, 4, 1]), "exactly match"),
            ("replacement_global_ids", np.array([1, 2]), "selected_global_ids"),
            ("removed_global_ids", np.array([0, 4]), "must not belong"),
            ("map_selected_ids_sha256", np.asarray("0" * 64), "does not match"),
        ):
            invalid = dict(geometry)
            invalid[field] = value
            with self.assertRaisesRegex(ValueError, message):
                validate_activation_geometry(invalid, fsm_map)

    def test_activation_geometry_accepts_v1_without_reserve(self):
        params = make_params(num_gaussians=5)
        fsm_map = build_fsm_map(params, indices=np.array([4, 3, 1], dtype=np.int64))
        geometry = make_activation_geometry(params, fsm_map)
        geometry["schema_version"] = np.asarray(1, dtype=np.int64)
        for field in list(geometry):
            if field.startswith("reserve_"):
                geometry.pop(field)
        validate_activation_geometry(geometry, fsm_map)

    def test_activation_geometry_rejects_reserve_shape_and_outside_mismatches(self):
        params = make_params(num_gaussians=5)
        fsm_map = build_fsm_map(params, indices=np.array([4, 3, 1], dtype=np.int64))
        geometry = make_activation_geometry(params, fsm_map)
        invalid_outside = dict(geometry)
        invalid_outside["reserve_global_ids"] = np.array([4], dtype=np.int64)
        invalid_outside["reserve_means"] = params["means3D"][[4]]
        invalid_outside["reserve_normals"] = np.array([[0.0, 0.0, 1.0]])
        invalid_outside["reserve_scores"] = np.array([1.0])
        invalid_outside["reserve_covariances"] = np.eye(3)[None]
        invalid_outside["reserve_scales"] = np.ones((1, 3))
        invalid_outside["reserve_rotations"] = np.array([[1.0, 0.0, 0.0, 0.0]])
        invalid_outside["reserve_count"] = np.asarray(1, dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "strictly outside"):
            validate_activation_geometry(invalid_outside, fsm_map)
        invalid_shape = dict(geometry)
        invalid_shape["reserve_global_ids"] = np.array([2], dtype=np.int64)
        invalid_shape["reserve_count"] = np.asarray(1, dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "reserve_means"):
            validate_activation_geometry(invalid_shape, fsm_map)


if __name__ == "__main__":
    unittest.main()
