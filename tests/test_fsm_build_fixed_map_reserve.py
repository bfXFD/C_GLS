import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts.fsm_build_fixed_map_reserve import main
from tests.test_fsm_localize import make_fake_map
from utils.fsm_map import load_activation_geometry, load_fsm_map


ROOT = Path(__file__).resolve().parent.parent


def test_fixed_map_reserve_preserves_base_ids_and_binds_sidecar(tmp_path):
    base_map, _ = make_fake_map(tmp_path)
    source = np.load(base_map, allow_pickle=False)
    selected = np.asarray(source["indices"], dtype=np.int64)
    source.close()
    params_path = tmp_path / "params.npz"
    count = 20
    np.savez(
        params_path,
        means3D=np.column_stack((np.linspace(-1, 1, count), np.zeros(count), np.ones(count))),
        log_scales=np.log(np.full((count, 1), 0.01)),
        unnorm_rotations=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]]), (count, 1)),
        logit_opacities=np.zeros((count, 1)),
    )
    output_map = tmp_path / "hybrid" / "fsm_map.npz"
    geometry = tmp_path / "hybrid" / "activation_geometry.npz"
    main([
        "--params", str(params_path), "--base-map", str(base_map),
        "--map-output", str(output_map), "--activation-geometry-output", str(geometry),
        "--reserve-count", "4",
    ])
    enhanced = load_fsm_map(str(output_map))
    np.testing.assert_array_equal(enhanced["indices"], selected)
    sidecar = load_activation_geometry(str(geometry), enhanced)
    assert sidecar["reserve_global_ids"].shape == (4,)
    assert not np.intersect1d(sidecar["reserve_global_ids"], selected).size
    done = json.loads((tmp_path / "hybrid" / "activation_geometry.npz.done.json").read_text())
    assert done["reserve_count"] == 4


def test_fixed_map_reserve_absolute_script_entrypoint_imports_repo_modules():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/fsm_build_fixed_map_reserve.py"), "--help"],
        cwd=ROOT.parent,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert "Attach a bounded independent reserve" in result.stdout
