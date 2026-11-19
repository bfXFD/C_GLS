import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fsm_prune_intermediate_maps import archive_selection, references


def test_archive_preserves_indices_metadata_but_not_geometry(tmp_path):
    params = tmp_path / "params.npz"
    np.savez(params, means3D=np.zeros((4, 3)))
    source = tmp_path / "fsm_map.npz"
    np.savez(source, means=np.zeros((2, 3)), indices=np.array([1, 3]),
             source_params_path=str(params), strategy="fps", scores=np.ones(4))
    target = tmp_path / "archive" / "selection.npz"
    result = archive_selection(source, target)
    with np.load(target) as data:
        np.testing.assert_array_equal(data["indices"], [1, 3])
        assert "means" not in data and "scores" not in data
        assert data["strategy"].item() == "fps"
    assert source.is_file() and result["selected_count"] == 2


def test_reference_match_preserves_named_map_directory(tmp_path, monkeypatch):
    import fsm_prune_intermediate_maps as module
    monkeypatch.setattr(module, "FSM", tmp_path)
    path = tmp_path / "scene" / "batch" / "maps" / "fps" / "fsm_map.npz"
    assert references(path, [("script.py", "scene/batch/maps/fps")]) == ["script.py"]
