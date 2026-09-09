import numpy as np
import pytest

from scripts import fsm_mvp_preflight as preflight


def test_preflight_checks_fake_inputs_and_npz_metadata(tmp_path, monkeypatch):
    config = tmp_path / "config.py"
    params = tmp_path / "params.npz"
    config.write_text("config = {}\n")
    np.savez(params, means3D=np.zeros((5, 3)))
    monkeypatch.setattr(preflight, "check_scipy", lambda: "scipy.spatial.cKDTree")
    monkeypatch.setattr(preflight, "free_disk_bytes", lambda path: 30 * preflight.GIB)
    monkeypatch.setattr(preflight, "available_ram_bytes", lambda: 20 * preflight.GIB)
    result = preflight.run_preflight([str(config)], [str(params)], str(tmp_path / "out"), preflight.GIB)
    assert result["ok"] is True
    assert result["npz_metadata"][0]["gaussian_count"] == 5


def test_preflight_rejects_low_disk(tmp_path, monkeypatch):
    value = tmp_path / "input"
    value.write_text("x")
    monkeypatch.setattr(preflight, "check_scipy", lambda: "scipy.spatial.cKDTree")
    monkeypatch.setattr(preflight, "free_disk_bytes", lambda path: 2 * preflight.GIB)
    monkeypatch.setattr(preflight, "available_ram_bytes", lambda: 20 * preflight.GIB)
    with pytest.raises(RuntimeError, match="free disk"):
        preflight.run_preflight([str(value)], [], str(tmp_path), 0)
