import numpy as np
from importlib.machinery import SourceFileLoader
from pathlib import Path

from datasets.gradslam_datasets.tum import TUMDataset


def test_tum_parse_list_uses_numpy_str_dtype(tmp_path):
    list_path = tmp_path / "rgb.txt"
    list_path.write_text("# comment\n0.0 rgb/0.png\n1.0 rgb/1.png\n")

    data = TUMDataset.parse_list(None, str(list_path), skiprows=1)

    assert data.shape == (2, 2)
    assert data.dtype.type is np.str_
    assert data[0, 1] == "rgb/0.png"


def test_tum_eval_config_has_required_depth_loss_flags(monkeypatch):
    monkeypatch.setenv("SEED", "0")
    monkeypatch.setenv("SCENE_NUM", "0")
    config_path = Path(__file__).resolve().parents[1] / "configs" / "tum" / "tum_eval.py"
    module = SourceFileLoader("tum_eval", str(config_path)).load_module()

    assert module.config["tracking"]["ignore_outlier_depth_loss"] is False
    assert module.config["mapping"]["ignore_outlier_depth_loss"] is False
