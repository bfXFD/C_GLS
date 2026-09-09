import csv
import json
from pathlib import Path

import scripts.fsm_mvp_paper_summary as summary


def test_write_csv_and_latex_outputs(tmp_path):
    rows = [{"scene": "Room_0", "ate": "0.12", "gain": "5.0%"}]
    summary.write_csv(tmp_path / "table.csv", rows)
    summary.write_latex(tmp_path / "table.tex", rows, (("scene", "Scene"), ("ate", "ATE"), ("gain", "Gain")))
    with open(tmp_path / "table.csv", newline="") as stream:
        assert list(csv.DictReader(stream)) == rows
    latex = (tmp_path / "table.tex").read_text()
    assert "Room\\_0" in latex
    assert "5.0\\%" in latex


def test_paper_summary_default_output_is_ignored_experiment_path():
    args = summary.parse_args([])
    assert Path(args.output_dir) == summary.MVP / "paper_summary_v1"
