import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fsm_deduplicate_verified_maps as dedup


def configure(monkeypatch, tmp_path, candidate_bytes=b"frozen-map"):
    canonical = tmp_path / "canonical.npz"
    candidate = tmp_path / "candidate.npz"
    canonical.write_bytes(b"frozen-map")
    candidate.write_bytes(candidate_bytes)
    report = tmp_path / "report.json"
    monkeypatch.setattr(dedup, "CANONICAL", canonical)
    monkeypatch.setattr(dedup, "CANDIDATES", [candidate])
    return canonical, candidate, report


def test_plan_preserves_inodes(monkeypatch, tmp_path):
    canonical, candidate, report = configure(monkeypatch, tmp_path)
    original = candidate.stat().st_ino
    monkeypatch.setattr(sys, "argv", ["dedup", "--report", str(report)])
    dedup.main()
    assert candidate.stat().st_ino == original
    assert canonical.stat().st_ino != original
    assert not json.loads(report.read_text())["applied"]


def test_apply_preserves_content_and_paths(monkeypatch, tmp_path):
    canonical, candidate, report = configure(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", ["dedup", "--apply", "--report", str(report)])
    dedup.main()
    assert candidate.exists() and canonical.exists()
    assert candidate.stat().st_ino == canonical.stat().st_ino
    assert candidate.read_bytes() == b"frozen-map"


def test_content_mismatch_is_never_replaced(monkeypatch, tmp_path):
    canonical, candidate, report = configure(monkeypatch, tmp_path, b"other-data")
    original = candidate.stat().st_ino
    monkeypatch.setattr(sys, "argv", ["dedup", "--apply", "--report", str(report)])
    with pytest.raises(ValueError, match="mismatch"):
        dedup.main()
    assert candidate.stat().st_ino == original
    assert candidate.read_bytes() == b"other-data"


def test_existing_report_cannot_overwrite_map(monkeypatch, tmp_path):
    canonical, candidate, _ = configure(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", ["dedup", "--report", str(canonical)])
    with pytest.raises(ValueError, match="new JSON"):
        dedup.main()
    assert canonical.read_bytes() == candidate.read_bytes() == b"frozen-map"
