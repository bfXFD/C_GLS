import json

import numpy as np
import pytest

from scripts import fsm_supplement_dynamic as bench


def probe(primary, candidate, submitted=False, frame=0, added=128):
    pose = np.eye(4)
    candidate_pose = pose.copy()
    candidate_pose[0, 3] = candidate
    main = pose.copy()
    main[0, 3] = primary
    row = dict(frame_id=frame, timestamp=frame, gt_pose=pose, force_eligible=True,
               natural_triggered=True, force_requested=False, candidate_attempted=True,
               candidate_submitted=False, candidate_basic_valid=True, gated_adopted=submitted,
               candidate_reserve_count=added, probe_latency_ms=12, candidate_latency_ms=9,
               reserve_metrics=dict(reserve_visible=256, reserve_selected=added))
    for branch, value in dict(primary=main, candidate=candidate_pose, baseline=main,
                              gated=candidate_pose if submitted else main, ungated=candidate_pose).items():
        row[f"{branch}_pose"] = value
        row[f"{branch}_tracking_valid"] = True
        row[f"{branch}_source"] = "primary" if value is main else "reserve"
    return bench.probe_record(row)


def test_probe_stats_keep_explicit_outcomes_and_candidate_effect():
    rows = [probe(.10, .03, False, 0), probe(.03, .10, True, 1), probe(.03, .03, True, 2)]
    assert all(row["has_gt"] is True for row in json.loads(json.dumps(rows, allow_nan=False)))
    stats = bench.probe_summary(rows)
    assert stats["cells"]["primary_0_candidate_1"]["not_submitted"] == 1
    assert stats["cells"]["primary_1_candidate_0"]["submitted"] == 1
    assert stats["cells"]["primary_0_candidate_0"]["submission_rate"] is None
    assert stats["gated_minus_baseline"] == -1
    assert stats["attempt_candidate_else_baseline_net"] == 0
    assert stats["probe_ms_total"] == 36


def test_equal_added_subset_and_trigger_sample_are_explicit():
    rows = {"a": [probe(.09, .01, frame=i) for i in range(2)],
            "b": [probe(.09, .01, frame=i, added=127 if i else 128) for i in range(2)]}
    assert bench.aligned_subset(rows)["a"]["flow"]["triggered"] == 1
    candidates = [{"frame_id": i, "triggered": i % 3 == 0, "eligible": True} for i in range(16)]
    picked = bench.trigger_sample(candidates, 3)
    assert len(set(picked)) == 6
    assert [v // 4 for v in picked] == [0, 0, 1, 2, 3, 3]


def test_dry_run_does_not_run_or_write(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bench, "run_localization", lambda *a, **kw: pytest.fail("ran localization"))
    out = tmp_path / "dry"
    bench.main(["--stage", "e4-closed", "--scenes", "room2", "--output-root", str(out)])
    plan = json.loads(capsys.readouterr().out)
    assert [x["arm"] for x in plan["runs"]] == ["baseline", "rerun", "static", "dynamic"]
    verifier = [r["config"]["observable_reserve"]["candidate_verifier"] for r in plan["runs"]]
    assert verifier.count(verifier[0]) == 4
    assert not out.exists()
