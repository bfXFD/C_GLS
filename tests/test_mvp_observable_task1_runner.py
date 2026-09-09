from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_mvp_observable_task1.sh"


def test_task1_runner_uses_strict_protocol_and_isolated_output():
    text = SCRIPT.read_text()

    assert "experiments/FSM/mvp_observable_v1/task1_strict_baseline" in text
    assert "--protocol mvp_strict" in text
    assert '--first-pose-protocol "$FIRST_POSE_PROTOCOL"' in text
    assert "uses_gt_current_frame_initialization" in text
    assert "uses_gt_fallback" in text
    assert "audit_v2_full_rerun" in text
    assert "cp \"$ROOM0_MAP\"" not in text
    assert "cp \"$OFFICE0_MAP\"" not in text


def test_task1_runner_defines_smoke_and_long_trajectory_cases():
    text = SCRIPT.read_text()

    assert 'run_case "replica_room0_0_smoke_0_50"' in text
    assert 'run_case "replica_room0_0_full"' in text
    assert 'run_case "replica_office0_0_full"' in text
    assert '"$ROOM0_CONFIG" "$ROOM0_MAP" 0 50' in text
    assert '"$ROOM0_CONFIG" "$ROOM0_MAP" 0 -1' in text
    assert '"$OFFICE0_CONFIG" "$OFFICE0_MAP" 0 -1' in text
