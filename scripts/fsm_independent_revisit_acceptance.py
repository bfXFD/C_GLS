import argparse
import hashlib
import json
from pathlib import Path


ARM_ORDER = ("full_map", "compact_fps10", "corr_weak128_v4", "replace128_v4")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root: Path, require_all: bool = False) -> dict:
    arms = {}
    failures = []
    for arm in ARM_ORDER:
        case_dir = root / arm / "localize" / arm
        summary_path = case_dir / "summary.json"
        state_path = root / arm / ".state" / f"{arm}.done.json"
        if not summary_path.is_file() or not state_path.is_file():
            arms[arm] = {"status": "missing"}
            if require_all or arm == "full_map":
                failures.append(f"{arm}: missing completed result")
            continue
        summary = json.loads(summary_path.read_text())
        state = json.loads(state_path.read_text())
        provenance = state.get("provenance", {})
        pose = provenance.get("dataset_pose_provenance", {})
        checks = {
            "strict_protocol": summary.get("evaluation_protocol") == "mvp_strict",
            "first_pose_only_gt": summary.get("uses_gt_first_pose") is True
            and summary.get("uses_gt_current_frame_initialization") is False
            and summary.get("uses_gt_fallback") is False,
            "map_sequence": pose.get("pose_reference_sequence") == "rgbd_dataset_freiburg1_desk2",
            "query_sequence": pose.get("query_sequence") == "rgbd_dataset_freiburg1_desk",
            "fixed_transform_recorded": pose.get("fixed_frame_transform_A") is not None,
            "no_query_map_frames": pose.get("map_construction_query_frame_count") == 0,
            "runtime_hash_recorded": bool(provenance.get("runtime_source_sha256")),
            "config_hash_recorded": bool(provenance.get("resolved_config_sha256")),
            "map_hash_recorded": bool(provenance.get("map_sha256")),
        }
        if arm in ("corr_weak128_v4", "replace128_v4"):
            checks["sidecar_hash_recorded"] = bool(provenance.get("observable_reserve_geometry_sha256"))
        status = "accepted" if all(checks.values()) else "rejected"
        arms[arm] = {
            "status": status,
            "checks": checks,
            "summary_sha256": _sha256(summary_path),
            "state_sha256": _sha256(state_path),
        }
        if status != "accepted":
            failures.append(f"{arm}: " + ", ".join(name for name, passed in checks.items() if not passed))
    order_ok = all(
        arms[ARM_ORDER[index]]["status"] != "accepted"
        or all(arms[earlier]["status"] == "accepted" for earlier in ARM_ORDER[:index])
        for index in range(len(ARM_ORDER))
    )
    if not order_ok:
        failures.append("serial arm order violated")
    return {"accepted": not failures, "arm_order": list(ARM_ORDER), "serial_order_ok": order_ok, "arms": arms, "failures": failures}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Audit the desk/desk2 independent-revisit protocol outputs.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    for path in (args.root, args.output):
        if path and "audit_v2_full_rerun" in str(Path(path)):
            parser.error(f"refusing frozen audit output path: {path}")
    result = audit(Path(args.root), require_all=args.require_all)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
    print(text, end="")
    if not result["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
