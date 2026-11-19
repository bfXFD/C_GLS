import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def file_record(path: str) -> dict:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return {
        "path": str(source),
        "size_bytes": source.stat().st_size,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Record reverse independent-revisit pipeline provenance.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--input", action="append", default=[], required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    if "audit_v2_full_rerun" in str(output):
        parser.error(f"refusing frozen audit output path: {output}")
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "map_sequence": "rgbd_dataset_freiburg1_desk2",
        "query_sequence": "rgbd_dataset_freiburg1_desk",
        "map_construction_query_frame_count": 0,
        "splatam_environment": {"SEED": 0, "SCENE_NUM": 1},
        "frame_start": args.frame_start,
        "frame_end": args.frame_end,
        "inputs": [file_record(path) for path in args.input],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "inputs": len(manifest["inputs"])}, sort_keys=True))


if __name__ == "__main__":
    main()
