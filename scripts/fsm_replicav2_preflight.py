import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_scene(root, scene):
    scene_root = root / scene / "imap"
    splits = {}
    for split in ("00", "01"):
        base = scene_root / split
        rgb = sorted((base / "rgb").glob("rgb_*.png"))
        depth = sorted((base / "depth").glob("depth_*.png"))
        trajectory = base / "traj_w_c.txt"
        if not rgb or len(rgb) != len(depth) or not trajectory.is_file():
            raise FileNotFoundError(f"Incomplete Replica V2 scene/split: {scene}/{split}")
        pose_count = sum(1 for line in trajectory.read_text().splitlines() if line.strip())
        if pose_count != len(rgb):
            raise ValueError(f"Pose/image mismatch {scene}/{split}: {pose_count} vs {len(rgb)}")
        splits[split] = {
            "frames": len(rgb), "trajectory": str(trajectory.resolve()),
            "trajectory_sha256": sha256(trajectory),
        }
    return {"scene": scene, "map_split": "imap/00", "query_split": "imap/01", "query_frames_used_for_map": 0, "splits": splits}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    root = Path(args.data_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"Replica V2 data root does not exist: {root}. "
            "Download it first with: DATA_ROOT=/home/ras/xbf/C_GLS_code/data/Replica_V2 "
            "bash /home/ras/xbf/C_GLS_code/bash_scripts/download_replicav2.sh"
        )
    document = {"schema_version": 1, "data_root": str(root), "scenes": [inspect_scene(root, scene) for scene in args.scenes]}
    text = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
