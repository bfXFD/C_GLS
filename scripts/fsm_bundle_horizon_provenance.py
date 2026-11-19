import argparse
import hashlib
import json
import shutil
from pathlib import Path


SCENES = ("office0", "room2", "office4", "fr3", "reverse")
HORIZONS = (1, 3, 5, 10)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--required-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(args.required_root).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists(): shutil.rmtree(output)
    output.mkdir(parents=True)
    records = []
    for scene in SCENES:
        scene_dir = output / scene; scene_dir.mkdir()
        paired = root / f"paired/{scene}"
        for source, name in ((paired / "manifest.json", "paired_manifest.json"), (paired / ".state/paired.done.json", "paired_done.json")):
            shutil.copy2(source, scene_dir / name)
            records.append({"path": str((scene_dir / name).relative_to(output)), "sha256": sha256(scene_dir / name)})
        for horizon in HORIZONS:
            hdir = scene_dir / f"h{horizon}"; hdir.mkdir()
            files = (
                (root / f"horizon_gtfree/{scene}/h{horizon}/provenance.json", "replay_provenance.json"),
                (root / f"horizon_labels/{scene}/h{horizon}/horizon_gt_labels_provenance.json", "label_provenance.json"),
            )
            for source, name in files:
                shutil.copy2(source, hdir / name)
                records.append({"path": str((hdir / name).relative_to(output)), "sha256": sha256(hdir / name)})
    (output / "bundle_manifest.json").write_text(json.dumps({"schema_version": 1, "records": records}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
