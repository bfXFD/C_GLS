import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Close bidirectional revisit provenance in one audit root.")
    parser.add_argument("--required-root", required=True)
    parser.add_argument("--forward-root", required=True)
    parser.add_argument("--combined-table", required=True)
    args = parser.parse_args()
    required = Path(args.required_root).resolve()
    forward = Path(args.forward_root).resolve()
    combined = Path(args.combined_table).resolve()
    target = required / "forward_revisit_reference"
    target.mkdir(parents=True, exist_ok=True)
    copies = {
        "acceptance.json": forward / "acceptance.json",
        "independent_revisit_table.csv": forward / "independent_revisit_table.csv",
        "independent_revisit_table.provenance.json": forward / "independent_revisit_table.provenance.json",
    }
    for name, source in copies.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, target / name)
    reverse = required / "reverse_revisit"
    document = {
        "schema_version": 1,
        "directions": {
            "desk_to_desk2": {
                "reference_dir": "forward_revisit_reference",
                "acceptance_sha256": sha256(target / "acceptance.json"),
                "table_sha256": sha256(target / "independent_revisit_table.csv"),
                "table_provenance_sha256": sha256(target / "independent_revisit_table.provenance.json"),
            },
            "desk2_to_desk": {
                "reference_dir": "reverse_revisit",
                "acceptance_sha256": sha256(reverse / "acceptance.json"),
                "table_sha256": sha256(reverse / "independent_revisit_table.csv"),
                "table_provenance_sha256": sha256(reverse / "independent_revisit_table.provenance.json"),
                "pipeline_provenance_sha256": sha256(reverse / "pipeline_provenance.json"),
            },
        },
        "combined_table_path": str(combined),
        "combined_table_sha256": sha256(combined),
        "note": "Forward run closure is copied by hash into the required-acceptance root; large per-frame outputs remain in the referenced source directory.",
    }
    output = required / "bidirectional_revisit_provenance.json"
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "directions": 2}, indent=2))


if __name__ == "__main__":
    main()
