import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Merge compatible FSM summary CSV files.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--input", nargs="+", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    fieldnames = None
    for input_path in args.input:
        with open(input_path, newline="") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or [])
            elif list(reader.fieldnames or []) != fieldnames:
                raise ValueError(f"Summary schema mismatch: {input_path}")
            rows.extend(reader)
    if not fieldnames:
        raise ValueError("No summary columns found")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Merged {len(rows)} rows into {output_path}")


if __name__ == "__main__":
    main()
