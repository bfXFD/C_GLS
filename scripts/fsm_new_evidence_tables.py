import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MVP = ROOT / "experiments/FSM/mvp_observable_v1"
FORWARD = MVP / "independent_revisit_desk2_v1/independent_revisit_table.csv"
REQUIRED = MVP / "required_acceptance_experiments_v1"
REVERSE = REQUIRED / "reverse_revisit/independent_revisit_table.csv"
OUT = MVP / "new_acceptance_paper_tables_v1"


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def revisit_table():
    rows = []
    for direction, source in (("desk_to_desk2", FORWARD), ("desk2_to_desk", REVERSE)):
        for row in read_csv(source):
            rows.append({"direction": direction, **row})
    write_csv(OUT / "table_bidirectional_independent_revisit.csv", rows)


def horizon_table():
    rows = []
    frozen = json.loads((REQUIRED / "frozen/horizon_thresholds_office0.json").read_text())
    for horizon, entry in sorted(frozen["horizons"].items(), key=lambda item: int(item[0])):
        metrics = entry["development_metrics"]
        rows.append({
            "split": "development", "scene": "office0", "horizon": int(horizon),
            **entry["thresholds"],
            **{key: metrics[key] for key in (
                "events", "harmful", "beneficial", "neutral", "harmful_rejection_recall",
                "beneficial_retention", "false_safe_rate", "false_reject_rate",
                "inconclusive_rate", "selection_rate",
            )},
        })
    for scene in ("room2", "office4", "fr3", "reverse"):
        report = json.loads((REQUIRED / f"evaluation/{scene}.json").read_text())
        for horizon, metrics in sorted(report["horizons"].items(), key=lambda item: int(item[0])):
            rows.append({
                "split": "test", "scene": scene, "horizon": int(horizon),
                **metrics["thresholds"],
                **{key: metrics[key] for key in (
                    "events", "harmful", "beneficial", "neutral", "harmful_rejection_recall",
                    "beneficial_retention", "false_safe_rate", "false_reject_rate",
                    "inconclusive_rate", "selection_rate",
                )},
            })
    write_csv(OUT / "table_frozen_horizon_diagnostics.csv", rows)


def protocol_table():
    rows = [
        {
            "direction": "desk_to_desk2", "map_sequence": "rgbd_dataset_freiburg1_desk",
            "query_sequence": "rgbd_dataset_freiburg1_desk2", "map_query_frame_count": 0,
            "first_pose_gt_only": True, "all_four_common": 635,
        },
        {
            "direction": "desk2_to_desk", "map_sequence": "rgbd_dataset_freiburg1_desk2",
            "query_sequence": "rgbd_dataset_freiburg1_desk", "map_query_frame_count": 0,
            "first_pose_gt_only": True, "all_four_common": 589,
        },
    ]
    write_csv(OUT / "table_independent_revisit_protocol.csv", rows)


def main():
    revisit_table()
    horizon_table()
    protocol_table()
    print(json.dumps({"output": str(OUT), "tables": 3}, indent=2))


if __name__ == "__main__":
    main()
