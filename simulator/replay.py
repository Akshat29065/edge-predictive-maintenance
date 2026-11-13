"""Replays windowed CWRU features according to a scenario file."""
import csv
from collections import defaultdict

import yaml

# Order matters: it defines the Modbus register map (2 registers per feature).
FEATURES = ["rms", "peak", "p2p", "crest", "kurtosis", "skew", "shape", "centroid"]


def load_windows(csv_path):
    """Return {file_name: [row, ...]} with rows sorted by window index."""
    by_file = defaultdict(list)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            by_file[row["file"]].append(row)
    for rows in by_file.values():
        rows.sort(key=lambda r: int(r["window"]))
    return dict(by_file)


def load_scenario(path):
    with open(path) as f:
        return yaml.safe_load(f)


def validate(by_file, scenario):
    missing = sorted({step["file"] for step in scenario["steps"]} - set(by_file))
    if missing:
        raise SystemExit(
            f"Scenario refers to files that are not in the feature table: {missing}. "
            f"Available: {sorted(by_file)}"
        )


def replay(by_file, scenario):
    """Yield one sample per tick, following the scenario steps in order.

    If a step lasts longer than the file has windows, the windows repeat.
    """
    while True:
        for index, step in enumerate(scenario["steps"]):
            rows = by_file[step["file"]]
            for t in range(int(step["duration_s"])):
                row = rows[t % len(rows)]
                yield {
                    "step": index,
                    "file": step["file"],
                    "cls": row["cls"],
                    "size_mil": int(row["size_mil"]),
                    "load_hp": int(row["load_hp"]),
                    "values": [float(row[name]) for name in FEATURES],
                }
        if not scenario.get("loop", True):
            return