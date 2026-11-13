import csv
from pathlib import Path

RAW = Path("data/raw/thermal")
OUT = Path("data/processed/thermal_manifest.csv")
CONDITIONS = {
    "HNL": ("healthy", "no"), "HML": ("healthy", "medium"), "HFL": ("healthy", "full"),
    "CNL": ("faulty", "no"), "CML": ("faulty", "medium"), "CFL": ("faulty", "full"),
}
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

rows = []
for folder in RAW.rglob("*"):
    if folder.is_dir() and folder.name.upper() in CONDITIONS:
        label, load = CONDITIONS[folder.name.upper()]
        for img in sorted(folder.iterdir()):
            if img.suffix.lower() in EXTS:
                rows.append([str(img), folder.name.upper(), label, load])

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["path", "condition", "label", "load"])
    w.writerows(rows)

counts = {}
for r in rows:
    counts[r[1]] = counts.get(r[1], 0) + 1
print(len(rows), "images", counts)