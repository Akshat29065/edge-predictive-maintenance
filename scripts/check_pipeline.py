"""Milestone A check: prints the latest points that reached InfluxDB next to the simulator's true state.

Run from the repo root with the .venv active, while `docker compose up` is running:
    python scripts/check_pipeline.py
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from influxdb_client import InfluxDBClient

load_dotenv()

URL = os.environ.get("INFLUXDB_HOST_URL", "http://localhost:8086")  # the host-side port mapping
TOKEN = os.environ["INFLUXDB_TOKEN"]
ORG = os.environ["INFLUXDB_ORG"]
BUCKET = os.environ["INFLUXDB_BUCKET"]

QUERY = f'''
from(bucket: "{BUCKET}")
  |> range(start: -30s)
  |> filter(fn: (r) => r._measurement == "bearing_features")
  |> filter(fn: (r) => r._field == "rms" or r._field == "kurtosis" or r._field == "heartbeat")
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> sort(columns: ["_time"], desc: true)
  |> limit(n: 10)
'''

with InfluxDBClient(url=URL, token=TOKEN, org=ORG) as client:
    records = [r for table in client.query_api().query(QUERY) for r in table.records]

if not records:
    print("No points in the last 30 s. Check: docker compose ps, docker compose logs telegraf")
else:
    print(f"{len(records)} most recent points (newest first):")
    for r in records:
        v = r.values
        print(f"  {r.get_time():%H:%M:%S}  heartbeat={v.get('heartbeat')}  "
              f"rms={v.get('rms'):.4f}  kurtosis={v.get('kurtosis'):.3f}")

state_path = Path("shared/state.json")
if state_path.exists():
    s = json.loads(state_path.read_text())
    print(f"\nSimulator is currently replaying: {s['file']} (class={s['cls']}, "
          f"fault size={s['size_mil']} mil, healthy={s['healthy']})")

# --- Phase 2: latest analytics output -------------------------------------------------
PRED = f'''
from(bucket: "{BUCKET}")
  |> range(start: -30s)
  |> filter(fn: (r) => r._measurement == "predictions")
  |> filter(fn: (r) => r._field == "condition" or r._field == "confidence" or r._field == "severity_mil"
                    or r._field == "anomaly_score" or r._field == "urgency" or r._field == "action")
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> sort(columns: ["_time"], desc: true)
  |> limit(n: 5)
'''
with InfluxDBClient(url=URL, token=TOKEN, org=ORG) as client:
    preds = [r for table in client.query_api().query(PRED) for r in table.records]
print()
if not preds:
    print("No predictions yet. Check: docker compose logs analytics (did you run python analytics/train.py?)")
else:
    print("Latest predictions (newest first):")
    for r in preds:
        v = r.values
        print(f"  {r.get_time():%H:%M:%S}  {v.get('condition'):7s} conf={v.get('confidence'):.2f} "
              f"size={v.get('severity_mil'):.0f} mil  anomaly={v.get('anomaly_score'):.2f}  urgency={v.get('urgency')}")
    print("  action:", preds[0].values.get("action"))