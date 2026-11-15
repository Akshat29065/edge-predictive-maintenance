"""Analytics service: reads vibration features from InfluxDB, writes diagnosis + prescription back.

Reads  bearing_features  (written by Telegraf)
Writes predictions       (condition, probabilities, health score, anomaly, urgency, action)
Never reads shared/state.json: the ground truth is not an input.
"""
import json
import logging
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

from features import FEATURES, anomaly_vector
from rules import URGENCY, prescribe

log = logging.getLogger("analytics")
MODELS = Path(os.environ.get("MODELS_DIR", "/models"))


class Analyzer:
    def __init__(self, models_dir=MODELS, smooth=10):
        self.smooth = smooth
        meta = json.loads((models_dir / "meta.json").read_text())
        self.classes, self.severities = meta["fault_classes"], meta["severities_mil"]
        self.fault = lgb.Booster(model_file=str(models_dir / "fault_model.txt"))
        self.sev = lgb.Booster(model_file=str(models_dir / "severity_model.txt"))
        a = json.loads((models_dir / "anomaly.json").read_text())
        self.mean, self.inv = np.array(a["mean"]), np.array(a["inv_cov"])
        self.threshold = a["threshold"]

    def analyze(self, rows):
        """rows: list of dicts (oldest -> newest) with the 8 features. Returns a prediction dict."""
        rows = rows[-self.smooth:]
        X = np.array([[r[f] for f in FEATURES] for r in rows])
        proba = self.fault.predict(X).mean(axis=0)
        sev_proba = self.sev.predict(X).mean(axis=0)
        diffs = np.array([anomaly_vector(r) for r in rows]) - self.mean
        dist = float(np.sqrt(np.einsum("ij,jk,ik->i", diffs, self.inv, diffs)).mean())
        cond = self.classes[int(proba.argmax())]
        severity = self.severities[int(sev_proba.argmax())] if cond != "normal" else 0
        anomaly = dist > self.threshold
        rx = prescribe(cond, severity, anomaly)
        return {
            "condition": cond, "confidence": float(proba.max()),
            "health_score": float(proba[self.classes.index("normal")]),
            "severity_mil": severity, "anomaly_score": dist / self.threshold,
            "anomaly": anomaly, "urgency": rx["urgency"], "urgency_level": URGENCY.index(rx["urgency"]),
            "action": rx["action"], "window": rx["window"], "n_windows": len(rows),
            **{f"p_{c}": float(p) for c, p in zip(self.classes, proba)},
        }


def main():
    from influxdb_client import InfluxDBClient, Point
    from influxdb_client.client.write_api import SYNCHRONOUS

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    url, token = os.environ["INFLUXDB_URL"], os.environ["INFLUXDB_TOKEN"]
    org, bucket = os.environ["INFLUXDB_ORG"], os.environ["INFLUXDB_BUCKET"]
    interval = float(os.environ.get("ANALYTICS_INTERVAL_S", "2"))
    smooth = int(os.environ.get("SMOOTH_WINDOWS", "10"))
    analyzer = Analyzer(smooth=smooth)
    fields = ",".join(f'"{f}"' for f in FEATURES)
    query = f'''
from(bucket: "{bucket}") |> range(start: -{smooth * 3 + 10}s)
  |> filter(fn: (r) => r._measurement == "bearing_features" and r.asset == "motor1")
  |> filter(fn: (r) => contains(value: r._field, set: [{fields}]))
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> sort(columns: ["_time"])'''
    last_time = None
    with InfluxDBClient(url=url, token=token, org=org) as client:
        write = client.write_api(write_options=SYNCHRONOUS)
        while True:
            try:
                recs = [r for t in client.query_api().query(query) for r in t.records]
                if len(recs) >= 3 and recs[-1].get_time() != last_time:
                    last_time = recs[-1].get_time()
                    result = analyzer.analyze([{f: r.values[f] for f in FEATURES} for r in recs])
                    p = Point("predictions").tag("asset", "motor1").time(last_time)
                    for k, v in result.items():
                        p = p.field(k, v)
                    write.write(bucket=bucket, org=org, record=p)
                    log.info("%s conf=%.2f sev=%s anomaly=%.2f urgency=%s", result["condition"],
                             result["confidence"], result["severity_mil"], result["anomaly_score"], result["urgency"])
            except Exception as exc:  # keep running if InfluxDB is briefly unavailable
                log.warning("cycle failed: %s", exc)
            time.sleep(interval)


if __name__ == "__main__":
    main()