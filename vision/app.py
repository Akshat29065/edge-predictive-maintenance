"""Vision service: captures a thermal frame, classifies it, writes the result to InfluxDB.

Writes  vision_inspection  (label, p_faulty, p_healthy, confidence, image)
The camera is simulated (camera.py); the classifier never sees the true condition.
"""
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from camera import SimulatedCamera

log = logging.getLogger("vision")
MODELS = Path(os.environ.get("MODELS_DIR", "/models"))


class Classifier:
    def __init__(self, models_dir=MODELS):
        meta = json.loads((models_dir / "meta.json").read_text())
        self.labels, self.size = meta["labels"], meta["size"]
        self.mask_top = meta.get("mask_top", 0.0)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1          # keep the footprint small on an edge device
        self.session = ort.InferenceSession(str(models_dir / "thermal.onnx"), opts,
                                            providers=["CPUExecutionProvider"])

    def classify(self, img):
        img = img.convert("RGB").resize((self.size, self.size))
        a = np.array(img)
        a[: int(self.size * self.mask_top)] = 0          # same blanking as in training
        x = a.astype(np.float32) / 255.0
        probs = self.session.run(None, {"image": x.transpose(2, 0, 1)[None]})[0][0]
        i = int(probs.argmax())
        return {"label": self.labels[i], "confidence": float(probs[i]),
                **{f"p_{l}": float(p) for l, p in zip(self.labels, probs)}}


def main():
    from influxdb_client import InfluxDBClient, Point
    from influxdb_client.client.write_api import SYNCHRONOUS

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    url, token = os.environ["INFLUXDB_URL"], os.environ["INFLUXDB_TOKEN"]
    org, bucket = os.environ["INFLUXDB_ORG"], os.environ["INFLUXDB_BUCKET"]
    interval = float(os.environ.get("VISION_INTERVAL_S", "5"))
    clf = Classifier()
    camera = SimulatedCamera(os.environ.get("DEMO_DIR", "/demo"), os.environ.get("STATE_FILE", "/shared/state.json"))
    with InfluxDBClient(url=url, token=token, org=org) as client:
        write = client.write_api(write_options=SYNCHRONOUS)
        while True:
            try:
                frame = camera.capture()
                if frame is not None:
                    img, frame_id = frame
                    r = clf.classify(img)
                    p = Point("vision_inspection").tag("asset", "motor1").field("image", frame_id)
                    for k, v in r.items():
                        p = p.field(k, v)
                    write.write(bucket=bucket, org=org, record=p)
                    log.info("%s p_faulty=%.2f", r["label"], r["p_faulty"])
            except Exception as exc:
                log.warning("cycle failed: %s", exc)
            time.sleep(interval)


if __name__ == "__main__":
    main()