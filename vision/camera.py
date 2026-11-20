"""Simulated thermal camera.

It looks at the simulator's current condition (shared/state.json) and returns a matching thermal
image from the demo set: a healthy motor shows a healthy image, a faulty bearing a corroded one,
at the matching load. This is the ONLY place that reads the ground truth, standing in for the
physical world. To use a real camera, replace `capture()` with a function that grabs a frame
(the rest of the service never sees the state file).
"""
import csv
import json
import os
import random
import time
from pathlib import Path

from PIL import Image

LOAD_BY_HP = {0: "no", 1: "medium", 2: "medium", 3: "full"}
# The simulator rewrites the state file every tick. An older file means the simulator is down:
# a camera that kept showing the last condition would hide that outage from the vision stream.
MAX_STATE_AGE_S = float(os.environ.get("CAMERA_STATE_MAX_AGE_S", "15"))


class SimulatedCamera:
    def __init__(self, demo_dir, state_file):
        self.dir, self.state_file = Path(demo_dir), Path(state_file)
        with (self.dir / "index.csv").open() as f:
            self.index = list(csv.DictReader(f))
        if not self.index:
            raise RuntimeError(f"no demo images in {self.dir}")

    def capture(self):
        """Returns (PIL image, frame id), or None if the simulator has not started or has stopped."""
        try:
            state = json.loads(self.state_file.read_text())
        except (OSError, ValueError):
            return None
        if time.time() - float(state.get("timestamp", 0)) > MAX_STATE_AGE_S:
            return None
        label = "healthy" if state["healthy"] else "faulty"
        load = LOAD_BY_HP.get(state["load_hp"], "medium")
        pool = [r for r in self.index if r["label"] == label and r["load"] == load] \
            or [r for r in self.index if r["label"] == label]
        pick = random.choice(pool)
        return Image.open(self.dir / pick["file"]).convert("RGB"), pick["file"]