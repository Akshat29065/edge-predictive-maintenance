"""The simulated camera must go quiet when the simulator does."""
import json
import time

from PIL import Image

import camera


def make(tmp_path, healthy, age_s):
    demo = tmp_path / "demo"; demo.mkdir()
    for i, label in enumerate(("healthy", "faulty")):
        Image.new("RGB", (8, 8), (i * 100, 0, 0)).save(demo / f"img_{i}.jpg")
    (demo / "index.csv").write_text("file,label,load\nimg_0.jpg,healthy,full\nimg_1.jpg,faulty,full\n")
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"timestamp": time.time() - age_s, "healthy": healthy, "load_hp": 3}))
    return camera.SimulatedCamera(demo, state)


def test_fresh_state_gives_a_matching_frame(tmp_path):
    img, frame_id = make(tmp_path, healthy=False, age_s=1).capture()
    assert frame_id == "img_1.jpg"


def test_old_state_means_no_frame(tmp_path):
    assert make(tmp_path, healthy=False, age_s=camera.MAX_STATE_AGE_S + 5).capture() is None


def test_missing_state_means_no_frame(tmp_path):
    cam = make(tmp_path, healthy=True, age_s=0)
    (tmp_path / "state.json").unlink()
    assert cam.capture() is None
