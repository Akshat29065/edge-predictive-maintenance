"""Analytics helpers: the anomaly transform and the prescriptive rule table."""
import numpy as np

from features import FEATURES, LOG_FEATURES, anomaly_vector
from rules import URGENCY, prescribe


def row(**over):
    base = {f: 1.0 for f in FEATURES}
    base.update(over)
    return base


def test_kurtosis_and_skew_are_not_log_transformed():
    assert "kurtosis" not in LOG_FEATURES and "skew" not in LOG_FEATURES
    v = anomaly_vector(row(kurtosis=-0.3, skew=-0.2))
    assert v[FEATURES.index("kurtosis")] == -0.3 and v[FEATURES.index("skew")] == -0.2


def test_positive_features_are_logged_and_never_infinite():
    v = anomaly_vector(row(rms=np.e, peak=0.0))
    assert np.isclose(v[FEATURES.index("rms")], 1.0)
    assert np.isfinite(v).all()


def test_prescribe_by_fault_size():
    assert prescribe("normal", 0, False)["urgency"] == "none"
    assert prescribe("normal", 0, True)["urgency"] == "low"
    assert prescribe("inner", 7, True)["urgency"] == "medium"
    assert prescribe("outer", 14, True)["urgency"] == "high"
    assert prescribe("ball", 21, True)["window"] == "at the next stop / immediately"
    assert all(prescribe(c, s, True)["urgency"] in URGENCY for c in ("inner", "outer", "ball") for s in (7, 14, 21))
