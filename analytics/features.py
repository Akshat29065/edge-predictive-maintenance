"""Shared feature handling for training and the live analyzer."""
import numpy as np

FEATURES = ["rms", "peak", "p2p", "crest", "kurtosis", "skew", "shape", "centroid"]
# skew and kurtosis can be <= 0 (kurtosis is the Fisher/excess kind: ~0 for healthy noise), so neither is
# log-transformed. Logging a value <= 0 clipped it to log(1e-9) and corrupted the anomaly baseline.
LOG_FEATURES = ["rms", "peak", "p2p", "crest", "shape", "centroid"]


def anomaly_vector(row):
    """row: dict feature->float. Log-transform the positive features."""
    return np.array(
        [np.log(max(row[f], 1e-9)) if f in LOG_FEATURES else row[f] for f in FEATURES]
    )


def anomaly_matrix(df):
    return np.vstack([anomaly_vector(r) for r in df[FEATURES].to_dict("records")])