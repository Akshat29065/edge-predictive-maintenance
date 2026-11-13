import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import resample_poly
from scipy.stats import kurtosis, skew

RAW = Path("data/raw/cwru")
OUT = Path("data/processed/cwru_features.csv")
FS = 12000                                   # common rate every file is converted to (Hz)
SOURCE_FS = {"Normal_0": 48000, "Normal_3": 48000}   # normal baseline files were recorded at 48 kHz
WIN, HOP = 2048, 1024

CLASS_NAMES = {"Normal": "normal", "IR": "inner", "B": "ball", "OR": "outer"}
NAME_RE = re.compile(r"^(Normal|IR|B|OR)(\d{3})?(?:at6)?_(\d)$")

EXPECTED = (
    ["Normal_0", "Normal_3"]
    + [f"{c}{s}{'at6' if c == 'OR' else ''}_{l}"
       for c in ("IR", "B", "OR") for s in ("007", "014", "021") for l in (0, 3)]
)


def window_features(w, fs):
    rms = np.sqrt(np.mean(w ** 2))
    peak = np.max(np.abs(w))
    spec = np.abs(np.fft.rfft(w * np.hanning(len(w))))
    freqs = np.fft.rfftfreq(len(w), 1 / fs)
    return {
        "rms": rms,
        "peak": peak,
        "p2p": np.ptp(w),
        "crest": peak / rms,
        "kurtosis": kurtosis(w),
        "skew": skew(w),
        "shape": rms / np.mean(np.abs(w)),
        "centroid": float(np.sum(freqs * spec) / np.sum(spec)),
    }


rows, missing = [], []
for name in EXPECTED:
    path = RAW / f"{name}.mat"
    if not path.exists():
        missing.append(name)
        continue

    m = NAME_RE.match(name)
    cls = CLASS_NAMES[m.group(1)]
    size_mil = int(m.group(2)) if m.group(2) else 0
    load = int(m.group(3))

    mat = sio.loadmat(path)
    key = next(k for k in mat if k.endswith("_DE_time"))
    x = mat[key].ravel().astype(float)
    n_raw = len(x)

    src_fs = SOURCE_FS.get(name, FS)
    if src_fs != FS:
        g = np.gcd(FS, src_fs)
        x = resample_poly(x, FS // g, src_fs // g)   # includes an anti-alias filter

    n_win = 0
    for i, start in enumerate(range(0, len(x) - WIN + 1, HOP)):
        feats = window_features(x[start:start + WIN], FS)
        rows.append({"file": name, "cls": cls, "size_mil": size_mil,
                     "load_hp": load, "window": i, **feats})
        n_win += 1
    print(f"{name:12s} source={src_fs // 1000:2d}k raw={n_raw:7d} "
          f"at12k={len(x):7d} windows={n_win}")

OUT.parent.mkdir(parents=True, exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)

print(f"\n{len(df)} windows from {len(EXPECTED) - len(missing)} files -> {OUT}")
print(df.groupby("cls").size().to_string())
if missing:
    print("MISSING:", ", ".join(missing))