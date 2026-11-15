"""Train the predictive models from the CWRU feature table.

Run from the repo root:  python analytics/train.py
Outputs (analytics/models/): fault_model.txt, severity_model.txt, anomaly.json, metrics.json

Evaluation never uses random splits: windows overlap by 50 %, so a random split
leaks. We train on one motor load and test on the other (0 hp <-> 3 hp).
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).parent))
from features import FEATURES, anomaly_matrix  # noqa: E402

CSV = Path("data/processed/cwru_features.csv")
OUT = Path("analytics/models")
FAULT_CLASSES = ["normal", "inner", "ball", "outer"]
SEVERITIES = [7, 14, 21]
PARAMS = dict(n_estimators=150, learning_rate=0.05, num_leaves=8, min_child_samples=10,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              class_weight="balanced", random_state=0, verbose=-1, n_jobs=2)


def fit_fault(df):
    y = df["cls"].map(FAULT_CLASSES.index)
    return lgb.LGBMClassifier(**PARAMS).fit(df[FEATURES], y)


def fit_severity(df):
    f = df[df.cls != "normal"]
    y = f["size_mil"].map(SEVERITIES.index)
    return lgb.LGBMClassifier(**PARAMS).fit(f[FEATURES], y)


def fit_anomaly(normal_df, q=0.99):
    X = anomaly_matrix(normal_df)
    mu = X.mean(axis=0)
    cov = np.cov(X, rowvar=False) + 1e-3 * np.eye(X.shape[1])   # ridge: only a few hundred windows
    inv = np.linalg.inv(cov)
    model = {"mean": mu.tolist(), "inv_cov": inv.tolist()}
    d = distances(model, normal_df)
    model["threshold"] = float(np.quantile(d, q))
    model["scale"] = float(np.median(d))
    return model


def distances(model, df):
    X = anomaly_matrix(df) - np.array(model["mean"])
    inv = np.array(model["inv_cov"])
    return np.sqrt(np.einsum("ij,jk,ik->i", X, inv, X))


def smooth_predict(proba, k=10):
    """Average class probabilities over the last k windows (what the live service does)."""
    out = np.zeros_like(proba)
    for i in range(len(proba)):
        out[i] = proba[max(0, i - k + 1): i + 1].mean(axis=0)
    return out


def evaluate(train, test, title):
    res = {"title": title}
    m = fit_fault(train)
    y = test["cls"].map(FAULT_CLASSES.index).values
    p = m.predict_proba(test[FEATURES])
    # smoothing must run per source file (a file is one continuous stretch of time)
    ps = np.vstack([smooth_predict(p[(test.file == f).values]) for f in test.file.unique()])
    order = np.concatenate([np.where(test.file == f)[0] for f in test.file.unique()])
    ys = y[order]
    res["fault_f1_raw"] = round(f1_score(y, p.argmax(1), average="macro"), 3)
    res["fault_f1_smoothed"] = round(f1_score(ys, ps.argmax(1), average="macro"), 3)
    res["fault_confusion_smoothed"] = confusion_matrix(ys, ps.argmax(1), labels=range(4)).tolist()
    res["healthy_vs_faulty_recall_smoothed"] = {
        "normal_recall": round(float(((ps.argmax(1) == 0) & (ys == 0)).sum() / max((ys == 0).sum(), 1)), 3),
        "fault_recall": round(float(((ps.argmax(1) != 0) & (ys != 0)).sum() / max((ys != 0).sum(), 1)), 3),
    }
    res["per_file_accuracy_smoothed"] = {
        f: round(float((ps.argmax(1)[test.file.values[order] == f] == ys[test.file.values[order] == f]).mean()), 2)
        for f in test.file.unique()}
    # severity (faulty windows only)
    s = fit_severity(train)
    tf = test[test.cls != "normal"]
    sy = tf["size_mil"].map(SEVERITIES.index).values
    res["severity_f1"] = round(f1_score(sy, s.predict(tf[FEATURES]), average="macro"), 3)
    return res


def evaluate_anomaly(df):
    """Healthy baseline = first 60 % of each Normal file; the last 30 % (after a 10 % gap) is held out.

    A baseline from one load does NOT transfer to the other (healthy 0 hp and 3 hp differ by many
    standard deviations), so the baseline must cover the loads the machine runs at.
    """
    tr, te = [], []
    for f, g in df[df.cls == "normal"].groupby("file"):
        n = len(g)
        tr.append(g.iloc[: int(.6 * n)]); te.append(g.iloc[int(.7 * n):])
    tr, te = pd.concat(tr), pd.concat(te)
    a = fit_anomaly(tr)
    thr = a["threshold"]
    faulty = df[df.cls != "normal"]
    d = distances(a, faulty)
    return {"threshold": round(thr, 2),
            "false_alarm_rate_on_held_out_normal": round(float((distances(a, te) > thr).mean()), 3),
            "detection_rate_by_class": {c: round(float((d[(faulty.cls == c).values] > thr).mean()), 3)
                                        for c in FAULT_CLASSES[1:]},
            "note": "baseline from one load alone gives ~100 % false alarms at the other load"}


def main():
    df = pd.read_csv(CSV)
    report = {"cross_load": [
        evaluate(df[df.load_hp == 0], df[df.load_hp == 3], "train 0 hp -> test 3 hp"),
        evaluate(df[df.load_hp == 3], df[df.load_hp == 0], "train 3 hp -> test 0 hp"),
    ]}
    report["anomaly_time_split"] = evaluate_anomaly(df)
    print(json.dumps(report["anomaly_time_split"], indent=1))
    for r in report["cross_load"]:
        print(r["title"], "fault_f1_smoothed", r["fault_f1_smoothed"], "severity_f1", r["severity_f1"], r["healthy_vs_faulty_recall_smoothed"])

    # Deployed models use every load: more variety, and the evaluation above is the honest estimate.
    OUT.mkdir(parents=True, exist_ok=True)
    fit_fault(df).booster_.save_model(str(OUT / "fault_model.txt"))
    fit_severity(df).booster_.save_model(str(OUT / "severity_model.txt"))
    anomaly = fit_anomaly(df[df.cls == "normal"])
    (OUT / "anomaly.json").write_text(json.dumps(anomaly))
    meta = {"features": FEATURES, "fault_classes": FAULT_CLASSES, "severities_mil": SEVERITIES}
    (OUT / "meta.json").write_text(json.dumps(meta))
    report["note"] = "cross-load metrics are the honest estimate; deployed models use all loads"
    (OUT / "metrics.json").write_text(json.dumps(report, indent=1))
    print("saved models to", OUT)


if __name__ == "__main__":
    main()