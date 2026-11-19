"""Agent service: reads analytics + vision results from InfluxDB, asks the local LLM (Qwen via Ollama)
which maintenance functions to call, checks the answer against a rule-based policy, and executes it.

The agent never reads shared/state.json. It sees only what the sensors' analytics report.
Falls back to the rule-based policy if the LLM is slow, down, or answers with something invalid.
"""
import json
import logging
import os
import time
from collections import Counter
from pathlib import Path

import policy
from llm import LLMError, OllamaClient
from tools import Executor, validate

log = logging.getLogger("agent")
ASSET = "motor1"


class Agent:
    def __init__(self, llm, models, executor):
        self.llm, self.models, self.executor = llm, [m for m in models if m], executor

    def decide(self, sit):
        """Returns (calls, mode, notes, error). Never raises."""
        errors = []
        for i, model in enumerate(self.models):
            try:
                raw = self.llm.decide(model, sit)
                calls = [c for c in (validate(r) for r in raw) if c]
                if not calls:
                    raise LLMError(f"invalid tool call(s): {str(raw)[:160]}")
                final, notes = policy.apply_guardrails(calls, sit)
                return final, ("llm" if i == 0 else "llm_fallback_model") + ("+guardrail" if notes else ""), notes, None
            except LLMError as exc:
                errors.append(f"{model}: {exc}")
        return policy.rule_decision(sit), "rules", [], "; ".join(errors)

    def act(self, sit):
        t0 = time.time()
        calls, mode, notes, error = self.decide(sit)
        for c in calls:
            self.executor.run(c, ASSET)
        latency = time.time() - t0
        record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "mode": mode, "latency_s": round(latency, 2),
                  "situation": sit, "calls": calls, "guardrail_notes": notes, "llm_error": error}
        self.executor._append("actions.log", json.dumps(record))
        return calls, mode, notes, error, latency


def read_situation(client, bucket, org, stale_s):
    now = time.time()
    pred_q = f'''
from(bucket: "{bucket}") |> range(start: -5m)
  |> filter(fn: (r) => r._measurement == "predictions")
  |> filter(fn: (r) => contains(value: r._field, set: ["condition","confidence","severity_mil","anomaly_score","anomaly","urgency","action","window"]))
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> sort(columns: ["_time"], desc: true) |> limit(n: 5)'''
    vis_q = f'''
from(bucket: "{bucket}") |> range(start: -5m)
  |> filter(fn: (r) => r._measurement == "vision_inspection")
  |> filter(fn: (r) => r._field == "label" or r._field == "p_faulty")
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> sort(columns: ["_time"], desc: true) |> limit(n: 8)'''
    api = client.query_api()
    # newest first; skip any row that is missing a field (a half-written or older-schema point)
    PRED_KEYS = ("condition", "confidence", "severity_mil", "anomaly", "anomaly_score", "urgency", "action", "window")
    preds = [r for t in api.query(pred_q, org=org) for r in t.records
             if all(r.values.get(k) is not None for k in PRED_KEYS)][:1]
    vis = [r for t in api.query(vis_q, org=org) for r in t.records
           if r.values.get("label") is not None and r.values.get("p_faulty") is not None][:3]
    sit = {"asset": ASSET, "analytics": None, "vision": None}
    if preds:
        r = preds[0]; v = r.values; age = now - r.get_time().timestamp()
        sit["analytics"] = {"condition": v["condition"], "confidence": round(float(v["confidence"]), 2),
                            "severity_mil": int(v["severity_mil"]), "anomaly": bool(v["anomaly"]),
                            "anomaly_score": round(float(v["anomaly_score"]), 2), "urgency": v["urgency"],
                            "recommended_action": v["action"], "window": v["window"],
                            "stale": age > stale_s}
    if vis:
        label = Counter(r.values["label"] for r in vis).most_common(1)[0][0]
        age = now - vis[0].get_time().timestamp()
        sit["vision"] = {"label": label, "p_faulty": round(sum(float(r.values["p_faulty"]) for r in vis) / len(vis), 2),
                         "frames": len(vis), "stale": age > stale_s}
    sit["agreement"] = policy.classify(sit)
    return sit


def situation_key(sit):
    a, v = sit.get("analytics") or {}, sit.get("vision") or {}
    return (sit["agreement"], a.get("condition"), a.get("severity_mil"), a.get("urgency"), v.get("label"))


def main():
    from influxdb_client import InfluxDBClient, Point
    from influxdb_client.client.write_api import SYNCHRONOUS

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    env = os.environ.get
    url, token, org, bucket = env("INFLUXDB_URL"), env("INFLUXDB_TOKEN"), env("INFLUXDB_ORG"), env("INFLUXDB_BUCKET")
    interval = float(env("AGENT_INTERVAL_S", "10"))
    reminder = float(env("REMINDER_S", "120"))
    min_conf = float(env("MIN_CONFIDENCE", "0.8"))
    stale_s = float(env("STALE_S", "30"))
    system = Path(env("PROMPT_FILE", "/app/prompts/system.txt")).read_text()
    llm = OllamaClient(env("OLLAMA_URL", "http://ollama:11434"), system, float(env("LLM_TIMEOUT_S", "30")), env("LLM_MODE", "json"))
    agent = Agent(llm, [env("LLM_MODEL", "qwen3:4b"), env("LLM_FALLBACK_MODEL", "")],
                  Executor(env("LOG_DIR", "/logs")))
    last_key, last_act = None, 0.0
    with InfluxDBClient(url=url, token=token, org=org) as client:
        write = client.write_api(write_options=SYNCHRONOUS)
        while True:
            try:
                sit = read_situation(client, bucket, org, stale_s)
                a = sit["analytics"]
                waiting = sit["agreement"] == "unknown"
                transition = a is not None and not a["stale"] and a["confidence"] < min_conf
                key = situation_key(sit)
                due = key != last_key or (sit["agreement"] != "healthy" and time.time() - last_act >= reminder)
                if waiting or transition or not due:
                    pass
                else:
                    calls, mode, notes, error, latency = agent.act(sit)
                    last_key, last_act = key, time.time()
                    names = ",".join(c["name"] for c in calls)
                    log.info("%s -> %s [%s, %.1fs]%s", sit["agreement"], names, mode, latency,
                             f" notes={notes}" if notes else "")
                    if error:
                        log.warning("LLM unavailable, used rules: %s", error)
                    p = (Point("agent_decisions").tag("asset", ASSET).field("agreement", sit["agreement"])
                         .field("mode", mode).field("tools", names).field("latency_s", latency)
                         .field("summary", json.dumps(calls)[:900]))
                    write.write(bucket=bucket, org=org, record=p)
            except Exception as exc:
                log.warning("cycle failed: %s", exc)
            time.sleep(interval)


if __name__ == "__main__":
    main()