"""Measures how reliably a local model makes valid tool calls, and how fast, on the agent's real prompt.

Run from the repo root with the .venv active (standard library only). Stop the agent first so the
model has the CPU to itself:   docker compose stop agent
    python scripts/probe_llm.py qwen3:4b qwen3:0.6b          # models must already be pulled
    python scripts/probe_llm.py qwen3:4b --trials 5
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "agent")
import policy                                   # noqa: E402
from llm import LLMError, OllamaClient          # noqa: E402
from tools import validate                      # noqa: E402


def sit(cond, sev, urg, vision, action, window):
    s = {"asset": "motor1",
         "analytics": {"condition": cond, "confidence": 1.0, "severity_mil": sev, "anomaly": cond != "normal",
                       "anomaly_score": 7.05 if cond != "normal" else 0.6, "urgency": urg,
                       "recommended_action": action, "window": window, "stale": False},
         "vision": {"label": vision, "p_faulty": 1.0 if vision == "faulty" else 0.0, "frames": 3, "stale": False}}
    s["agreement"] = policy.classify(s)
    return s


CASES = {
    "healthy": sit("normal", 0, "none", "healthy", "No action. Continue routine monitoring.", "-"),
    "fault":   sit("inner", 14, "high", "faulty",
                   "Inner-race defect. Plan bearing replacement; check shaft alignment and lubrication. Estimated fault size 14 mil.",
                   "at the next stop / immediately"),
    "conflict": sit("inner", 7, "medium", "healthy", "Inner-race defect. Plan bearing replacement.", "within 2 weeks"),
}

ap = argparse.ArgumentParser()
ap.add_argument("models", nargs="+")
ap.add_argument("--trials", type=int, default=3)
ap.add_argument("--mode", choices=["json", "tools"], default="json")
ap.add_argument("--url", default=os.environ.get("OLLAMA_HOST_URL", "http://localhost:11434"))
args = ap.parse_args()
system = Path("agent/prompts/system.txt").read_text()
client = OllamaClient(args.url, system, timeout_s=180, mode=args.mode)

for model in args.models:
    print(f"\n=== {model} (mode={args.mode})")
    for name, s in CASES.items():
        ok = grounded_ok = 0
        times = []
        for i in range(args.trials):
            t0 = time.time()
            try:
                calls = [c for c in (validate(r) for r in client.decide(model, s)) if c]
                times.append(time.time() - t0)
                ok += bool(calls)
                final, notes = policy.apply_guardrails(calls, s) if calls else ([], [])
                grounded_ok += bool(calls) and not any("ungrounded" in n for n in notes)
                if i == 0 and calls:
                    print(f"  [{name}] first answer: {[(c['name'], c['arguments']) for c in calls]}")
            except LLMError as exc:
                times.append(time.time() - t0)
                print(f"  [{name}] trial {i + 1} failed: {exc}")
        avg = sum(times) / len(times)
        print(f"  [{name}] valid tool calls {ok}/{args.trials}, text grounded {grounded_ok}/{args.trials}, avg {avg:.1f}s")