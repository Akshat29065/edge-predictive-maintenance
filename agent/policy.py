"""Deterministic policy: classifies the situation, and decides what MUST happen.

It is the fallback when the LLM is unavailable, and a guardrail around the LLM's choices:
the model may add detail, but it cannot skip a required alert or invent one the sensors don't support.
"""
import re

from tools import SEVERITIES

FAULT_NAMES = {"inner": "inner-race", "outer": "outer-race", "ball": "rolling-element (ball)"}
URGENCY_TO_SEVERITY = {"none": "info", "low": "info", "medium": "warning", "high": "critical"}


def classify(sit):
    """healthy | confirmed_fault | conflict | stale | unknown"""
    a, v = sit.get("analytics"), sit.get("vision")
    if (a and a["stale"]) or (v and v["stale"]):
        return "stale"
    if not a or not v:
        return "unknown"
    a_fault, v_fault = a["condition"] != "normal", v["label"] == "faulty"
    if a_fault and v_fault:
        return "confirmed_fault"
    if a_fault != v_fault:
        return "conflict"
    return "healthy"


def rule_decision(sit):
    kind = sit["agreement"]
    a, v = sit.get("analytics") or {}, sit.get("vision") or {}
    if kind == "confirmed_fault":
        sev = URGENCY_TO_SEVERITY.get(a.get("urgency"), "warning")
        sev = "warning" if sev == "info" else sev
        return [
            {"name": "send_alert", "arguments": {"severity": sev, "message":
                f"{FAULT_NAMES.get(a['condition'], a['condition'])} bearing fault (~{a['severity_mil']} mil) confirmed by vibration and thermal vision."}},
            {"name": "create_maintenance_ticket", "arguments": {"priority": a.get("urgency", "medium")
                if a.get("urgency") in ("low", "medium", "high") else "medium",
                "action": a.get("recommended_action", "Inspect bearing."), "within": a.get("window", "-")}},
        ]
    if kind == "conflict":
        return [
            {"name": "send_alert", "arguments": {"severity": "warning", "message":
                f"Sensors disagree: vibration says {a.get('condition')}, thermal vision says {v.get('label')}."}},
            {"name": "create_maintenance_ticket", "arguments": {"priority": "low",
                "action": "Manual inspection to resolve conflicting sensor readings.", "within": "next routine check"}},
        ]
    if kind == "stale":
        return [{"name": "send_alert", "arguments": {"severity": "warning",
                 "message": "Fresh sensor data is missing; monitoring may be blind."}}]
    return [{"name": "no_action", "arguments": {"reason": "Motor healthy; both sensors agree."}}]


NUM = re.compile(r"\d+(?:\.\d+)?")
KEYWORDS = {"inner": {"inner"}, "outer": {"outer"}, "ball": {"ball", "rolling"}}


def _numbers(*values):
    return {float(n) for v in values for n in NUM.findall(str(v))}


def grounded(text, sit):
    """True if every number, fault type and urgency word in `text` is backed by the sensor report.
    Small models paraphrase freely and sometimes invent figures ("within 2 weeks" for a critical fault)."""
    a, v = sit.get("analytics") or {}, sit.get("vision") or {}
    allowed = _numbers(a.get("confidence"), a.get("severity_mil"), a.get("anomaly_score"),
                       a.get("recommended_action"), a.get("window"), v.get("p_faulty"), v.get("frames"))
    clean = re.sub(r"(?i)motor\s*1|line\s*1", "", text)
    for n in NUM.findall(clean):
        if not any(abs(float(n) - y) <= 0.06 for y in allowed):
            return False
    low = text.lower()
    if "no action" in low:      # an alert or ticket that says "no action required" contradicts itself
        return False
    for cond, words in KEYWORDS.items():
        if cond != a.get("condition") and any(re.search(rf"\b{w}\b", low) for w in words):
            return False
    m = re.search(r"urgency\s+(none|low|medium|high)", low)
    return not (m and m.group(1) != a.get("urgency"))


def apply_guardrails(llm_calls, sit):
    """Returns (calls, notes). Required actions come from rule_decision."""
    required = rule_decision(sit)
    notes = []
    req_has_action = any(c["name"] != "no_action" for c in required)
    if not req_has_action:
        # Sensors show nothing wrong: drop any alarm the model invented.
        kept = [c for c in llm_calls if c["name"] == "no_action"]
        if len(kept) != len(llm_calls):
            notes.append("dropped alarm not supported by sensors")
        return kept or required, notes
    out = list(llm_calls)
    for req in required:
        mine = [c for c in out if c["name"] == req["name"]]
        if not mine:
            out.append(req); notes.append(f"added required {req['name']}")
        elif req["name"] == "send_alert":
            floor = SEVERITIES.index(req["arguments"]["severity"])
            for c in mine:
                if SEVERITIES.index(c["arguments"]["severity"]) < floor:
                    c["arguments"]["severity"] = req["arguments"]["severity"]
                    notes.append("raised alert severity to required level")
        elif req["name"] == "create_maintenance_ticket":
            order = ["low", "medium", "high"]
            for c in mine:
                if order.index(c["arguments"]["priority"]) < order.index(req["arguments"]["priority"]):
                    c["arguments"]["priority"] = req["arguments"]["priority"]
                    notes.append("raised ticket priority to required level")
    out = [c for c in out if c["name"] != "no_action"]
    by_name = {c["name"]: c["arguments"] for c in required}
    for c in out:
        ref, args = by_name.get(c["name"]), c["arguments"]
        if ref is None:
            continue
        for field in ("message", "action", "within"):
            if field in args and field in ref and not grounded(args[field], sit):
                args[field] = ref[field]
                notes.append(f"replaced ungrounded {c['name']}.{field}")
    return out, notes