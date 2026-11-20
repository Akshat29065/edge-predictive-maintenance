"""Deterministic policy: classifies the situation, and decides what MUST happen.

It is the fallback when the LLM is unavailable, and a guardrail around the LLM's choices:
the model may add detail, but it cannot skip a required alert or invent one the sensors don't support.
"""
import re

FAULT_NAMES = {"inner": "inner-race", "outer": "outer-race", "ball": "rolling-element (ball)"}
URGENCY_TO_SEVERITY = {"none": "info", "low": "info", "medium": "warning", "high": "critical"}


def classify(sit, grace_over=True):
    """healthy | confirmed_fault | conflict | stale | unknown

    Missing or old data means `stale` (operators must hear about a dead sensor). During the start-up
    grace period (`grace_over` False) it is `unknown` instead: rows left in InfluxDB from a previous run
    are old, and the services are still warming up, so an alert would only be noise."""
    a, v = sit.get("analytics"), sit.get("vision")
    if not a or not v or a["stale"] or v["stale"]:
        return "stale" if grace_over else "unknown"
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
    """Returns (calls, notes). The rule table decides WHAT must happen; the model words the alert.

    - Only the functions the situation requires are kept (an invented alarm or ticket is dropped).
    - Alert severity and ticket priority are fixed to the rule-table value, so the model can neither
      play a fault down nor escalate it (a 7 mil fault once came back as a HIGH ticket).
    - Ticket action and time window always come from the rule table (the model once garbled them).
    - The alert message is the model's own words, kept only if every number and fault type in it is
      backed by the sensor report; otherwise the rule-table message is used.
    - A required function the model left out is added; duplicates are removed.
    """
    required = rule_decision(sit)
    notes = []
    if all(c["name"] == "no_action" for c in required):
        # Sensors show nothing wrong: drop any alarm the model invented.
        kept = [c for c in llm_calls if c["name"] == "no_action"]
        if len(kept) != len(llm_calls):
            notes.append("dropped alarm not supported by sensors")
        return kept[:1] or required, notes
    out = []
    for req in required:
        mine = [c for c in llm_calls if c["name"] == req["name"]]
        if not mine:
            out.append(req)
            notes.append(f"added required {req['name']}")
            continue
        c = mine[0]
        if len(mine) > 1:
            notes.append(f"removed duplicate {req['name']}")
        args = c["arguments"]
        if req["name"] == "send_alert":
            if args["severity"] != req["arguments"]["severity"]:
                args["severity"] = req["arguments"]["severity"]
                notes.append("set alert severity to required level")
            if not grounded(args["message"], sit):
                args["message"] = req["arguments"]["message"]
                notes.append("replaced ungrounded send_alert.message")
        else:  # create_maintenance_ticket
            if args["priority"] != req["arguments"]["priority"]:
                args["priority"] = req["arguments"]["priority"]
                notes.append("set ticket priority to required level")
            if args.get("action") != req["arguments"]["action"] or args.get("within") != req["arguments"]["within"]:
                args["action"], args["within"] = req["arguments"]["action"], req["arguments"]["within"]
                notes.append("used rule-table ticket text")
        out.append(c)
    names = {c["name"] for c in out}
    if any(c["name"] not in names for c in llm_calls):
        notes.append("dropped function not required by the situation")
    return out, notes
