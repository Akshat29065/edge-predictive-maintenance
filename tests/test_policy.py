"""The agent's safety net: classification, rule decisions and the guardrails around the LLM."""
import policy
from tools import validate


def analytics(cond="inner", sev=21, urgency="high", conf=0.99, stale=False, action="Inner-race defect. Plan bearing replacement.", window="at the next stop / immediately"):
    return {"condition": cond, "confidence": conf, "severity_mil": sev, "anomaly": cond != "normal", "anomaly_score": 5.0,
            "urgency": urgency, "recommended_action": action, "window": window, "stale": stale}


def vision(label="faulty", stale=False):
    return {"label": label, "p_faulty": 0.99 if label == "faulty" else 0.01, "frames": 3, "stale": stale}


def sit(a=None, v=None, grace_over=True):
    s = {"asset": "motor1", "analytics": a, "vision": v}
    s["agreement"] = policy.classify(s, grace_over)
    return s


def call(name, **args):
    return validate({"name": name, "arguments": args})


# ---- classify ----
def test_classify_all_states():
    assert sit(analytics("normal", 0, "none"), vision("healthy"))["agreement"] == "healthy"
    assert sit(analytics(), vision("faulty"))["agreement"] == "confirmed_fault"
    assert sit(analytics(), vision("healthy"))["agreement"] == "conflict"
    assert sit(analytics("normal", 0, "none"), vision("faulty"))["agreement"] == "conflict"
    assert sit(analytics(stale=True), vision())["agreement"] == "stale"
    assert sit(analytics(), vision(stale=True))["agreement"] == "stale"


def test_missing_sensor_is_stale_after_grace_and_unknown_during_it():
    assert sit(None, vision())["agreement"] == "stale"
    assert sit(analytics(), None)["agreement"] == "stale"
    assert sit(None, None, grace_over=False)["agreement"] == "unknown"
    assert sit(analytics(stale=True), vision(), grace_over=False)["agreement"] == "unknown"


# ---- rule_decision ----
def test_rule_decision_confirmed_fault():
    calls = policy.rule_decision(sit(analytics(), vision()))
    alert, ticket = calls
    assert alert["name"] == "send_alert" and alert["arguments"]["severity"] == "critical"
    assert ticket["name"] == "create_maintenance_ticket" and ticket["arguments"]["priority"] == "high"


def test_rule_decision_medium_fault_is_a_warning():
    s = sit(analytics(sev=7, urgency="medium", window="within 2 weeks"), vision())
    alert, ticket = policy.rule_decision(s)
    assert alert["arguments"]["severity"] == "warning" and ticket["arguments"]["priority"] == "medium"


def test_rule_decision_conflict_and_stale_and_healthy():
    c = policy.rule_decision(sit(analytics(), vision("healthy")))
    assert [x["name"] for x in c] == ["send_alert", "create_maintenance_ticket"] and c[1]["arguments"]["priority"] == "low"
    assert [x["name"] for x in policy.rule_decision(sit(None, vision()))] == ["send_alert"]
    assert [x["name"] for x in policy.rule_decision(sit(analytics("normal", 0, "none"), vision("healthy")))] == ["no_action"]


# ---- grounded ----
def test_grounded_accepts_report_facts_and_rejects_inventions():
    s = sit(analytics(), vision())
    assert policy.grounded("Inner-race fault, about 21 mil, confirmed by thermal vision.", s)
    assert not policy.grounded("Replace within 2 weeks.", s)             # 2 is not in the report
    assert not policy.grounded("Outer-race fault detected.", s)          # wrong fault type
    assert not policy.grounded("No action required.", s)                 # contradicts an alert
    assert not policy.grounded("Urgency low for this fault.", s)         # wrong urgency


# ---- guardrails ----
def test_guardrails_drop_invented_alarm_when_healthy():
    s = sit(analytics("normal", 0, "none"), vision("healthy"))
    out, notes = policy.apply_guardrails([call("send_alert", severity="critical", message="Bearing failing")], s)
    assert [c["name"] for c in out] == ["no_action"] and notes


def test_guardrails_add_missing_required_calls():
    s = sit(analytics(), vision())
    out, notes = policy.apply_guardrails([call("no_action", reason="ok")], s)
    assert [c["name"] for c in out] == ["send_alert", "create_maintenance_ticket"]
    assert any("added required" in n for n in notes)


def test_guardrails_fix_severity_and_priority_both_ways():
    low = sit(analytics(sev=7, urgency="medium", window="within 2 weeks"), vision())
    llm = [call("send_alert", severity="critical", message="Inner-race fault, 7 mil."),
           call("create_maintenance_ticket", priority="high", action="x", within="y")]
    out, _ = policy.apply_guardrails(llm, low)                 # model over-escalates: capped
    assert out[0]["arguments"]["severity"] == "warning" and out[1]["arguments"]["priority"] == "medium"
    high = sit(analytics(), vision())
    llm = [call("send_alert", severity="info", message="Inner-race fault, 21 mil."),
           call("create_maintenance_ticket", priority="low", action="x", within="y")]
    out, _ = policy.apply_guardrails(llm, high)                # model plays it down: raised
    assert out[0]["arguments"]["severity"] == "critical" and out[1]["arguments"]["priority"] == "high"


def test_guardrails_ticket_text_comes_from_the_rule_table():
    s = sit(analytics(), vision())
    llm = [call("send_alert", severity="critical", message="Inner-race fault, 21 mil."),
           call("create_maintenance_ticket", priority="high", action="check shaft alignment and lubric: lubrication", within="2 weeks")]
    out, notes = policy.apply_guardrails(llm, s)
    ticket = out[1]["arguments"]
    assert ticket["action"] == s["analytics"]["recommended_action"] and ticket["within"] == s["analytics"]["window"]
    assert "used rule-table ticket text" in notes


def test_guardrails_replace_ungrounded_message_and_keep_grounded_one():
    s = sit(analytics(), vision())
    bad = [call("send_alert", severity="critical", message="Replace bearing within 2 weeks")]
    out, notes = policy.apply_guardrails(bad, s)
    assert "2 weeks" not in out[0]["arguments"]["message"] and "replaced ungrounded send_alert.message" in notes
    good = [call("send_alert", severity="critical", message="Confirmed inner-race fault, 21 mil, both sensors agree.")]
    out, _ = policy.apply_guardrails(good, s)
    assert out[0]["arguments"]["message"].startswith("Confirmed inner-race")


def test_guardrails_drop_duplicates_and_unrequired_functions():
    s = sit(None, vision())                                     # stale: only an alert is required
    llm = [call("send_alert", severity="warning", message="Sensor data missing."),
           call("send_alert", severity="warning", message="Sensor data missing again."),
           call("create_maintenance_ticket", priority="high", action="Replace bearing")]
    out, notes = policy.apply_guardrails(llm, s)
    assert [c["name"] for c in out] == ["send_alert"]
    assert any("duplicate" in n for n in notes) and any("not required" in n for n in notes)
