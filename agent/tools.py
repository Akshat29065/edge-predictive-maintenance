"""Equipment-specific functions (skills) the agent can call for motor1.

Every call is validated before it runs, so a model that invents a function or an out-of-range value
cannot do anything. To add a skill: add its schema to TOOLS, a validator and an executor here.
"""
import json
import time
from pathlib import Path

SEVERITIES = ["info", "warning", "critical"]
PRIORITIES = ["low", "medium", "high"]

TOOLS = [
    {"type": "function", "function": {
        "name": "send_alert",
        "description": "Raise an alert for the operators of motor1 (written to the alert log).",
        "parameters": {"type": "object", "properties": {
            "severity": {"type": "string", "enum": SEVERITIES},
            "message": {"type": "string", "description": "What is wrong, under 40 words."}},
            "required": ["severity", "message"]}}},
    {"type": "function", "function": {
        "name": "create_maintenance_ticket",
        "description": "Open a maintenance ticket for motor1.",
        "parameters": {"type": "object", "properties": {
            "priority": {"type": "string", "enum": PRIORITIES},
            "action": {"type": "string", "description": "The maintenance work to carry out."},
            "within": {"type": "string", "description": "Time window, e.g. 'within 2 weeks'."}},
            "required": ["priority", "action"]}}},
    {"type": "function", "function": {
        "name": "no_action",
        "description": "Record that the motor is healthy and nothing needs to be done.",
        "parameters": {"type": "object", "properties": {
            "reason": {"type": "string"}}, "required": ["reason"]}}},
]


def validate(call):
    """Returns a cleaned call dict, or None if the call is not allowed."""
    name, args = call.get("name"), call.get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return None
    if not isinstance(args, dict):
        return None
    text = lambda k, d="": str(args.get(k, d))[:300]
    # A missing severity/priority defaults to a mid value; the policy raises it to the required level anyway.
    # A value that is present but not allowed is still rejected.
    severity, priority = args.get("severity", "warning"), args.get("priority", "medium")
    if name == "send_alert" and severity in SEVERITIES and text("message"):
        return {"name": name, "arguments": {"severity": severity, "message": text("message")}}
    if name == "create_maintenance_ticket" and priority in PRIORITIES and text("action"):
        return {"name": name, "arguments": {"priority": priority, "action": text("action"),
                                            "within": text("within", "-")}}
    if name == "no_action":
        return {"name": name, "arguments": {"reason": text("reason", "healthy")}}
    return None


class Executor:
    def __init__(self, log_dir):
        self.dir = Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _append(self, filename, line):
        with (self.dir / filename).open("a") as f:
            f.write(line + "\n")

    def run(self, call, asset="motor1"):
        a, ts = call["arguments"], time.strftime("%Y-%m-%d %H:%M:%S")
        if call["name"] == "send_alert":
            self._append("alerts.log", f"{ts} [{a['severity'].upper()}] {asset}: {a['message']}")
        elif call["name"] == "create_maintenance_ticket":
            self._append("tickets.log", f"{ts} [{a['priority'].upper()}] {asset}: {a['action']} ({a['within']})")
        # no_action writes nothing on purpose; the decision itself is recorded in actions.log
        return call["name"]