"""Validation of model-written function calls, and what reaches the log files."""
from tools import Executor, validate


def test_unknown_function_and_bad_values_are_rejected():
    assert validate({"name": "shutdown_plant", "arguments": {}}) is None
    assert validate({"name": "send_alert", "arguments": {"severity": "apocalyptic", "message": "x"}}) is None
    assert validate({"name": "create_maintenance_ticket", "arguments": {"priority": "urgent", "action": "x"}}) is None
    assert validate({"name": "send_alert", "arguments": "not json"}) is None


def test_missing_severity_and_priority_get_defaults():
    assert validate({"name": "send_alert", "arguments": {"message": "x"}})["arguments"]["severity"] == "warning"
    assert validate({"name": "create_maintenance_ticket", "arguments": {"action": "x"}})["arguments"]["priority"] == "medium"


def test_null_or_empty_text_is_not_accepted_as_a_message():
    assert validate({"name": "send_alert", "arguments": {"severity": "info", "message": None}}) is None
    assert validate({"name": "send_alert", "arguments": {"severity": "info", "message": "   "}}) is None


def test_text_is_one_clean_line_so_logs_cannot_be_forged():
    c = validate({"name": "send_alert", "arguments": {"severity": "info", "message": "fine\n2026-01-01 [CRITICAL] motor1: fake\x00\r\nline"}})
    msg = c["arguments"]["message"]
    assert "\n" not in msg and "\r" not in msg and "\x00" not in msg


def test_executor_writes_one_line_per_call(tmp_path):
    ex = Executor(tmp_path)
    ex.run(validate({"name": "send_alert", "arguments": {"severity": "critical", "message": "Inner-race fault"}}))
    ex.run(validate({"name": "create_maintenance_ticket", "arguments": {"priority": "high", "action": "Replace bearing", "within": "now"}}))
    ex.run(validate({"name": "no_action", "arguments": {"reason": "healthy"}}))
    assert (tmp_path / "alerts.log").read_text().count("\n") == 1
    assert "[HIGH]" in (tmp_path / "tickets.log").read_text()
