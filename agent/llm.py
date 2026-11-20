"""Minimal Ollama client (standard library only). Default 'json' mode asks for schema-constrained JSON
function calls; 'tools' mode uses Ollama's native tool calling."""
import json
import re
import urllib.request

from tools import PRIORITIES, SEVERITIES, TOOLS


class LLMError(Exception):
    pass


def parse_text_calls(content):
    """Small models sometimes print the call as text (<tool>{...}</tool>) instead of using the tool API."""
    calls = []
    for chunk in re.findall(r"\{.*\}", content or "", re.S):
        try:
            obj = json.loads(chunk)
        except ValueError:
            continue
        if isinstance(obj, dict) and "name" in obj:
            calls.append({"name": obj["name"], "arguments": obj.get("arguments") or obj.get("parameters") or {}})
    return calls


# "json" mode: the model must answer with JSON matching this schema (grammar-constrained decoding),
# so it cannot ramble or "think" before answering. Each item is one function call.
CALL_SCHEMA = {
    "type": "object",
    "properties": {"calls": {"type": "array", "minItems": 1, "maxItems": 2, "items": {
        "type": "object",
        "properties": {
            "name": {"enum": [t["function"]["name"] for t in TOOLS]},
            "severity": {"enum": SEVERITIES}, "priority": {"enum": PRIORITIES},
            "message": {"type": "string"}, "action": {"type": "string"},
            "within": {"type": "string"}, "reason": {"type": "string"}},
        "required": ["name"]}}},
    "required": ["calls"],
}
JSON_HELP = """
Reply with JSON only: {"calls": [ ... ]}. Each call has "name" plus that function's arguments:
- send_alert: severity (info|warning|critical), message
- create_maintenance_ticket: priority (low|medium|high), action, within
- no_action: reason
"""


class OllamaClient:
    def __init__(self, url, system_prompt, timeout_s=30, mode="json"):
        self.url, self.system, self.timeout, self.mode = url.rstrip("/"), system_prompt, timeout_s, mode

    def decide(self, model, situation):
        json_mode = self.mode == "json"
        body = {
            "model": model, "stream": False, "think": False, "keep_alive": "30m",
            "messages": [{"role": "system", "content": self.system + (JSON_HELP if json_mode else "")},
                         {"role": "user", "content": json.dumps(situation) + " /no_think"}],
            "options": {"temperature": 0, "num_ctx": 2048, "num_predict": 400},
        }
        if json_mode:
            body["format"] = CALL_SCHEMA
        else:
            body["tools"] = TOOLS
        req = urllib.request.Request(f"{self.url}/api/chat", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.load(resp)
        except Exception as exc:                       # timeout, refused, 404 model missing, bad JSON
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc
        msg = data.get("message") or {}
        if json_mode:
            try:
                items = json.loads(msg.get("content") or "")["calls"]
                return [{"name": i["name"], "arguments": {k: v for k, v in i.items() if k != "name" and v not in ("", None)}}
                        for i in items]
            except (ValueError, KeyError, TypeError) as exc:
                raise LLMError(f"bad JSON answer ({exc}): {str(msg.get('content'))[:160]!r}") from exc
        calls = msg.get("tool_calls") or []
        if not calls:
            calls = [{"function": c} for c in parse_text_calls(msg.get("content"))]
        if not calls:
            raise LLMError(f"no tool call; model said: {str(msg.get('content'))[:160]!r}")
        return [c["function"] for c in calls if "function" in c]