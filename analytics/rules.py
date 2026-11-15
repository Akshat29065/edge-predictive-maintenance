"""Prescriptive layer: turns a diagnosis into a recommended maintenance action.

A plain lookup table on purpose: auditable, instant, and what a maintenance engineer would write.
"""

URGENCY = ["none", "low", "medium", "high"]

FAULT_ACTIONS = {
    "inner": "Inner-race defect. Plan bearing replacement; check shaft alignment and lubrication.",
    "outer": "Outer-race defect. Plan bearing replacement; check housing fit and mounting.",
    "ball": "Rolling-element defect. Plan bearing replacement; inspect lubricant for contamination.",
}
# fault size in mil (0.001 in) -> urgency
SEVERITY_URGENCY = {7: "medium", 14: "high", 21: "high"}
WINDOW = {"medium": "within 2 weeks", "high": "at the next stop / immediately"}


def prescribe(condition, severity_mil, anomaly_flag):
    if condition == "normal":
        if anomaly_flag:
            return {"urgency": "low", "action": "Vibration deviates from the healthy baseline but matches no known fault. "
                    "Re-inspect at next routine check.", "window": "next routine check"}
        return {"urgency": "none", "action": "No action. Continue routine monitoring.", "window": "-"}
    urgency = SEVERITY_URGENCY.get(severity_mil, "high")
    return {"urgency": urgency,
            "action": f"{FAULT_ACTIONS[condition]} Estimated fault size {severity_mil} mil.",
            "window": WINDOW[urgency]}