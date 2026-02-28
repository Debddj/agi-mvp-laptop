import re
from agents.tools import calculator, run_python


def extract_math_expression(task: str):
    match = re.search(r"([-+*/().0-9\s]+)", task)
    if match:
        return match.group(1).strip()
    return None


def infer_intent(step: str):
    step = step.lower()

    if any(k in step for k in ["compute", "calculate", "solve", "execute", "result"]):
        return "math"

    if "python" in step:
        return "python"

    return "noop"


def execute(step: str, task: str):
    intent = infer_intent(step)

    if intent == "math":
        expression = extract_math_expression(task)
        if expression:
            return calculator(expression)
        return "Error: Could not extract math expression"

    if intent == "python":
        return run_python(task)

    return f"Executed step: {step}"
