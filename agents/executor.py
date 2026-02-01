import re
from agents.tools import calculator, run_python


def extract_math_expression(task: str):
    """
    Extracts a mathematical expression from a task string.
    Example:
    'Calculate 12 * (7 + 3)' -> '12 * (7 + 3)'
    """
    match = re.search(r"([-+*/().0-9\s]+)", task)
    if match:
        return match.group(1).strip()
    return None


def execute(step: str, task: str):
    step_lower = step.lower()

    if "compute" in step_lower or "calculate" in step_lower:
        expression = extract_math_expression(task)
        if expression:
            return calculator(expression)
        return "Error: Could not extract mathematical expression"

    if "python" in step_lower:
        return run_python(task)

    return f"Executed step: {step}"
