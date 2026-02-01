import math
import subprocess


def calculator(expression: str):
    try:
        return eval(expression, {"__builtins__": {}, "math": math})
    except Exception as e:
        return f"Error: {e}"


def run_python(code: str):
    try:
        result = subprocess.check_output(
            ["python", "-c", code],
            stderr=subprocess.STDOUT,
            timeout=5,
        )
        return result.decode()
    except Exception as e:
        return str(e)
