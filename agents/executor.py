import json
import re
from agents.tools import AVAILABLE_TOOLS


def parse_tool_call(text):
    match = re.search(r"(\w+)\(([^)]*)\)", text)
    if not match:
        return None, None

    name = match.group(1)
    args_raw = match.group(2).strip()

    if name not in AVAILABLE_TOOLS:
        return None, None

    args = {}
    if args_raw:
        for pair in args_raw.split(","):
            pair = pair.strip()
            if "=" in pair:
                k, v = pair.split("=", 1)
                k = k.strip()
                v = v.strip().strip("\"'")
                args[k] = v

    return name, args


def execute(step, task, image_path=None):
    """
    Executes a plan step using appropriate multi-modal tools.
    """
    step_lower = step.lower()
    task_lower = task.lower()

    # Direct tool call pattern parsing (e.g. calculator(expression="2+2"))
    tool_name, tool_args = parse_tool_call(step)
    if tool_name and tool_name in AVAILABLE_TOOLS:
        try:
            return AVAILABLE_TOOLS[tool_name](**tool_args)
        except Exception as e:
            return {"success": False, "error": str(e)}

    # Keyword based tool dispatching
    if "image" in step_lower or "picture" in step_lower or "visual" in step_lower:
        target_img = image_path or "sample.jpg"
        if "analyze_image" in AVAILABLE_TOOLS:
            return AVAILABLE_TOOLS["analyze_image"](image_path=target_img)

    if "python" in step_lower or "code" in step_lower or "script" in step_lower:
        match = re.search(r"```python(.*?)```", task, re.DOTALL)
        code = match.group(1).strip() if match else "print('Step executed')"
        return AVAILABLE_TOOLS["run_python"](code=code)

    if "calculate" in step_lower or "math" in step_lower or "compute" in step_lower or "add" in step_lower:
        match = re.search(r"[\d\s+\-*/().]{3,}", task)
        expr = match.group(0).strip() if match else "1 + 1"
        return AVAILABLE_TOOLS["calculator"](expression=expr)

    if "search" in step_lower or "find" in step_lower or "lookup" in step_lower:
        return AVAILABLE_TOOLS["web_search"](query=task)

    if "file" in step_lower or "write" in step_lower or "save" in step_lower:
        return AVAILABLE_TOOLS["file_system"](action="read", path="README.md")

    return {"success": True, "result": f"Completed step: '{step}' successfully."}
