def plan(task: str):
    """
    Very simple planner:
    converts a task into ordered steps.
    """
    steps = []

    if "calculate" in task.lower():
        steps.append("Identify the mathematical expression")
        steps.append("Compute the result using calculator")
        steps.append("Return the final answer")

    elif "write" in task.lower():
        steps.append("Understand the topic")
        steps.append("Generate structured text")
        steps.append("Review for clarity")

    else:
        steps.append("Analyze the task")
        steps.append("Execute the solution")
        steps.append("Verify the result")

    return steps
