def critique(output):
    if output is None:
        return False, "No output produced"

    if isinstance(output, dict):
        if not output.get("success"):
            return False, output.get("error", "Execution failed")
        result = output.get("result")
        if result is None:
            return False, "No result returned"
        return True, "Step completed successfully"

    if isinstance(output, str):
        if "error" in output.lower():
            return False, "Execution error detected"
        if not output.strip():
            return False, "Empty output"
        return True, "Output looks acceptable"

    return True, "Step completed"
