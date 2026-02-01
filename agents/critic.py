def critique(output):
    if output is None:
        return False, "No output produced"

    if isinstance(output, str) and "error" in output.lower():
        return False, "Execution error detected"

    return True, "Output looks acceptable"
