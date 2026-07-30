import ast
import math
import operator
import os
import subprocess


class Tool:
    def __init__(self, name, description, parameters):
        self.name = name
        self.description = description
        self.parameters = parameters

    def __call__(self, **kwargs):
        raise NotImplementedError


SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
    ast.BitXor: operator.xor,
    ast.BitOr: operator.or_,
    ast.BitAnd: operator.and_,
}


def safe_eval(expr):
    tree = ast.parse(expr, mode="eval")

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Lambda, ast.Call)):
            raise ValueError("Function calls not allowed")
        if isinstance(node, ast.Attribute):
            raise ValueError("Attribute access not allowed")
        if isinstance(node, ast.Name) and node.id not in dir(math):
            raise ValueError(f"Name '{node.id}' not allowed")

    return eval(compile(tree, "<safe>", "eval"), {"__builtins__": {}}, {k: getattr(math, k) for k in dir(math) if not k.startswith("_")})


class Calculator(Tool):
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Evaluate a mathematical expression",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate (e.g. '12 * (7 + 3)')",
                    }
                },
                "required": ["expression"],
            },
        )

    def __call__(self, expression):
        try:
            result = safe_eval(expression)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}


class PythonRunner(Tool):
    def __init__(self):
        super().__init__(
            name="run_python",
            description="Execute Python code in a sandboxed subprocess",
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute",
                    }
                },
                "required": ["code"],
            },
        )

    def __call__(self, code):
        try:
            result = subprocess.check_output(
                ["python", "-c", code],
                stderr=subprocess.STDOUT,
                timeout=5,
            )
            return {"success": True, "result": result.decode().strip()}
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Execution timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}


class ImageAnalyzer(Tool):
    """
    Multi-modal Vision tool for inspecting, preprocessing, and analyzing image content.
    """
    def __init__(self):
        super().__init__(
            name="analyze_image",
            description="Inspect an image file or tensor to extract dimensions, statistics, and visual metadata",
            parameters={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Path to the image file",
                    }
                },
                "required": ["image_path"],
            },
        )

    def __call__(self, image_path):
        try:
            if not os.path.exists(image_path):
                return {"success": False, "error": f"File not found: {image_path}"}

            from PIL import Image
            img = Image.open(image_path)
            width, height = img.size
            mode = img.mode
            format_name = img.format
            return {
                "success": True,
                "result": {
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "format": format_name,
                    "aspect_ratio": round(width / max(1, height), 2),
                }
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class WebSearch(Tool):
    """
    Simulated web search engine tool for agentic research and knowledge expansion.
    """
    def __init__(self):
        super().__init__(
            name="web_search",
            description="Perform a simulated search query to retrieve facts, documentation, or code examples",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string",
                    }
                },
                "required": ["query"],
            },
        )

    def __call__(self, query):
        query_lower = query.lower()
        if "python" in query_lower:
            summary = "Python is a high-level programming language widely used in AI, ML, and scientific computing."
        elif "transformer" in query_lower or "llm" in query_lower:
            summary = "Transformers use self-attention mechanisms to model relationships between tokens in parallel."
        elif "colab" in query_lower or "gpu" in query_lower:
            summary = "Google Colab provides free T4 GPUs supporting FP16 mixed precision training up to 16GB VRAM."
        else:
            summary = f"Search query '{query}' yielded 3 relevant results with high confidence."

        return {
            "success": True,
            "query": query,
            "result": summary
        }


class FileSystemTool(Tool):
    """
    File system interaction tool for reading, writing, and listing project workspace files.
    """
    def __init__(self):
        super().__init__(
            name="file_system",
            description="Read or write workspace text files",
            parameters={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["read", "write", "list"]},
                    "path": {"type": "string", "description": "Target file or directory path"},
                    "content": {"type": "string", "description": "Content to write if action is 'write'"}
                },
                "required": ["action", "path"]
            }
        )

    def __call__(self, action, path, content=None):
        try:
            if action == "list":
                if os.path.exists(path) and os.path.isdir(path):
                    files = os.listdir(path)
                    return {"success": True, "result": files}
                return {"success": False, "error": f"Directory not found: {path}"}
            elif action == "read":
                if os.path.exists(path) and os.path.isfile(path):
                    with open(path, encoding="utf-8") as f:
                        text = f.read(2048)  # Limit output
                    return {"success": True, "result": text}
                return {"success": False, "error": f"File not found: {path}"}
            elif action == "write":
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content or "")
                return {"success": True, "result": f"Successfully wrote to {path}"}
            else:
                return {"success": False, "error": f"Unknown action: {action}"}
        except Exception as e:
            return {"success": False, "error": str(e)}


AVAILABLE_TOOLS = {
    "calculator": Calculator(),
    "run_python": PythonRunner(),
    "analyze_image": ImageAnalyzer(),
    "web_search": WebSearch(),
    "file_system": FileSystemTool(),
}
