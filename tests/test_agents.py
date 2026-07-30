import os
import pytest
from agents.tools import Calculator, safe_eval
from agents.executor import execute, parse_tool_call
from agents.critic import critique
from agents.memory import load_memory, save_memory, get_cached_plan, store_plan


def test_calculator_add():
    calc = Calculator()
    result = calc(expression="2 + 3")
    assert result["success"]
    assert result["result"] == 5


def test_calculator_complex():
    calc = Calculator()
    result = calc(expression="12 * (7 + 3)")
    assert result["success"]
    assert result["result"] == 120


def test_calculator_power():
    calc = Calculator()
    result = calc(expression="2 ** 10")
    assert result["success"]
    assert result["result"] == 1024


def test_calculator_safe_eval_blocked():
    with pytest.raises(ValueError, match="Attribute access not allowed"):
        safe_eval("().__class__")


def test_safe_eval_simple():
    assert safe_eval("2 + 2") == 4
    assert safe_eval("10 / 2") == 5.0
    assert safe_eval("2 ** 8") == 256


def test_parse_tool_call_simple():
    name, args = parse_tool_call("calculator(expression=12 * 5)")
    assert name == "calculator"
    assert args == {"expression": "12 * 5"}


def test_parse_tool_call_invalid():
    name, args = parse_tool_call("No tool call here")
    assert name is None
    assert args is None


def test_execute_math():
    result = execute("Compute the result", "12 * (7 + 3)")
    assert isinstance(result, dict)
    assert result["success"]


def test_critic_success():
    ok, msg = critique({"success": True, "result": 42})
    assert ok


def test_critic_none():
    ok, msg = critique(None)
    assert not ok


def test_critic_dict_failure():
    ok, msg = critique({"success": False, "error": "Something broke"})
    assert not ok


def test_critic_empty_string():
    ok, msg = critique("")
    assert not ok


def test_memory_roundtrip():
    test_task = "test task memory roundtrip"
    test_steps = ["step1", "step2"]

    try:
        store_plan(test_task, test_steps)
        loaded = get_cached_plan(test_task)
        assert loaded == test_steps
    finally:
        mem_path = "agents/plan_memory.json"
        if os.path.exists(mem_path):
            data = load_memory()
            data.pop(test_task, None)
            save_memory(data)
