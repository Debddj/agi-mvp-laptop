import json
import os

MEMORY_FILE = "agents/plan_memory.json"


def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return {}
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)


def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)


def get_cached_plan(task):
    memory = load_memory()
    return memory.get(task)


def store_plan(task, steps):
    memory = load_memory()
    memory[task] = steps
    save_memory(memory)