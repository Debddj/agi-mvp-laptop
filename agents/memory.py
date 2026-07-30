import json
import os
import re

MEMORY_FILE = "agents/plan_memory.json"


def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return {}
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_memory(memory):
    try:
        os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(memory, f, indent=2)
    except Exception as e:
        print(f"Memory save error: {e}")


def _tokenize_text(text):
    words = re.findall(r"\w+", text.lower())
    return set(words)


def find_similar_task(task, similarity_threshold=0.6):
    """
    Episodic memory lookup using Jaccard text similarity to find prior solved plans.
    """
    memory = load_memory()
    task_words = _tokenize_text(task)
    if not task_words or not memory:
        return None

    best_match = None
    best_score = 0.0

    for stored_task, steps in memory.items():
        stored_words = _tokenize_text(stored_task)
        if not stored_words:
            continue
        intersection = task_words.intersection(stored_words)
        union = task_words.union(stored_words)
        similarity = len(intersection) / len(union) if union else 0.0

        if similarity > best_score and similarity >= similarity_threshold:
            best_score = similarity
            best_match = steps

    return best_match


def get_cached_plan(task):
    memory = load_memory()
    if task in memory:
        return memory[task]
    return find_similar_task(task)


def store_plan(task, steps):
    memory = load_memory()
    memory[task] = steps
    save_memory(memory)