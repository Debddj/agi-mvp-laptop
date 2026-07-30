from agents.critic import critique
from agents.executor import execute
from agents.llm_planner import llm_plan
from agents.memory import get_cached_plan, store_plan


def run_agent(model, task: str, image_path: str = None, max_iters=3):
    print("\n==========================================")
    print(f"MULTI-MODAL AGI AGENT TASK: {task}")
    if image_path:
        print(f"IMAGE INPUT: {image_path}")
    print("==========================================\n")

    cached = get_cached_plan(task)
    if cached:
        print("[Memory] Using cached plan from episodic memory.")
        steps = cached
    else:
        print("[Planner] Generating step-by-step reasoning plan...")
        steps = llm_plan(model, "tokenizer.model", task, image_input=image_path)
        store_plan(task, steps)

    print("\nREASONING PLAN:")
    for i, step in enumerate(steps, 1):
        print(f"  {i}. {step}")

    execution_history = []

    for step in steps:
        for attempt in range(max_iters):
            output = execute(step, task, image_path=image_path)
            success, feedback = critique(output)

            print(f"\n-> STEP: {step}")
            print(f"   OUTPUT: {output}")
            print(f"   CRITIC: {feedback}")

            execution_history.append((step, output, success))

            if success:
                break

    print("\nTask execution finished successfully.\n")
    return execution_history
