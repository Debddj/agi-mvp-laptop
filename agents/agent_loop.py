from agents.planner import plan
from agents.executor import execute
from agents.critic import critique


def run_agent(task: str, max_iters=3):
    print(f"\nTASK: {task}\n")

    steps = plan(task)
    print("PLAN:")
    for i, step in enumerate(steps, 1):
        print(f"{i}. {step}")

    memory = []

    for step in steps:
        for attempt in range(max_iters):
            output = execute(step, task)
            success, feedback = critique(output)

            print(f"\nSTEP: {step}")
            print(f"OUTPUT: {output}")
            print(f"CRITIC: {feedback}")

            memory.append((step, output, success))

            if success:
                break

    return memory
