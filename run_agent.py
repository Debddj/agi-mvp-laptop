from agents.agent_loop import run_agent
from agents.planner import plan
from agents.executor import execute
from agents.critic import critique


if __name__ == "__main__":
    run_agent("Calculate 12 * (7 + 3)")
