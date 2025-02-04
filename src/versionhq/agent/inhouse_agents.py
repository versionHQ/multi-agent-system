from versionhq.agent.model import Agent
from versionhq.llm.model import DEFAULT_MODEL_NAME

"""
In-house agents to be called across the project.
[Rules] agents' names and roles start with `vhq_`.
"""

vhq_client_manager = Agent(
    role="vhq-Client Manager",
    goal="Efficiently communicate with the client on the task progress",
    llm=DEFAULT_MODEL_NAME
)

vhq_task_evaluator = Agent(
    role="vhq-Task Evaluator",
    goal="score the output according to the given evaluation criteria.",
    llm=DEFAULT_MODEL_NAME,
    llm_config=dict(top_p=0.8, top_k=30, max_tokens=5000, temperature=0.9),
    maxit=1,
    max_retry_limit=1
)

vhq_formation_planner = Agent(
    role="vhq-Formation Planner",
    goal="Plan a formation of agents based on the given task descirption.",
    llm="gemini/gemini-2.0-flash-exp",
    llm_config=dict(top_p=0.8, top_k=30, temperature=0.9),
    maxit=1,
    max_retry_limit=1
)
