from versionhq.agent.model import Agent
from versionhq.llm.model import DEFAULT_MODEL_NAME

"""
List up agents to be called across the project.
"""

client_manager = Agent(role="Client Manager", goal="communicate with clients on the task progress", llm=DEFAULT_MODEL_NAME)

task_evaluator = Agent(role="Task Evaluator", goal="score the output according to the given evaluation criteria.", llm=DEFAULT_MODEL_NAME)
