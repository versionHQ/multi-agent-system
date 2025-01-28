from versionhq.agent.model import Agent
from versionhq.llm.model import DEFAULT_MODEL_NAME

client_manager = Agent(role="Client Manager", goal="communicate with clients on the task progress", llm=DEFAULT_MODEL_NAME)
client_manager._train()
