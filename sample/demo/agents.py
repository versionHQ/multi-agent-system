import os
from dotenv import load_dotenv
from framework.agent.model import Agent

load_dotenv(override=True)
MODEL_NAME = os.environ.get("LITELLM_MODEL_NAME", "gpt-3.5-turbo")

agent_a = Agent(
    role="Demo Agent A",
    goal="""My amazing goals""",
    backstory="My amazing backstory",
    verbose=True,
    llm=MODEL_NAME,
    max_token=3000
)

agent_b1 =  Agent(
    role="Demo Agent B-1",
    goal="""My amazing goals""",
    backstory="My amazing backstory",
    verbose=True,
    llm=MODEL_NAME,
    max_token=3000
)
