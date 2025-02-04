from typing import Any, Callable

from versionhq.agent.model import Agent, LLM
from versionhq.task.model import Task
from versionhq.tool.model import Tool

from tests.task import DemoOutcomeNoNest, demo_response_fields

"""
Test a connection to the llm platform (litellm or custom provider's interface).
"""

llms_to_test = [
    # "gpt-4o",
    # "gemini/gemini-2.0-flash",
    "openrouter/deepseek/deepseek-r1:free",
    "claude-3-haiku",
]

def set_agent(llm: str) -> Agent:
    agent = Agent(role="demo", goal="demo", llm=llm, maxit=1, max_retry_limit=1, max_tokens=3000)
    assert isinstance(agent.llm, LLM)
    assert agent.llm._init_model_name == llm and agent.llm.provider and agent.llm.max_tokens == agent.max_tokens
    return agent


def simple_task(agent: Agent):
    task = Task(description="write a random poem.")
    res = task.execute_sync(agent=agent, context="We are running a test.")

    assert res.raw and res.tool_output is None and res.callback_output is None


def schema_task(agent: Agent):
    task = Task(description="return random values strictly following the given response format.", pydantic_output=DemoOutcomeNoNest)
    res = task.execute_sync(agent=agent, context="We are running a test.")
    assert [
        getattr(res.pydantic, k) and type(getattr(res.pydantic, k)) == v for k, v in DemoOutcomeNoNest.__annotations__.items()
    ]

def res_field_task(agent: Agent):
    task = Task(description="return random values strictly following the given response format.", response_fields=demo_response_fields)
    res = task.execute_sync(agent=agent, context="We are running a test.")
    assert [k in item.title for item in demo_response_fields for k, v in res.json_dict.items()]


def tool_task(agent: Agent):
    class DemoTool(Tool):
        func: Callable[..., Any] = lambda x: "Demo"

    task = Task(
        description="Simply execute the given tools.",
        tools=[DemoTool,],
        tool_res_as_final=True
    )
    res = task.execute_sync(agent=agent, context="We are running a test.")
    assert res.tool_output and res.raw



def _test_connection():
    """
    comment out
    """

    agents = [set_agent(llm=llm) for llm in llms_to_test]
    for agent in agents:
        simple_task(agent=agent)
        tool_task(agent=agent)
