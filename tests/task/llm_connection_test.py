from typing import Any, Callable

from pydantic import BaseModel

from versionhq.agent.model import Agent, LLM
from versionhq.task.model import Task, ResponseField
from versionhq.tool.model import Tool

"""
Test a connection to the llm platform (litellm or custom provider's interface).
"""

llms_to_test = [
    "gpt-4o",
    "gemini/gemini-2.0-flash",
    "openrouter/deepseek/deepseek-r1:free",
    "claude-3-haiku",
    "openrouter/google/gemini-2.0-flash-thinking-exp:free",
    "openrouter/google/gemini-2.0-flash-thinking-exp-1219:free",
    "openrouter/google/gemini-2.0-flash-001",
]


class Demo(BaseModel):
    """
    A demo pydantic class to validate the outcome with various nested data types.
    """
    demo_1: int
    demo_2: float
    demo_3: str
    demo_4: bool
    demo_5: list[str]
    demo_6: dict[str, Any]
    demo_nest_1: list[dict[str, Any]]
    demo_nest_2: list[list[str]]
    demo_nest_3: dict[str, list[str]]
    demo_nest_4: dict[str, dict[str, Any]]


demo_response_fields = [
    ResponseField(title="demo_1", data_type=int),
    ResponseField(title="demo_2", data_type=float),
    ResponseField(title="demo_3", data_type=str),
    ResponseField(title="demo_4", data_type=bool),
    ResponseField(title="demo_5", data_type=list, items=str),
    ResponseField(title="demo_6", data_type=dict, properties=[ResponseField(title="demo-item", data_type=str)]),
    ResponseField(title="demo_nest_1", data_type=list, items=str, properties=([
        ResponseField(title="nest1", data_type=dict, properties=[ResponseField(title="nest11", data_type=str)])
    ])), # you can specify field title of nested items
    ResponseField(title="demo_nest_2", data_type=list, items=list),
    ResponseField(title="demo_nest_3", data_type=dict, properties=[
        ResponseField(title="nest1", data_type=list, items=str)
    ]),
    ResponseField(title="demo_nest_4", data_type=dict, properties=[
        ResponseField(title="nest1", data_type=dict, properties=[ResponseField(title="nest12", data_type=str)])
    ])
]


def set_agent(llm: str) -> Agent:
    agent = Agent(role="demo", goal="demo", llm=llm, maxit=1, max_retry_limit=1, max_tokens=3000)
    assert isinstance(agent.llm, LLM)
    assert agent.llm._init_model_name == llm and agent.llm.provider and agent.llm.max_tokens == agent.max_tokens
    return agent


def simple_task(agent: Agent):
    task = Task(description="write a random poem.")
    res = task.execute(agent=agent, context="We are running a test.")

    assert res.raw and res.tool_output is None and res.callback_output is None


def schema_task(agent: Agent):
    task = Task(description="return random values strictly following the given response format.", pydantic_output=Demo)
    res = task.execute(agent=agent, context="We are running a test.")
    assert [
        getattr(res.pydantic, k) and type(getattr(res.pydantic, k)) == v for k, v in Demo.__annotations__.items()
    ]

def res_field_task(agent: Agent):
    task = Task(description="return random values strictly following the given response format.", response_fields=demo_response_fields)
    res = task.execute(agent=agent, context="We are running a test.")
    assert [k in item.title for item in demo_response_fields for k, v in res.json_dict.items()]


def tool_task(agent: Agent):
    class DemoTool(Tool):
        func: Callable[..., Any] = lambda x: "Demo"

    task = Task(
        description="Simply execute the given tools.",
        tools=[DemoTool,],
        tool_res_as_final=True
    )
    res = task.execute(agent=agent, context="We are running a test.")
    assert res.tool_output and res.raw



# comment out
def _test_connection():
    agents = [set_agent(llm=llm) for llm in llms_to_test]
    for agent in agents:
        simple_task(agent=agent)
        tool_task(agent=agent)
