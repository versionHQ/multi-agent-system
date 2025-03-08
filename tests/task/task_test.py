from unittest.mock import patch
from typing import Callable

from versionhq.agent.model import Agent
from versionhq.llm.model import LLM
from versionhq.task.model import Task, ResponseField, TaskOutput
from versionhq.tool.model import Tool, ToolSet
from versionhq.tool.decorator import tool


def test_store_task_log():
    task = Task(
        description="return the output following the given prompt.",
        response_fields=[ResponseField(title="task_1", data_type=str, required=True),],
    )
    task.execute()

    from versionhq.storage.task_output_storage import TaskOutputStorageHandler
    assert TaskOutputStorageHandler().load() is not None


def test_task_with_agent_tools():
    simple_tool = Tool(name="simple tool", func=lambda x: "simple func")
    agent = Agent(role="demo", goal="execute tools", tools=[simple_tool,], maxit=1, max_tokens=3000)
    task = Task(description="execute tool", can_use_agent_tools=True, tool_res_as_final=True)
    res = task.execute(agent=agent)
    assert res.tool_output == "simple func"

    def empty_func():
        return "empty func"

    func_tool = Tool(name="func tool", func=empty_func)
    agent.tools = [func_tool]
    res = task.execute(agent=agent)
    assert res.tool_output == "empty func"


    def demo_func(message: str) -> str:
        return message + "_demo"

    class CustomTool(Tool):
        name: str = "custom tool"

    custom_tool = CustomTool(func=demo_func)
    agent.tools = [custom_tool]
    res = task.execute(agent=agent)
    assert "_demo" in res.tool_output


def test_task_with_tools():

    def random_func(message: str) -> str:
        return message + "_demo"

    tool = Tool(name="tool", func=random_func)
    tool_set = ToolSet(tool=tool, kwargs=dict(message="empty func"))
    task = Task(description="execute the given tools", tools=[tool_set,], tool_res_as_final=True)
    # res = task.execute()
    # assert res.tool_output == "empty func_demo"

    class CustomTool(Tool):
        name: str = "custom tool"
        func: Callable = None

    custom_tool = CustomTool(func=random_func)
    task.tools = [custom_tool]
    res = task.execute()
    assert "_demo" in res.tool_output

    task.tools = [custom_tool]
    res = task.execute()
    assert res.tool_output is not None


def test_task_without_response_format():
    task = Task(description="return a simple output with any random values.")
    res = task.execute()

    assert res and isinstance(res, TaskOutput)
    assert res.json_dict and isinstance(res.json_dict, dict)
    assert res.pydantic is None


def test_build_agent_without_developer_prompt():
    agent = Agent(
        role="analyst",
        goal="analyze the company's website and retrieve the product overview",
        backstory="You are competitive analysts who have abundand knowledge in marketing, product management.",
        use_developer_prompt=False
    )
    task = Task(description="return a simple output with any random values.")
    res = task.execute(agent=agent)

    assert res and isinstance(res, TaskOutput)
    assert res.json_dict and isinstance(res.json_dict, dict)
    assert res.pydantic is None


def test_task_with_agent_callback():
    import litellm

    def dummy_func(*args, **kwargs) -> str:
        return "Demo func"

    agent = Agent(role="demo", goal="amazing project goal", maxit=1, max_tokens=3000, callbacks=[dummy_func,])
    task = Task(description="Amazing task")
    res = task.execute(agent=agent)

    assert res.raw and res.task_id == task.id
    assert litellm.callbacks == [dummy_func]


def test_rpm():
    agent = Agent(role="demo", goal="use the given tools", max_tokens=3000, max_rpm=3)
    assert agent._rpm_controller and agent._rpm_controller.max_rpm == 3

    a = "hello"
    tool = Tool(func=lambda x: a + x)
    tool_set = ToolSet(tool=tool, kwargs={ "x": "_demo" })
    task = Task(description="Summarize overview of the given tool in sentence, then execute the tool.", tools=[tool_set,])
    res = task.execute(agent=agent)

    assert "hello_demo" in res.raw if res.raw else res
    assert agent._rpm_controller._current_rpm < agent._rpm_controller.max_rpm


def test_maxit():
    @tool
    def demo() -> str:
        """Get the final answer but don't give it yet, just re-use this
        tool non-stop."""
        return "demo"

    agent = Agent(role="demo", goal="amazing demo", maxit=2)
    task = Task(description="Summarize overview of the given tool in sentences.", tools=[demo,])

    with patch.object(LLM, "call", wraps=agent.llm.call) as mock:
        task.execute(agent=agent)
        assert mock.call_count <= 2
