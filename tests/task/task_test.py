import os
import pytest
from unittest.mock import patch
from typing import Dict, Any, List, Optional, Callable

from pydantic import BaseModel, Field, InstanceOf

from versionhq.agent.model import Agent
from versionhq.task.model import Task, ResponseField, TaskOutput, ConditionalTask
from versionhq.tool.model import Tool, ToolSet
from tests.task import DemoOutcome, demo_response_fields, base_agent



def test_sync_execute_task_with_pydantic_outcome():
    task = Task(
        description="Output random values strictly following the data type defined in the given response format.",
        pydantic_custom_output=DemoOutcome
    )
    res = task.execute_sync(agent=base_agent)

    assert isinstance(res, TaskOutput) and res.task_id is task.id
    assert isinstance(res.raw, str) and isinstance(res.json_dict, dict)
    assert [hasattr(res.pydantic, k) and getattr(res.pydantic, k) == v for k, v in res.json_dict.items()]


def test_sync_execute_task_with_json_dict():
    task = Task(
        description="Output random values strictly following the data type defined in the given response format.",
        response_fields=demo_response_fields
    )
    res = task.execute_sync(agent=base_agent)

    assert isinstance(res, TaskOutput) and res.task_id is task.id
    assert res.raw and isinstance(res.raw, str)
    assert res.pydantic is None
    assert res.json_dict and isinstance(res.json_dict, dict)
    assert [v and type(v) == task.response_fields[i].data_type for i, (k, v) in enumerate(res.json_dict.items())]


def test_async_execute_task():
    task = Task(description="Return string: 'test'")

    with patch.object(Agent, "execute_task", return_value="test") as execute:
        execution = task.execute_async(agent=base_agent)
        result = execution.result()
        assert result.raw == "test"
        execute.assert_called_once_with(task=task, context=None, task_tools=list())


def test_sync_execute_with_task_context():
    """
    Use case = One agent handling multiple tasks sequentially using context set in the main task.
    """
    sub_task = Task(
        description="return the output following the given prompt.",
        response_fields=[
            ResponseField(title="subtask_result", data_type=str, required=True),
        ]
    )
    main_task = Task(
        description="return the output following the given prompt.",
        response_fields=[
            ResponseField(title="test1", data_type=int, required=True),
            ResponseField(title="test2", data_type=str, required=True),
        ],
        context=[sub_task,]
    )
    res = main_task.execute_sync(agent=base_agent)

    assert isinstance(res, TaskOutput)
    assert res.task_id is main_task.id
    assert res.raw is not None
    assert isinstance(res.raw, str)
    assert res.json_dict is not None
    assert isinstance(res.json_dict, dict)
    assert res.pydantic is None
    assert sub_task.output is not None
    assert sub_task.output.json_dict is not None
    assert "subtask_result" in main_task.prompt(model_provider=base_agent.llm.provider)


def test_sync_execute_task_with_prompt_context():
    """
    Use case:
    - One agent handling multiple tasks sequentially using context set in the main task.
    - On top of that, the agent receives context when they execute the task.
    """
    class Outcome(BaseModel):
        test1: int = Field(default=None)
        test2: str = Field(default=None)

    sub_task = Task(
        description="return the output following the given prompt.",
        response_fields=[ResponseField(title="result", data_type=str, required=True),]
    )
    main_task = Task(
        description="return the output following the given prompt.",
        pydantic_custom_output=Outcome,
        response_fields=[
            ResponseField(title="test1", data_type=int, required=True),
            ResponseField(title="test2", data_type=str, required=True),
        ],
        context=[sub_task]
    )
    res = main_task.execute_sync(agent=base_agent, context="plan a Black Friday campaign.")

    assert isinstance(res, TaskOutput) and res.task_id is main_task.id
    assert res.raw and isinstance(res.raw, str)
    assert res.json_dict and isinstance(res.json_dict, dict)
    assert res.pydantic.test1 == res.json_dict["test1"] and res.pydantic.test2 == res.json_dict["test2"]
    assert sub_task.output is not None
    assert "result" in main_task.prompt(model_provider=base_agent.llm.provider)
    assert main_task.prompt_context == "plan a Black Friday campaign."
    assert "plan a Black Friday campaign." in main_task.prompt(model_provider=base_agent.llm.provider)


def test_callback():
    """
    See if the callback function is executed well with kwargs.
    """

    def callback_func(kwargs: Dict[str, Any]):
        task_id = kwargs.get("task_id", None)
        added_condition = kwargs.get("added_condition", None)
        assert task_id is not None
        assert added_condition is not None
        return f"Result: {task_id}, condition added: {added_condition}"

    task = Task(
        description="return the output following the given prompt.",
        response_fields=[
            ResponseField(title="test1", data_type=str, required=True),
        ],
        callback=callback_func,
        callback_kwargs=dict(added_condition="demo for pytest")
    )
    res = task.execute_sync(agent=base_agent)

    assert res is not None
    assert isinstance(res, TaskOutput)
    assert res.task_id is task.id
    assert res.raw is not None


def test_delegate():
    agent = Agent(role="demo agent 6", goal="My amazing goals")
    task = Task(
        description="return the output following the given prompt.",
        response_fields=[
            ResponseField(title="test1", data_type=str, required=True),
        ],
        allow_delegation=True
    )
    task.execute_sync(agent=agent)

    assert task.output is not None
    assert "delegated_agent" in task.processed_by_agents
    assert task.delegations != 0


def test_conditional_task():
    task = Task(
        description="erturn the output following the given prompt.",
        response_fields=[ResponseField(title="test1", data_type=str, required=True),],
    )
    res = task.execute_sync(agent=base_agent)

    conditional_task = ConditionalTask(
        description="return the output following the given prompt.",
        response_fields=[ResponseField(title="test1", data_type=str, required=True),],
        condition=lambda x: bool("zzz" in task.output.raw)
    )
    should_execute = conditional_task.should_execute(context=res)

    assert res.raw is not None
    assert should_execute is False

    conditional_res = conditional_task._handle_conditional_task(task_outputs=[res,], task_index=1, was_replayed=False)
    if not should_execute:
        assert conditional_res is None
    else:
        assert conditional_res.task_id is conditional_task.id


def test_store_task_log():
    task = Task(
        description="return the output following the given prompt.",
        response_fields=[ResponseField(title="task_1", data_type=str, required=True),],
    )
    assert task._task_output_handler.load() is not None


def test_task_with_agent_tools():
    simple_tool = Tool(name="simple tool", func=lambda x: "simple func")
    agent = Agent(role="demo", goal="execute tools", tools=[simple_tool,])
    task = Task(description="execute tool", can_use_agent_tools=True, tool_res_as_final=True)
    res = task.execute_sync(agent=agent)
    assert res.tool_output == "simple func"

    def empty_func():
        return "empty func"

    func_tool = Tool(name="func tool", func=empty_func)
    agent.tools = [func_tool]
    res = task.execute_sync(agent=agent)
    assert res.tool_output == "empty func"


    def demo_func(message: str) -> str:
        return message + "_demo"

    class CustomTool(Tool):
        name: str = "custom tool"

    custom_tool = CustomTool(func=demo_func)
    agent.tools = [custom_tool]
    res = task.execute_sync(agent=agent)
    assert "_demo" in res.tool_output


def test_task_with_tools():

    def random_func(message: str) -> str:
        return message + "_demo"

    tool = Tool(name="tool", func=random_func)
    tool_set = ToolSet(tool=tool, kwargs=dict(message="empty func"))

    agent = Agent(role="Tool Handler", goal="execute tools")
    task = Task(description="execute the function", tools=[tool_set,], tool_res_as_final=True)
    res = task.execute_sync(agent=agent)
    assert res.tool_output == "empty func_demo"

    class CustomTool(Tool):
        name: str = "custom tool"
        func: Callable = None

    custom_tool = CustomTool(func=random_func)
    task.tools = [custom_tool]
    res = task.execute_sync(agent=base_agent)
    assert "_demo" in res.tool_output

    task.tools = [custom_tool]
    res = task.execute_sync(agent=base_agent)
    assert res.tool_output is not None



def test_task_without_response_format():
    task = Task(description="return a simple output with any random values.")
    res = task.execute_sync(agent=base_agent)

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
    res = task.execute_sync(agent=agent)

    assert res and isinstance(res, TaskOutput)
    assert res.json_dict and isinstance(res.json_dict, dict)
    assert res.pydantic is None




# task - maxit, loop, rpm
