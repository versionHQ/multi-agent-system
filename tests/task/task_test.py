import os
import pytest
from unittest.mock import patch
from typing import Dict, Any, List, Optional, Callable

from pydantic import BaseModel, Field, InstanceOf

from versionhq.agent.model import Agent
from versionhq.task.model import Task, ResponseField, TaskOutput, ConditionalTask
from versionhq.tool.model import Tool, ToolSet
from versionhq.llm.llm_vars import MODELS
from versionhq.llm.model import DEFAULT_MODEL_NAME, LLM


class DemoChild(BaseModel):
    """
    A nested outcome class.
    """
    ch_1: str
    ch_2: dict[str, str]


class DemoOutcome(BaseModel):
    """
    A demo pydantic class to validate the outcome with various nested data types.
    """
    test0: int
    test1: float
    test2: str
    test3: bool
    test4: list[str]
    # test5: dict[str, Any]
    # test6: list[dict[str, Any]]
    test7: Optional[list[str]]

    test8: list[list[str]]
    # children: List[DemoChild]


demo_nested_response_fields = [
    ResponseField(title="test0", data_type=int),
    ResponseField(title="test1", data_type=str, required=True),
    ResponseField(title="test2", data_type=list, items=str),
    ResponseField(title="test3", data_type=list, items=dict, properties=[
        ResponseField(title="nest1", data_type=str),
        ResponseField(title="nest2", type=dict, properties=[ResponseField(title="test", data_type=str)])
    ]),
    ResponseField(title="test4", data_type=dict, properties=[ResponseField(title="ch", data_type=tuple)]),
    ResponseField(title="test5", data_type=bool),
    ResponseField(title="test6", data_type=list, items=Any, required=False),
    # ResponseField(title="children", data_type=list, items=type(DemoChild)),
]


def create_base_agent(model: str | LLM | Dict[str, Any]) -> Agent:
    agent = Agent(role="demo", goal="My amazing goals", llm=model, max_tokens=3000)
    return agent

agent = create_base_agent(model=DEFAULT_MODEL_NAME)


def test_sync_execute_task_with_pydantic_outcome():
    task = Task(
        description="Output random values strictly following the given response foramt and prompt.",
        pydantic_custom_output=DemoOutcome
    )
    res = task.execute_sync(agent=agent)

    assert isinstance(res, TaskOutput) and res.task_id is task.id
    assert isinstance(res.raw, str)
    assert isinstance(res.json_dict, dict)
    assert res.pydantic == DemoOutcome(**res.json_dict)
    assert [v and type(v) is type(getattr(res.pydantic, k)) for k, v in res.pydantic.dict().items()]
    # assert [isinstance(item.ch_1, str) and isinstance(item.ch_2, dict) for item in res.pydantic.children]


def test_sync_execute_task_with_json_dict():
    task = Task(
        description="Output random values strictly following the given response foramt and prompt.",
        response_fields=demo_nested_response_fields
    )
    res = task.execute_sync(agent=agent)

    assert isinstance(res, TaskOutput) and res.task_id is task.id
    assert res.raw and isinstance(res.raw, str)
    assert res.pydantic is None
    assert res.json_dict and isinstance(res.json_dict, dict)
    assert [v and type(v) == task.response_fields[i].data_type for i, (k, v) in enumerate(res.json_dict.items())]
    # assert [isinstance(item, DemoChild) and isinstance(item.ch_1, str) and isinstance(item.ch_2, dict)
    #         for item in res.json_dict["children"]]


def test_async_execute_task():
    task = Task(description="Return string: 'test'")

    with patch.object(Agent, "execute_task", return_value="test") as execute:
        execution = task.execute_async(agent=agent)
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
    res = main_task.execute_sync(agent=agent)

    assert isinstance(res, TaskOutput)
    assert res.task_id is main_task.id
    assert res.raw is not None
    assert isinstance(res.raw, str)
    assert res.json_dict is not None
    assert isinstance(res.json_dict, dict)
    assert res.pydantic is None
    assert sub_task.output is not None
    assert sub_task.output.json_dict is not None
    assert "subtask_result" in main_task.prompt(model_provider=agent.llm.provider)


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
        response_fields=[
            ResponseField(title="result", data_type=str, required=True),
        ]
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
    res = main_task.execute_sync(agent=agent, context="plan a Black Friday campaign.")

    assert isinstance(res, TaskOutput) and res.task_id is main_task.id
    assert res.raw and isinstance(res.raw, str)
    assert res.json_dict and isinstance(res.json_dict, dict)
    assert res.pydantic == Outcome(test1=res.json_dict["test1"], test2=res.json_dict["test2"])

    assert sub_task.output is not None
    assert sub_task.output.json_dict is not None
    assert sub_task.output.pydantic is None

    assert "result" in main_task.prompt(model_provider=agent.llm.provider)
    assert main_task.prompt_context == "plan a Black Friday campaign."
    assert "plan a Black Friday campaign." in main_task.prompt(model_provider=agent.llm.provider)


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
    res = task.execute_sync(agent=agent)

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
    res = task.execute_sync(agent=agent)

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
    res = task.execute_sync(agent=agent)
    assert "_demo" in res.tool_output

    task.tools = [custom_tool]
    res = task.execute_sync(agent=agent)
    assert res.tool_output is not None



def test_task_without_response_format():
    task = Task(description="return a simple output with any random values.")
    res = task.execute_sync(agent=agent)

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




if __name__ == "__main__":
    test_task_with_tools()


# tool - use_llm = true -
# task - agent - maxit
# agents with multiple callbacks
