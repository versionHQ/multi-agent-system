import os
import pytest
from unittest.mock import patch
from typing import Dict, Any, List, Optional

from pydantic import BaseModel, Field, InstanceOf

from versionhq.agent.model import Agent
from versionhq.task.model import Task, ResponseField, TaskOutput, ConditionalTask
from versionhq.tool.model import Tool, ToolSet
from versionhq.llm.llm_variables import MODELS
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
        execute.assert_called_once_with(task=task, context=None, task_tools=None)


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


def test_task_with_agent_tools(model: str | LLM | Dict[str, Any] = DEFAULT_MODEL_NAME):
    def empty_func():
        return "empty function"

    class CustomTool(Tool):
        name: str = "custom tool"

        def _run(self) -> str:
            return "empty function"

    custom_tool = CustomTool()

    agent_str_tool = Agent(role="demo 1", goal="test a tool", tools=["random tool 1", "random tool 2",], llm=model)
    agent_dict_tool = Agent(role="demo 2", goal="test a tool", tools=[dict(name="tool 1", function=empty_func)], llm=model)
    agent_custom_tool = Agent(role="demo 3", goal="test a tool", tools=[custom_tool, custom_tool], llm=model)
    agents = [agent_str_tool, agent_dict_tool, agent_custom_tool]

    task = Task(description="execute the function", can_use_agent_tools=True, take_tool_res_as_final=True)

    for agent in agents:
        res = task.execute_sync(agent=agent)
        assert "empty function" in res.tool_output
        assert len(res.tool_output) == len(agent.tools)


def test_task_with_tools():

    def random_func(str: str = None) -> str:
        return str

    class CustomTool(Tool):
        name: str = "custom tool"

        def _run(self) -> str:
            return "custom function"

    tool = Tool(name="tool", function=random_func)
    custom_tool = CustomTool()
    tool_set = ToolSet(tool=tool, kwargs=dict(str="empty function"))
    task = Task(description="execute the function", tools=[custom_tool, tool, tool_set], take_tool_res_as_final=True)
    res = task.execute_sync(agent=agent)

    assert res.tool_output is not None
    assert isinstance(res.tool_output, list)
    assert len(res.tool_output) == len(task.tools)
    assert res.tool_output[0] == "custom function"
    assert res.tool_output[1] is None
    assert res.tool_output[2] == "empty function"


def test_task_without_response_format():
    task = Task(description="return a simple output with any random values.")
    res = task.execute_sync(agent=agent)

    assert res and isinstance(res, TaskOutput)
    assert res.json_dict and isinstance(res.json_dict, dict)
    assert res.pydantic is None


def _test_switch_model():
    """
    See if diff models can return their outputs in a correct format.
    """
    models = MODELS.get("openai")[0]

    for model in models:
        test_create_json_output_from_complex_response_schema(model=model)
        test_create_class_output_from_complex_response_scheme(model=model)
        test_task_without_res_field(model=model)



if __name__ == "__main__":
    test_task_without_response_format()


# tool - use_llm = true
