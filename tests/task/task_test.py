import os
import pytest
from unittest.mock import patch
from typing import Dict, Any
from versionhq.agent.model import Agent
from versionhq.task.model import Task, ResponseField, TaskOutput, ConditionalTask

DEFAULT_MODEL_NAME = os.environ.get("LITELLM_MODEL_NAME", "gpt-3.5-turbo")
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY")


def test_sync_execute_task():
    agent = Agent(
        role="demo agent 1",
        goal="My amazing goals",
        backstory="My amazing backstory",
        verbose=True,
        llm=DEFAULT_MODEL_NAME,
        max_tokens=3000,
    )

    task = Task(
        description="Analyze the client's business model and define the optimal cohort timeframe.",
        expected_output_json=True,
        expected_output_pydantic=True,
        output_field_list=[
            ResponseField(title="test1", type=str, required=True),
            ResponseField(title="test2", type=list, required=True),
        ],
    )
    res = task.execute_sync(agent=agent)

    assert isinstance(res, TaskOutput)
    assert res.task_id is task.id
    assert res.raw is not None
    assert isinstance(res.raw, str)
    assert res.json_dict is not None
    assert isinstance(res.json_dict, dict)
    assert res.pydantic is not None

    if hasattr(res.pydantic, "output"):
        assert res.pydantic.output is not None
    else:
        assert hasattr(res.pydantic, "test1")
        if res.pydantic.test1:
            assert type(res.pydantic.test1) == str

        assert hasattr(res.pydantic, "test2")
        if res.pydantic.test2:
            assert type(res.pydantic.test2) == list


def test_async_execute_task():
    agent = Agent(
        role="demo agent 2",
        goal="My amazing goals",
    )
    task = Task(
        description="Analyze the client's business model and define the optimal cohort timeframe.",
        expected_output_json=True,
        expected_output_pydantic=False,
        output_field_list=[
            ResponseField(title="test1", type=str, required=True),
            ResponseField(title="test2", type=list, required=True),
        ],
    )

    with patch.object(Agent, "execute_task", return_value="ok") as execute:
        execution = task.execute_async(agent=agent)
        result = execution.result()
        assert result.raw == "ok"
        execute.assert_called_once_with(task=task, context=None, tools=None)


def test_sync_execute_with_task_context():
    """
    Use case = One agent handling multiple tasks sequentially using context set in the main task.
    """

    agent = Agent(role="demo 3", goal="My amazing goals", verbose=True, llm=DEFAULT_MODEL_NAME, max_tokens=3000,)
    sub_task = Task(
        description="analyze the client's business model",
        expected_output_json=True,
        output_field_list=[
            ResponseField(title="subtask_result", type=str, required=True),
        ]
    )
    main_task = Task(
        description="Define the optimal cohort timeframe in days and target audience.",
        expected_output_json=True,
        output_field_list=[
            ResponseField(title="test1", type=int, required=True),
            ResponseField(title="test2", type=str, required=True),
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
    assert "subtask_result" in main_task.prompt()


def test_sync_execute_task_with_prompt_context():
    """
    Use case:
    - One agent handling multiple tasks sequentially using context set in the main task.
    - On top of that, the agent receives context when they execute the task.
    """

    agent = Agent(
        role="demo agent 4",
        goal="My amazing goals",
        backstory="My amazing backstory",
        verbose=True,
        llm=DEFAULT_MODEL_NAME,
        max_tokens=3000,
    )

    sub_task = Task(
        description="analyze the client's business model",
        expected_output_json=True,
        expected_output_pydantic=False,
        output_field_list=[
            ResponseField(title="result", type=str, required=True),
        ]
    )
    main_task = Task(
        description="Define the optimal cohort timeframe in days and target audience.",
        expected_output_json=True,
        expected_output_pydantic=True,
        output_field_list=[
            ResponseField(title="test1", type=int, required=True),
            ResponseField(title="test2", type=str, required=True),
        ],
        context=[sub_task]
    )
    res = main_task.execute_sync(agent=agent, context="plan a Black Friday campaign.")

    assert isinstance(res, TaskOutput)
    assert res.task_id is main_task.id
    assert res.raw is not None
    assert isinstance(res.raw, str)
    assert res.json_dict is not None
    assert isinstance(res.json_dict, dict)
    assert res.pydantic is not None

    if hasattr(res.pydantic, "output"):
        assert res.pydantic.output is not None
    else:
        assert hasattr(res.pydantic, "test1")
        if res.pydantic.test1:
            assert type(res.pydantic.test1) == int | str

        assert hasattr(res.pydantic, "test2")
        if res.pydantic.test2:
            assert type(res.pydantic.test2) == list | str

    assert sub_task.output is not None
    assert sub_task.output.json_dict is not None
    assert sub_task.output.pydantic is None

    assert "result" in main_task.prompt()
    assert main_task.prompt_context == "plan a Black Friday campaign."
    assert "plan a Black Friday campaign." in main_task.prompt()


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

    agent = Agent(role="demo agent 5", goal="My amazing goals")
    task = Task(
        description="Analyze the client's business model and define the optimal cohort timeframe.",
        expected_output_json=True,
        expected_output_pydantic=False,
        output_field_list=[
            ResponseField(title="test1", type=str, required=True),
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
        description="Analyze the client's business model and define the optimal cohort timeframe.",
        expected_output_json=True,
        expected_output_pydantic=False,
        output_field_list=[
            ResponseField(title="test1", type=str, required=True),
        ],
        allow_delegation=True
    )

    task.execute_sync(agent=agent)

    assert task.output is not None
    assert "delegated_agent" in task.processed_by_agents
    assert task.delegations != 0


def test_conditional_task():
    agent = Agent(role="demo agent 6", goal="My amazing goals")
    task = Task(
        description="Analyze the client's business model and define the optimal cohort timeframe.",
        output_field_list=[ResponseField(title="test1", type=str, required=True),],
    )
    res = task.execute_sync(agent=agent)

    conditional_task = ConditionalTask(
        description="Analyze the client's business model.",
        output_field_list=[ResponseField(title="test1", type=str, required=True),],
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
        description="Analyze the client's business model.",
        output_field_list=[ResponseField(title="task_1", type=str, required=True),],
    )
    assert task._task_output_handler.load() is not None

# tools, token usage
