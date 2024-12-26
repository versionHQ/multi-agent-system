import os
import pytest
from typing import Union
from versionhq.agent.model import Agent
from versionhq.task.model import Task, ResponseField, TaskOutput, AgentOutput

DEFAULT_MODEL_NAME = os.environ.get("LITELLM_MODEL_NAME", "gpt-3.5-turbo")
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY")


def test_sync_execute_task():
    agent = Agent(
        role="demo agent 000",
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
        context=None,
        callback=None,
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


def test_sync_execute_task_with_context():
    """
    Use case = One agent handling multiple tasks sequentially using context set in the main task.
    """

    agent = Agent(
        role="demo agent 001",
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
    res = main_task.execute_sync(agent=agent)

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
            assert type(res.pydantic.test1) == Union[int, str]

        assert hasattr(res.pydantic, "test2")
        if res.pydantic.test2:
            assert type(res.pydantic.test2) == Union[list, str]

    assert sub_task.output is not None
    assert sub_task.output.json_dict is not None
    assert "result" in main_task.prompt()


# CALLBACKS, tools, FUTURE, ASYNC, CONDITIONAL, token usage
