import os
from typing import Type
from pydantic import BaseModel
from versionhq.agent.model import Agent
from versionhq.task.model import Task, ResponseField, TaskOutput, AgentOutput

MODEL_NAME = os.environ.get("LITELLM_MODEL_NAME", "gpt-3.5-turbo")


def test_sync_execute_task():
    agent_a = Agent(
        role="Demo Agent A",
        goal="My amazing goals",
        backstory="My amazing backstory",
        verbose=True,
        llm=MODEL_NAME,
        max_tokens=3000,
    )

    task = Task(
        description="Analyze the client's business model, target audience, and customer information and define the optimal cohort timeframe based on customer lifecycle and product usage patterns.",
        expected_output_json=True,
        output_field_list=[
            ResponseField(title="test1", type=str, required=True),
            ResponseField(title="test2", type=list, required=True),
        ],
        expected_output_pydantic=True,
        context=[],
        tools=[],
        callback=None,
    )
    res = task.execute_sync(agent=agent_a)

    assert isinstance(res, TaskOutput)
    assert res.task_id is task.id
    assert res.raw is not None
    assert isinstance(res.raw, str)
    assert res.json_dict is not None
    assert isinstance(res.json_dict, dict)
    assert res.pydantic is not None
    assert hasattr(res.pydantic, "test1")
    assert type(res.pydantic.test1) == str
    assert hasattr(res.pydantic, "test2")
    assert type(res.pydantic.test2) == list

# CALLBACKS, TASK HANDLED BY AGENTS WITH TOOLS, FUTURE, ASYNC, CONDITIONAL, token usage
