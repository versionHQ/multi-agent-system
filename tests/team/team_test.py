import os
from pydantic import BaseModel

from versionhq.agent.model import Agent
from versionhq.task.model import Task, ResponseField, TaskOutput
from versionhq.team.model import Team, TeamMember, TaskHandlingProcess, TeamOutput
from versionhq._utils.usage_metrics import UsageMetrics

MODEL_NAME = os.environ.get("LITELLM_MODEL_NAME", "gpt-3.5-turbo")


def test_form_team():
    agent_a = Agent(
        role="Demo Agent A",
        goal="My amazing goals",
        backstory="My amazing backstory",
        verbose=True,
        llm=MODEL_NAME,
        max_tokens=3000,
    )

    agent_b = Agent(
        role="Demo Agent B-1",
        goal="My amazing goals",
        verbose=True,
        llm=MODEL_NAME,
        max_tokens=3000,
    )

    task_1 = Task(
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

    task_2 = Task(
        description="Amazing task description",
        expected_output_json=True,
        expected_output_pydantic=True,
        output_field_list=[
            ResponseField(title="test1", type=int, required=True),
            ResponseField(title="test2", type=list, required=True),
        ],
        context=[],
        tools=[],
        callback=None,
    )

    team = Team(
        members=[
            TeamMember(agent=agent_a, is_manager=True, task=task_1),
            TeamMember(agent=agent_b, is_manager=False, task=task_2),
        ],
    )

    assert team.id is not None
    assert team.key is not None
    assert isinstance(team.key, str)
    assert team.manager_agent is not None
    assert team.manager_task.id is task_1.id
    assert len(team.tasks) == 2
    for item in team.tasks:
        assert item.id is task_1.id or item.id is task_2.id


def test_form_team_without_leader():
    agent_a = Agent(
        role="Demo Agent A",
        goal="My amazing goals",
        backstory="My amazing backstory",
        verbose=True,
        llm=MODEL_NAME,
        max_tokens=3000,
    )

    agent_b = Agent(
        role="Demo Agent B-1",
        goal="My amazing goals",
        verbose=True,
        llm=MODEL_NAME,
        max_tokens=3000,
    )

    task_1 = Task(
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

    task_2 = Task(
        description="Amazing task description",
        expected_output_json=True,
        expected_output_pydantic=True,
        output_field_list=[
            ResponseField(title="test1", type=int, required=True),
            ResponseField(title="test2", type=list, required=True),
        ],
        context=[],
        tools=[],
        callback=None,
    )

    team = Team(
        members=[
            TeamMember(agent=agent_a, is_manager=False, task=task_1),
            TeamMember(agent=agent_b, is_manager=False, task=task_2),
        ],
    )

    assert team.id is not None
    assert team.key is not None
    assert isinstance(team.key, str)
    assert team.manager_agent is None
    assert team.manager_task is None
    assert len(team.tasks) == 2
    for item in team.tasks:
        assert item.id is task_1.id or item.id is task_2.id


def test_kickoff_team_without_leader():
    agent_a = Agent(
        role="Demo Agent A",
        goal="My amazing goals",
        backstory="My amazing backstory",
        verbose=True,
        llm=MODEL_NAME,
        max_tokens=3000,
    )

    agent_b = Agent(
        role="Demo Agent B-1",
        goal="My amazing goals",
        verbose=True,
        llm=MODEL_NAME,
        max_tokens=3000,
    )

    task_1 = Task(
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

    task_2 = Task(
        description="Amazing task description",
        expected_output_json=True,
        expected_output_pydantic=True,
        output_field_list=[
            ResponseField(title="test1", type=int, required=True),
            ResponseField(title="test2", type=list, required=True),
        ],
        context=[],
        tools=[],
        callback=None,
    )

    team = Team(
        members=[
            TeamMember(agent=agent_a, is_manager=False, task=task_1),
            TeamMember(agent=agent_b, is_manager=False, task=task_2),
        ],
    )
    res = team.kickoff()
    res_all = res.return_all_task_outputs()

    assert isinstance(res, TeamOutput)
    assert res.team_id is team.id
    assert res.raw is not None
    assert isinstance(res.raw, str)
    assert res.json_dict is not None
    assert isinstance(res.json_dict, dict)
    assert res.pydantic is not None
    for item in res.task_output_list:
        assert isinstance(item, TaskOutput)

    assert isinstance(res_all, list)
    assert len(res_all) == 2
    for item in res_all:
        assert isinstance(item, dict)
        assert "test1" in item
        assert "test2" in item

    assert isinstance(res.token_usage, UsageMetrics)
    assert res.token_usage.total_tokens == 0 # as we dont set token usage on agent



# def test_kickoff_with_team_leader():
#     agent_a = Agent(
#         role="Demo Agent A",
#         goal="My amazing goals",
#         backstory="My amazing backstory",
#         verbose=True,
#         llm=MODEL_NAME,
#         max_tokens=3000,
#     )

#     agent_b1 = Agent(
#         role="Demo Agent B-1",
#         goal="My amazing goals",
#         verbose=True,
#         llm=MODEL_NAME,
#         max_tokens=3000,
#     )

#     task_1 = Task(
#         description="Analyze the client's business model, target audience, and customer information and define the optimal cohort timeframe based on customer lifecycle and product usage patterns.",
#         expected_output_json=True,
#         output_field_list=[
#             ResponseField(title="test1", type=str, required=True),
#             ResponseField(title="test2", type=list, required=True),
#         ],
#         expected_output_pydantic=True,
#         context=[],
#         tools=[],
#         callback=None,
#     )

#     task_2 = Task(
#         description="Amazing task description",
#         expected_output_json=True,
#         expected_output_pydantic=True,
#         output_field_list=[
#             ResponseField(title="test1", type=int, required=True),
#             ResponseField(title="test2", type=list, required=True),
#         ],
#         context=[],
#         tools=[],
#         callback=None,
#     )

#     team_task = Task(
#         description="Amazing team task description",
#         expected_output_json=True,
#         expected_output_pydantic=True,
#         output_field_list=[
#             ResponseField(title="field-1", type=str, required=True),
#         ],
#     )

#     team = Team(
#         members=[
#             TeamMember(agent=agent_a, is_manager=True, task=task_1),
#             TeamMember(agent=agent_b1, is_manager=False, task=task_2),
#         ],
#         team_tasks=[team_task,],
#         process=TaskHandlingProcess.sequential,
#         verbose=True,
#         memory=False,
#         before_kickoff_callbacks=[],  # add any callables
#         after_kickoff_callbacks=[],
#         prompt_file="sample.demo.Prompts.demo.py",
#     )

#     res = team.kickoff()
#     res_all = res.return_all_task_outputs()

#     assert isinstance(res, TeamOutput)
#     assert res.team_id == team.id
#     assert res.raw is not None
#     assert isinstance(res.raw, str)
#     assert res.json_dict is not None
#     assert isinstance(res.json_dict, dict)
#     assert isinstance(res.pydantic, BaseModel)

#     assert isinstance(res_all, list)
#     assert len(res_all) == 2
#     for item in res_all:
#         assert isinstance(item, TaskOutput)
#         assert "test1" in item
#         assert "test2" in item

#     assert isinstance(res.token_usage, UsageMetrics)
#     assert res.token_usage.total_tokens > 0
