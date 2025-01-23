import os

from versionhq.agent.model import Agent
from versionhq.task.model import Task, ResponseField, TaskOutput
from versionhq.team.model import Team, TeamMember, TaskHandlingProcess, TeamOutput
from versionhq._utils.usage_metrics import UsageMetrics
from versionhq.llm.llm_vars import MODELS
from versionhq.llm.model import DEFAULT_MODEL_NAME, LLM


def test_form_team():
    agent_a = Agent(role="agent a", goal="My amazing goals", backstory="My amazing backstory", llm=DEFAULT_MODEL_NAME, max_tokens=3000)
    agent_b = Agent(role="agent b", goal="My amazing goals", llm=DEFAULT_MODEL_NAME, max_tokens=3000)
    task_1 = Task(
        description="Analyze the client's business model.",
        response_fields=[
            ResponseField(title="test1", data_type=str, required=True),
            ResponseField(title="test2", data_type=list, required=True),
        ],
    )
    task_2 = Task(
        description="Define the cohort.",
        response_fields=[
            ResponseField(title="test1", data_type=int, required=True),
            ResponseField(title="test2", data_type=list, required=True),
        ],
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
    assert team.managers is not None
    assert task_1 in team.manager_tasks
    assert len(team.tasks) == 2
    for item in team.tasks:
        assert item.id is task_1.id or item.id is task_2.id


def test_form_team_without_leader():
    agent_a = Agent(role="agent a", goal="My amazing goals", backstory="My amazing backstory", llm=DEFAULT_MODEL_NAME, max_tokens=3000)
    agent_b = Agent(role="agent b", goal="My amazing goals", llm=DEFAULT_MODEL_NAME, max_tokens=3000)
    task_1 = Task(
        description="Analyze the client's business model.",
        response_fields=[
            ResponseField(title="test1", data_type=str, required=True),
            ResponseField(title="test2", data_type=list, required=True),
        ]
    )
    task_2 = Task(
        description="Define the cohort.",
        response_fields=[
            ResponseField(title="test1", data_type=int, required=True),
            ResponseField(title="test2", data_type=list, required=True),
        ],
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
    assert team.managers is None
    assert team.manager_tasks is None
    assert len(team.tasks) == 2
    for item in team.tasks:
        assert item.id is task_1.id or item.id is task_2.id


def test_kickoff_without_leader():
    agent_a = Agent(role="agent a", goal="My amazing goals", llm=DEFAULT_MODEL_NAME)
    agent_b = Agent(role="agent b", goal="My amazing goals", llm=DEFAULT_MODEL_NAME)
    task_1 = Task(
        description="Analyze the client's business model.",
        response_fields=[
            ResponseField(title="test1", data_type=str, required=True),
            ResponseField(title="test2", data_type=list, required=True),
        ],
    )
    task_2 = Task(
        description="Define the cohort.",
        response_fields=[
            ResponseField(title="test1", data_type=int, required=True),
            ResponseField(title="test2", data_type=list, required=True),
        ],
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
    assert res.pydantic is None
    for item in res.task_output_list:
        assert isinstance(item, TaskOutput)
    assert isinstance(res_all, list)
    assert len(res_all) == 2
    for item in res_all:
        assert isinstance(item, dict)
    assert isinstance(res.token_usage, UsageMetrics)
    assert res.token_usage.total_tokens == 0 # as we dont set token usage on agent


def team_kickoff_with_task_callback():
    """
    Each task has callback with callback kwargs.
    """
    demo_list = []
    def demo_callback(item: str) -> None:
        demo_list.append(item)

    agent_a = Agent(
        role="agent a",
        goal="My amazing goals",
        llm=DEFAULT_MODEL_NAME
    )

    agent_b = Agent(
        role="agent b",
        goal="My amazing goals",
        llm=DEFAULT_MODEL_NAME
    )

    task_1 = Task(
        description="Analyze the client's business model.",
        response_fields=[ResponseField(title="test1", data_type=str, required=True),],
        callback=demo_callback,
        callback_kwargs=dict(item="pytest demo 1")
    )

    task_2 = Task(
        description="Define the cohort.",
        response_fields=[ResponseField(title="test1", data_type=int, required=True),],
        callback=demo_callback,
        callback_kwargs=dict(item="pytest demo 2")
    )

    team = Team(
        members=[
            TeamMember(agent=agent_a, is_manager=False, task=task_1),
            TeamMember(agent=agent_b, is_manager=False, task=task_2),
        ],
    )
    res = team.kickoff()

    assert res.raw is not None
    assert res.json_dict is not None
    assert len(res.return_all_task_outputs()) == 2
    assert len(demo_list) == 2
    assert "pytest" in demo_list[0]
    assert "pytest" in demo_list[1]


def test_delegate_in_team():
    """
    When the agent belongs to the team, the team manager or peers are prioritized to delegete the task.
    """

    agent_a = Agent(role="agent a", goal="My amazing goals", llm=DEFAULT_MODEL_NAME)
    agent_b = Agent(role="agent b", goal="My amazing goals", llm=DEFAULT_MODEL_NAME)
    task_1 = Task(
        description="Analyze the client's business model.",
        response_fields=[ResponseField(title="test1", data_type=str, required=True),],
        allow_delegation=True
    )
    task_2 = Task(
        description="Define the cohort.",
        response_fields=[ResponseField(title="test1", data_type=int, required=True),],
        allow_delegation=False
    )
    team = Team(
        members=[
            TeamMember(agent=agent_a, is_manager=False, task=task_1),
            TeamMember(agent=agent_b, is_manager=False, task=task_2),
        ],
    )
    res = team.kickoff()

    assert res.raw is not None
    assert res.json_dict is not None
    assert "agent b" in task_1.processed_by_agents


def test_kickoff_with_leader():
    agent_a = Agent(role="agent a", goal="My amazing goals", llm=DEFAULT_MODEL_NAME)
    agent_b = Agent(role="agent b", goal="My amazing goals", llm=DEFAULT_MODEL_NAME)
    task_1 = Task(
        description="Analyze the client's business model.",
        response_fields=[ResponseField(title="task_1", data_type=str, required=True),],
    )
    task_2 = Task(
        description="Define the cohort timeframe.",
        response_fields=[
            ResponseField(title="task_2_1", data_type=int, required=True),
            ResponseField(title="task_2_2", data_type=list, required=True),
        ],
    )
    team = Team(
        members=[
            TeamMember(agent=agent_a, is_manager=False, task=task_1),
            TeamMember(agent=agent_b, is_manager=True, task=task_2),
        ],
    )
    res = team.kickoff()

    assert isinstance(res, TeamOutput)
    assert res.team_id is team.id
    assert res.raw is not None
    assert res.json_dict is not None
    assert team.managers[0].agent.id is agent_b.id
    assert len(res.task_output_list) == 2
    assert [item.raw is not None for item in res.task_output_list]
    assert len(team.tasks) == 2
    assert team.tasks[0].output.raw == res.raw


def test_hierarchial_process():
    """
    Manager to handle the top priority task first.
    """

    agent_a = Agent(role="agent a", goal="My amazing goals", llm=DEFAULT_MODEL_NAME)
    agent_b = Agent(role="agent b", goal="My amazing goals", llm=DEFAULT_MODEL_NAME)
    agent_c = Agent(role="agent c", goal="My amazing goals", llm=DEFAULT_MODEL_NAME)
    task_1 = Task(
        description="Analyze the client's business model.",
        response_fields=[ResponseField(title="task_1", data_type=str, required=True),],
    )
    task_2 = Task(
        description="Define the cohort timeframe.",
        response_fields=[
            ResponseField(title="task_2_1", data_type=int, required=True),
            ResponseField(title="task_2_2", data_type=list, required=True),
        ],
    )
    team = Team(
        members=[
            TeamMember(agent=agent_a, is_manager=False, task=task_1),
            TeamMember(agent=agent_b, is_manager=True, task=task_2),
            TeamMember(agent=agent_c, is_manager=False)
        ],
        process=TaskHandlingProcess.hierarchical
    )
    res = team.kickoff()

    assert isinstance(res, TeamOutput)
    assert res.team_id is team.id
    assert res.raw is not None
    assert res.json_dict is not None
    assert team.managers[0].agent.id is agent_b.id
    assert len(res.task_output_list) == 2
    assert [item.raw is not None for item in res.task_output_list]
    assert len(team.tasks) == 2
    assert team.tasks[0].output.raw == res.raw


def test_handle_team_task():
    """
    Make the best team formation with agents and tasks given.
    """

    agent_a = Agent(role="agent a", goal="My amazing goals", llm=DEFAULT_MODEL_NAME)
    agent_b = Agent(role="agent b", goal="My amazing goals", llm=DEFAULT_MODEL_NAME)
    agent_c = Agent(role="agent c", goal="My amazing goals", llm=DEFAULT_MODEL_NAME)
    team_task = Task(
        description="Define outbound strategies.",
        response_fields=[ResponseField(title="team_task_1", data_type=str, required=True),],
    )
    task_1 = Task(
        description="Analyze the client's business model.",
        response_fields=[ResponseField(title="task_1", data_type=str, required=True),],
    )
    task_2 = Task(
        description="Define the cohort timeframe.",
        response_fields=[
            ResponseField(title="task_2_1", data_type=int, required=True),
            ResponseField(title="task_2_2", data_type=list, required=True),
        ],
    )
    team_solo = Team(
        members=[
            TeamMember(agent=agent_c, is_manager=False)
        ],
        team_tasks=[team_task, task_1, task_2, ]
    )
    team_flat =  Team(
        members=[
            TeamMember(agent=agent_a, is_manager=False, task=task_1),
            TeamMember(agent=agent_c, is_manager=False)
        ],
        team_tasks=[team_task, task_2,]
    )
    team_leader =  Team(
        members=[
            TeamMember(agent=agent_a, is_manager=False, task=task_1),
            TeamMember(agent=agent_b, is_manager=True, task=task_2),
            TeamMember(agent=agent_c, is_manager=False)
        ],
        team_tasks=[team_task, ]
    )
    team_dual_leaders =  Team(
        members=[
            TeamMember(agent=agent_a, is_manager=False, task=task_1),
            TeamMember(agent=agent_b, is_manager=True, task=task_2),
            TeamMember(agent=agent_c, is_manager=True)
        ],
        team_tasks=[team_task, ]
    )
    team_leader_without_task =  Team(
        members=[
            TeamMember(agent=agent_a, is_manager=False, task=task_1),
            TeamMember(agent=agent_b, is_manager=False, task=task_2),
            TeamMember(agent=agent_c, is_manager=True)
        ],
        team_tasks=[team_task,]
    )
    teams = [team_solo, team_flat, team_leader, team_dual_leaders, team_leader_without_task]

    for team in teams:
        res = team.kickoff()
        assert team._get_responsible_agent(task=team_task) is not None
        assert isinstance(res, TeamOutput)
        assert res.team_id is team.id
        assert team.tasks[0].id is team_task.id
        assert res.raw is not None
        assert len(team.members) == 3
        assert len(team.tasks) == 3
