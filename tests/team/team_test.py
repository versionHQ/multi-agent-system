
from versionhq.agent.model import Agent
from versionhq.task.model import Task, ResponseField, TaskOutput
from versionhq.team.model import Team, Member, TaskHandlingProcess, TeamOutput
from versionhq._utils.usage_metrics import UsageMetrics
from versionhq.llm.model import DEFAULT_MODEL_NAME


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
            Member(agent=agent_a, is_manager=True, tasks=[task_1,]),
            Member(agent=agent_b, is_manager=False, tasks=[task_2,]),
        ],
    )

    assert team.id and team.key and isinstance(team.key, str) and team.managers
    assert task_1 in team.manager_tasks
    assert team.tasks == [task_1, task_2]
    assert [item.id is task_1.id or item.id is task_2.id for item in team.tasks]


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
            Member(agent=agent_a, is_manager=False, tasks=[task_1,]),
            Member(agent=agent_b, is_manager=False, tasks=[task_2,]),
        ],
    )

    assert team.id and team.key and isinstance(team.key, str)
    assert team.managers is None
    assert team.manager_tasks == []
    assert team.tasks == [task_1, task_2]
    assert [item.id is task_1.id or item.id is task_2.id for item in team.tasks]


def test_launch_without_leader():
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
            Member(agent=agent_a, is_manager=False, tasks=[task_1,]),
            Member(agent=agent_b, is_manager=False, tasks=[task_2,]),
        ],
    )
    res = team.launch()
    res_all = res.return_all_task_outputs()

    assert isinstance(res, TeamOutput) and res.team_id is team.id
    assert isinstance(res.raw, str) and isinstance(res.json_dict, dict)
    assert res.pydantic is None
    assert [isinstance(item, TaskOutput) for item in res.task_outputs]
    assert isinstance(res_all, list) and len(res_all) == 2 and [isinstance(item, dict) for item in res_all]
    assert isinstance(res.token_usage, UsageMetrics)
    assert res.token_usage.total_tokens == 0 # as we dont set token usage on agent


def team_launch_with_task_callback():
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
            Member(agent=agent_a, is_manager=False, tasks=[task_1,]),
            Member(agent=agent_b, is_manager=False, tasks=[task_2,]),
        ],
    )
    res = team.launch()

    assert res.raw and res.json_dict
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
            Member(agent=agent_a, is_manager=False, tasks=[task_1,]),
            Member(agent=agent_b, is_manager=False, tasks=[task_2,]),
        ],
    )
    res = team.launch()

    assert res.raw is not None
    assert res.json_dict is not None
    assert "agent b" in task_1.processed_agents


def test_launch_with_leader():
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
            Member(agent=agent_a, is_manager=False, tasks=[task_1,]),
            Member(agent=agent_b, is_manager=True, tasks=[task_2,]),
        ],
    )
    res = team.launch()

    assert isinstance(res, TeamOutput)
    assert res.team_id is team.id
    assert res.raw is not None
    assert res.json_dict is not None
    assert team.managers[0].agent.id is agent_b.id
    assert len(res.task_outputs) == 2
    assert [item.raw is not None for item in res.task_outputs]
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
            Member(agent=agent_a, is_manager=False, tasks=[task_1,]),
            Member(agent=agent_b, is_manager=True, tasks=[task_2,]),
            Member(agent=agent_c, is_manager=False)
        ],
        process=TaskHandlingProcess.hierarchical
    )
    res = team.launch()

    assert isinstance(res, TeamOutput)
    assert res.team_id is team.id
    assert res.raw is not None
    assert res.json_dict is not None
    assert team.managers[0].agent.id is agent_b.id
    assert len(res.task_outputs) == 2
    assert [item.raw is not None for item in res.task_outputs]
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
            Member(agent=agent_c, is_manager=False)
        ],
        team_tasks=[team_task, task_1, task_2, ]
    )
    team_flat =  Team(
        members=[
            Member(agent=agent_a, is_manager=False, tasks=[task_1,]),
            Member(agent=agent_c, is_manager=False)
        ],
        team_tasks=[team_task, task_2,]
    )
    team_leader =  Team(
        members=[
            Member(agent=agent_a, is_manager=False, tasks=[task_1,]),
            Member(agent=agent_b, is_manager=True, tasks=[task_2,]),
            Member(agent=agent_c, is_manager=False)
        ],
        team_tasks=[team_task, ]
    )
    team_dual_leaders =  Team(
        members=[
            Member(agent=agent_a, is_manager=False, tasks=[task_1,]),
            Member(agent=agent_b, is_manager=True, tasks=[task_2,]),
            Member(agent=agent_c, is_manager=True)
        ],
        team_tasks=[team_task, ]
    )
    team_leader_without_task =  Team(
        members=[
            Member(agent=agent_a, is_manager=False, tasks=[task_1,]),
            Member(agent=agent_b, is_manager=False, tasks=[task_2,]),
            Member(agent=agent_c, is_manager=True)
        ],
        team_tasks=[team_task,]
    )
    teams = [team_solo, team_flat, team_leader, team_dual_leaders, team_leader_without_task]

    for (i, team) in enumerate(teams):
        res = team.launch()
        assert team._get_responsible_agent(task=task_1) is not None
        assert isinstance(res, TeamOutput) and res.team_id is team.id
        assert team.tasks[0].id is team_task.id
        assert res.raw is not None
