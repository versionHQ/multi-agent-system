from unittest.mock import patch

from versionhq.agent.model import Agent
from versionhq.task.model import Task, ResponseField, TaskOutput
from versionhq.agent_network.model import AgentNetwork, Member, TaskHandlingProcess, NetworkOutput
from versionhq._utils.usage_metrics import UsageMetrics
from versionhq.llm.model import DEFAULT_MODEL_NAME


def test_form_agent_network():
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
    network = AgentNetwork(
        members=[
            Member(agent=agent_a, is_manager=True, tasks=[task_1,]),
            Member(agent=agent_b, is_manager=False, tasks=[task_2,]),
        ],
    )

    assert network.id and network.key and isinstance(network.key, str) and network.managers
    assert task_1 in network.manager_tasks
    assert network.tasks == [task_1, task_2]
    assert [item.id is task_1.id or item.id is task_2.id for item in network.tasks]


def test_form_network_without_leader():
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
    network = AgentNetwork(
        members=[
            Member(agent=agent_a, is_manager=False, tasks=[task_1,]),
            Member(agent=agent_b, is_manager=False, tasks=[task_2,]),
        ],
    )

    assert network.id and network.key and isinstance(network.key, str)
    assert network.managers is None
    assert network.manager_tasks == []
    assert network.tasks == [task_1, task_2]
    assert [item.id is task_1.id or item.id is task_2.id for item in network.tasks]


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
    network = AgentNetwork(
        members=[
            Member(agent=agent_a, is_manager=False, tasks=[task_1,]),
            Member(agent=agent_b, is_manager=False, tasks=[task_2,]),
        ],
    )
    res = network.launch()
    res_all = res.return_all_task_outputs()

    assert isinstance(res, NetworkOutput) and res.network_id is network.id
    assert isinstance(res.raw, str) and isinstance(res.json_dict, dict)
    assert res.pydantic is None
    assert [isinstance(item, TaskOutput) for item in res.task_outputs]
    assert isinstance(res_all, list) and len(res_all) == 2 and [isinstance(item, dict) for item in res_all]
    # assert isinstance(res.token_usage, UsageMetrics)
    # assert res.token_usage.total_tokens == 0 # as we dont set token usage on agent


def test_launch_with_task_callback():
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

    network = AgentNetwork(
        members=[
            Member(agent=agent_a, is_manager=False, tasks=[task_1,]),
            Member(agent=agent_b, is_manager=False, tasks=[task_2,]),
        ],
    )
    res = network.launch()

    assert res.raw and res.json_dict
    assert len(res.return_all_task_outputs()) == 2
    assert len(demo_list) == 2
    assert "pytest" in demo_list[0]
    assert "pytest" in demo_list[1]


# def test_delegate_in_network():
#     """
#     When the agent belongs to the agent network, its manager or peers are prioritized to delegete the task.
#     """

#     agent_a = Agent(role="agent a", goal="My amazing goals", llm=DEFAULT_MODEL_NAME)
#     agent_b = Agent(role="agent b", goal="My amazing goals", llm=DEFAULT_MODEL_NAME)
#     task_1 = Task(
#         description="Analyze the client's business model.",
#         response_fields=[ResponseField(title="test1", data_type=str, required=True),],
#         allow_delegation=True
#     )
#     task_2 = Task(
#         description="Define the cohort.",
#         response_fields=[ResponseField(title="test1", data_type=int, required=True),],
#         allow_delegation=False
#     )
#     network = AgentNetwork(
#         members=[
#             Member(agent=agent_a, is_manager=False, tasks=[task_1,]),
#             Member(agent=agent_b, is_manager=False, tasks=[task_2,]),
#         ],
#     )
#     res = network.launch()
#
#     assert res.raw is not None
#     assert res.json_dict is not None
#     assert "vhq-Delegated-Agent" in task_1.processed_agents



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
    network = AgentNetwork(
        members=[
            Member(agent=agent_a, is_manager=False, tasks=[task_1,]),
            Member(agent=agent_b, is_manager=True, tasks=[task_2,]),
        ],
    )
    res = network.launch()

    assert isinstance(res, NetworkOutput)
    assert res.network_id is network.id
    assert res.raw is not None
    assert res.json_dict is not None
    assert network.managers[0].agent.id is agent_b.id
    assert len(res.task_outputs) == 2
    assert [item.raw is not None for item in res.task_outputs]
    assert len(network.tasks) == 2
    assert network.tasks[0].output.raw == res.raw


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
    network = AgentNetwork(
        members=[
            Member(agent=agent_a, is_manager=False, tasks=[task_1,]),
            Member(agent=agent_b, is_manager=True, tasks=[task_2,]),
            Member(agent=agent_c, is_manager=False)
        ],
        process=TaskHandlingProcess.HIERARCHY
    )
    res = network.launch()

    assert isinstance(res, NetworkOutput)
    assert res.network_id is network.id
    assert res.raw is not None
    assert res.json_dict is not None
    assert network.managers[0].agent.id is agent_b.id
    assert len(res.task_outputs) == 2
    assert [item.raw is not None for item in res.task_outputs]
    assert len(network.tasks) == 2
    assert network.tasks[0].output.raw == res.raw


def test_handle_network_task():
    """
    Make the best network formation with agents and tasks given.
    """

    agent_a = Agent(role="agent a", goal="My amazing goals", llm=DEFAULT_MODEL_NAME)
    agent_b = Agent(role="agent b", goal="My amazing goals", llm=DEFAULT_MODEL_NAME)
    agent_c = Agent(role="agent c", goal="My amazing goals", llm=DEFAULT_MODEL_NAME)
    network_task = Task(
        description="Define outbound strategies.",
        response_fields=[ResponseField(title="network_task_1", data_type=str, required=True),],
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

    network_solo = AgentNetwork(
        members=[
            Member(agent=agent_c, is_manager=False)
        ],
        network_tasks=[network_task, task_1, task_2, ]
    )
    network_flat =  AgentNetwork(
        members=[
            Member(agent=agent_a, is_manager=False, tasks=[task_1,]),
            Member(agent=agent_c, is_manager=False)
        ],
        network_tasks=[network_task, task_2,]
    )
    network_leader =  AgentNetwork(
        members=[
            Member(agent=agent_a, is_manager=False, tasks=[task_1,]),
            Member(agent=agent_b, is_manager=True, tasks=[task_2,]),
            Member(agent=agent_c, is_manager=False)
        ],
        network_tasks=[network_task, ]
    )
    network_dual_leaders =  AgentNetwork(
        members=[
            Member(agent=agent_a, is_manager=False, tasks=[task_1,]),
            Member(agent=agent_b, is_manager=True, tasks=[task_2,]),
            Member(agent=agent_c, is_manager=True)
        ],
        network_tasks=[network_task, ]
    )
    network_leader_without_task =  AgentNetwork(
        members=[
            Member(agent=agent_a, is_manager=False, tasks=[task_1,]),
            Member(agent=agent_b, is_manager=False, tasks=[task_2,]),
            Member(agent=agent_c, is_manager=True)
        ],
        network_tasks=[network_task,]
    )
    networks = [network_solo, network_flat, network_leader, network_dual_leaders, network_leader_without_task]


    for item in networks:
        with patch.object(AgentNetwork, "_execute_tasks", kwargs=dict(tasks=item.tasks)) as private_mock:
            item.launch()
            private_mock.assert_called_once()
