def test_doc_an_formation_a():
    import versionhq as vhq

    network = vhq.form_agent_network(
        task="Find the best trip destination this summer.",
        expected_outcome="a list of destinations and why it's suitable",
        context="planning a suprise trip for my friend", # optional
    )
    assert isinstance(network, vhq.AgentNetwork)
    assert network.members # auto-generated agents as network members
    assert network.tasks # auto-defined sub-tasks to achieve the main task goal


def test_doc_an_formation_b():
    import versionhq as vhq
    from pydantic import BaseModel

    class Outcome(BaseModel):
        destinations: list[str]
        why_suitable: list[str]


    network = vhq.form_agent_network(
        task="Find the best trip destination this summer.",
        expected_outcome=Outcome,
        context="planning a suprise trip for my friend", # optional
    )

    assert isinstance(network, vhq.AgentNetwork)
    assert network.members
    assert network.tasks


def test_doc_an_formation_c():
    import versionhq as vhq
    from pydantic import BaseModel

    my_agent = vhq.Agent(
        role="Travel Agent",
        goal="select best trip destination",
        knowledge_sources=[".....","url1",]
    )

    class Outcome(BaseModel):
        destinations: list[str]
        why_suitable: list[str]

    network = vhq.form_agent_network(
        task="Find the best trip destination this summer.",
        expected_outcome=Outcome,
        context="planning a suprise trip for my friend",
        agents=[my_agent,]
    )

    assert isinstance(network, vhq.AgentNetwork)
    assert [member for member in network.members if member.agent.role == my_agent.role]
    assert network.tasks


def test_doc_an_formation_d():
    import versionhq as vhq
    from pydantic import BaseModel

    my_agent = vhq.Agent(
        role="Travel Agent",
        goal="select best trip destination",
        knowledge_sources=[".....","url1",]
    )

    class Outcome(BaseModel):
        destinations: list[str]
        why_suitable: list[str]

    network = vhq.form_agent_network(
        task="Find the best trip destination this summer.",
        expected_outcome=Outcome,
        context="planning a suprise trip for my friend",
        agents=[my_agent,],
        formation=vhq.Formation.SUPERVISING
    )

    assert isinstance(network, vhq.AgentNetwork)
    assert [member for member in network.members if member.agent == my_agent]
    assert network.tasks
    assert network.formation == vhq.Formation.SUPERVISING


def test_doc_an_config_a():
    import versionhq as vhq

    network = vhq.AgentNetwork(
        members=[
            vhq.Member(
                agent=vhq.Agent(role="new member", goal="work in the network"),
                is_manager=False  # explicitly mentioned. Setting `True` makes this member a manager of the network.
            ),
        ]
    )
    assert isinstance(network.members[0].agent, vhq.Agent)


def test_doc_an_config_b():
    import versionhq as vhq

    network = vhq.AgentNetwork(
    members=[
        vhq.Member(agent=vhq.Agent(role="member 1", goal="work in the network")),
        vhq.Member(agent=vhq.Agent(role="member 2", goal="work in the network")),
        vhq.Member(agent=vhq.Agent(role="member 3", goal="work in the network")),
    ],
    formation=vhq.Formation.SQUAD,
    )

    assert network.formation == vhq.Formation.SQUAD


def test_doc_an_config_c():
    import versionhq as vhq

    network = vhq.AgentNetwork(
    members=[
        vhq.Member(agent=vhq.Agent(role="member 1", goal="work in the network"), tasks=[vhq.Task(description="Run a demo 1")]),
        vhq.Member(agent=vhq.Agent(role="member 2", goal="work in the network"), tasks=[vhq.Task(description="Run a demo")]),
        vhq.Member(agent=vhq.Agent(role="member 3", goal="work in the network")),
    ],
    )

    res, tg = network.launch()

    assert isinstance(res, vhq.TaskOutput)
    assert isinstance(tg, vhq.TaskGraph)


def test_doc_an_config_d():
    import versionhq as vhq

    network = vhq.AgentNetwork(
    members=[
        vhq.Member(agent=vhq.Agent(role="member 1", goal="work in the network"), tasks=[vhq.Task(description="Run a demo 1")]),
        vhq.Member(agent=vhq.Agent(role="member 2", goal="work in the network"), tasks=[vhq.Task(description="Run a demo 2")]),
        vhq.Member(agent=vhq.Agent(role="member 3", goal="work in the network")),
    ],
    process=vhq.TaskHandlingProcess.CONSENSUAL,
    consent_trigger=lambda x: True, # consent trigger event is a MUST for TaskHandlingProcess.CONSENSUAL
    )

    res, tg = network.launch()

    assert isinstance(res, vhq.TaskOutput)
    assert isinstance(tg, vhq.TaskGraph)
