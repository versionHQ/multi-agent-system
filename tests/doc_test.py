"""
Test use cases on index.md and quickstart.md
"""

def test_doc_index_a():
    import versionhq as vhq

    network = vhq.form_agent_network(
        task="YOUR AMAZING TASK OVERVIEW",
        expected_outcome="YOUR OUTCOME EXPECTATION",
    )
    res, _ = network.launch()
    assert res.raw is not None


def test_doc_index_b():
    import versionhq as vhq
    from pydantic import BaseModel

    class CustomOutput(BaseModel):
        test1: str
        test2: list[str]

    def dummy_func(message: str, test1: str, test2: list[str]) -> str:
        return f"""{message}: {test1}, {", ".join(test2)}"""


    agent = vhq.Agent(role="demo", maxit=1)

    task = vhq.Task(
        description="Amazing task",
        pydantic_output=CustomOutput,
        callback=dummy_func,
        callback_kwargs=dict(message="Hi! Here is the result: ")
    )

    res = task.execute(agent=agent, context="amazing context to consider.")
    assert "Hi! Here is the result:" in res.callback_output
    assert [getattr(res.pydantic, k) for k, v in CustomOutput.model_fields.items()]


def test_doc_index_c():
    import versionhq as vhq

    agent_a = vhq.Agent(role="agent a", llm="llm-of-your-choice")
    agent_b = vhq.Agent(role="agent b", llm="llm-of-your-choice")

    task_1 = vhq.Task(
        description="Analyze the client's business model.",
        response_fields=[vhq.ResponseField(title="test1", data_type=str, required=True),],
        allow_delegation=True
    )

    task_2 = vhq.Task(
        description="Define a cohort.",
        response_fields=[vhq.ResponseField(title="test1", data_type=int, required=True),],
        allow_delegation=False
    )

    network =vhq.AgentNetwork(
        members=[
            vhq.Member(agent=agent_a, is_manager=False, tasks=[task_1]),
            vhq.Member(agent=agent_b, is_manager=True, tasks=[task_2]),
        ],
    )
    res, tg = network.launch()

    assert isinstance(res, vhq.TaskOutput)
    assert isinstance(tg, vhq.TaskGraph)
    assert agent_b.key in task_1.processed_agents # delegated by agent_a
    assert agent_b.key in task_2.processed_agents
