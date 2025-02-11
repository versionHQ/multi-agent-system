"""
Test use cases on index.md and quickstart.md
"""

def test_doc_index_a():
    import versionhq as vhq

    network = vhq.form_agent_network(
        task="YOUR AMAZING TASK OVERVIEW",
        expected_outcome="YOUR OUTCOME EXPECTATION",
    )
    res = network.launch()
    assert res.raw is not None


def test_doc_index_b():
    import versionhq as vhq
    from pydantic import BaseModel

    class CustomOutput(BaseModel):
        test1: str
        test2: list[str]

    def dummy_func(message: str, test1: str, test2: list[str]) -> str:
        return f"""{message}: {test1}, {", ".join(test2)}"""


    agent = vhq.Agent(role="demo", goal="amazing project goal", maxit=1)

    task = vhq.Task(
        description="Amazing task",
        pydantic_output=CustomOutput,
        callback=dummy_func,
        callback_kwargs=dict(message="Hi! Here is the result: ")
    )

    res = task.execute(agent=agent, context="amazing context to consider.")
    assert "Hi! Here is the result:" in res.callback_output
    assert [getattr(res.pydantic, k) for k, v in CustomOutput.model_fields.items()]
