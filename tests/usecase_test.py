"""frontend home use cases pre-test"""

def test_demo_hello_world():
    import versionhq as vhq

    task = vhq.Task(description='hello world')
    res = task.execute()

    assert res.raw


def test_demo_network():
    import versionhq as vhq

    network = vhq.form_agent_network(
        task='draft a promotional email for the given client',
        expected_outcome='email subject and body in string',
        context="use email_subject and email_body as keys in your response."
    )

    assert isinstance(network, vhq.AgentNetwork) and isinstance(network.formation, vhq.Formation)

    res, _ = network.launch()

    assert res.pydantic is not None



def test_demo_agent_customization():
    import versionhq as vhq
    import pathlib
    current_path = pathlib.Path(__file__).parent.resolve()

    agent = vhq.Agent(
        role='Demo Manager',
        goal='run a demo successfully'
    )

    agent.update(
        llm='gemini-2.0',
        llm_config = dict(
            temperature=1,
            top_p=0.1,
            n=1,
            stop="test",
        ),
        knowledge_sources = [
            'https://business.linkedin.com',
            f'{current_path}/demo.csv',
        ],
        with_memory=True
    )

    assert "gemini-2.0" in agent.llm.model
    assert agent.llm.temperature == 1 and agent.llm.top_p == 0.1 and agent.llm.n == 1 and agent.llm.stop == "test"
    assert agent.knowledge_sources == ['https://business.linkedin.com', f'{current_path}/demo.csv',]
    assert agent.with_memory == True


def test_solo_tg_eval():
    import versionhq as vhq

    network = vhq.form_agent_network(
        task="test",
        expected_outcome="test"
    )

    res, tg = network.launch()
    eval = tg.evaluate(
        eval_criteria=["cost", "", "{criteria_3}"]
    )
    assert isinstance(eval, vhq.Evaluation)


def test_llm_as_judge():
    from versionhq._utils.llm_as_a_judge import LLMJudge, generate_summaries, validate
    class MockSummarizer:
        def summarize(self, text: str) -> str:
            return f"Summary of: {text[:50]}..."

    from pathlib import Path
    current_dir = Path(__file__).parent
    file_path = current_dir / "sample.json"
    summaries = generate_summaries(file_path=file_path, summarizer=MockSummarizer())
    results = validate(judge=LLMJudge(), data=summaries, threshold=0.6)

    assert results is not None
