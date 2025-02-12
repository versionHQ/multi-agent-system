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

    # res = network.launch()



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
        use_memory=True
    )

    assert "gemini-2.0" in agent.llm.model
    assert agent.llm.temperature == 1 and agent.llm.top_p == 0.1 and agent.llm.n==1 and agent.llm.stop == "test"
    assert agent.knowledge_sources == ['https://business.linkedin.com', f'{current_path}/demo.csv',]
    assert agent.use_memory == True
