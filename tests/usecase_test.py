def test_demo_hello_world():
    import versionhq as vhq

    agent = vhq.Agent(role='Demo Manager', goal='run a demo successfully')
    task = vhq.Task(description='hello world')
    res = task.execute_sync(agent=agent)

    assert res.raw


def test_demo_network():
    import versionhq as vhq

    network = vhq.form_agent_network(
        task='draft a promotional email for the given client',
        expected_outcome='email subject and body in string',
        context="use email_subject and email_body as keys in your response."
    )
    assert isinstance(network, vhq.Team) and isinstance(network.formation, vhq.Formation)

    # res = network.launch()



def test_demo_agent_customization():
    import versionhq as vhq

    agent = vhq.Agent(
        role='Demo Manager',
        goal='run a demo successfully',
        llm='gemini-2.0',
        llm_config = dict(
            temperature=1,
            top_p=0.1,
            n=1,
            stop="test",
        )
    )


    import pathlib
    current_path = pathlib.Path(__file__).parent.resolve()

    agent.knowledge_sources = [
        'https://business.linkedin.com',
        f'{current_path}/demo.csv',
    ]
    agent.use_memory = True

    assert "gemini-2.0" in agent.llm.model
    assert agent.llm.temperature == 1 and agent.llm.top_p == 0.1 and agent.llm.n==1 and agent.llm.stop == "test"
    assert agent.knowledge_sources == ['https://business.linkedin.com', f'{current_path}/demo.csv',]
    assert agent.use_memory == True

# {'email_subject':
#  'Unlock Exclusive Benefits Just for You!',
#  'email_body': "Dear Valued Customer,\n\nWe're excited to introduce our latest offer tailored just for you! As a token of our appreciation, we invite you to enjoy exclusive benefits that will enhance your experience with us.\n\n🌟 **Exclusive Offer:** Receive 20% off your next purchase when you use the code THANKYOU20 at checkout.\n\n🛍️ **Why Choose Us?**\n- Premium quality products that you can trust.\n- Fast and reliable shipping right to your door.\n- Exceptional customer service ready to assist you.\n\n**Hurry! This offer is valid until [insert expiration date], so make sure to take advantage of it today!**\n\nThank you for being a valued member of our community. We look forward to serving you again soon!\n\nBest Regards,\n[Your Company Name] Team\n[Contact Information]\n[Website URL]"}
