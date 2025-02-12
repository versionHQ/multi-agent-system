def test_minimum_inputs():
    import versionhq as vhq

    network = vhq.form_agent_network(
        task="Describe the color of the sky by weather condition",
        expected_outcome="Dict that pairs color of the sky with specific weather condition"
    )
    assert isinstance(network, vhq.AgentNetwork)
    assert network.members
    assert network.tasks
    assert isinstance(network.formation, vhq.Formation)
    assert network.managers if network.formation in (vhq.Formation.SUPERVISING, vhq.Formation.HYBRID) else not network.managers


def _test_specific_formation():
    import versionhq as vhq

    formations_to_test = [
        vhq.Formation.NETWORK,
        vhq.Formation.RANDOM,
        vhq.Formation.SOLO,
        vhq.Formation.SUPERVISING,
        vhq.Formation.UNDEFINED,
        "network",
        1,
        "dummy",
        1000,
        None
    ]

    for item in formations_to_test:
        network = vhq.form_agent_network(
            task="Describe the color of the sky by weather condition",
            expected_outcome="Dict that pairs color of the sky with specific weather condition",
            formation=item,

        )

        assert isinstance(network, vhq.AgentNetwork)
        assert network.formation == item if isinstance(item, vhq.Formation) and item != vhq.Formation.UNDEFINED else isinstance(network.formation, vhq.Formation)
        assert network.members
        assert network.tasks
        assert network.managers is not None if network.formation == vhq.Formation.SUPERVISING else not network.manager_tasks
