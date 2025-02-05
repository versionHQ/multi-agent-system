

"""
Test for use cases in the documentation.
"""

def test_quick_start():
    from versionhq import form_agent_network

    network = form_agent_network(
        task="YOUR AMAZING TASK OVERVIEW",
        expected_outcome="YOUR OUTCOME EXPECTATION",
    )
    res = network.launch()

    assert res.raw is not None
