def test_rag_tool():
    import versionhq as vhq
    rt = vhq.RagTool(url="https://github.com/chroma-core/chroma/issues/3233", query="What is the next action plan?")
    res = rt.run()

    assert rt.text is not None
    assert res is not None


def test_rag_tool_with_agent():
    import versionhq as vhq

    agent = vhq.Agent(role="RAG Tool Tester")
    rt = vhq.RagTool(url="https://github.com/chroma-core/chroma/issues/3233", query="What is the next action plan?")
    res = rt.run(agent=agent)

    assert agent.knowledge_sources is not None
    assert rt.text is not None
    assert res is not None


def test_use_rag_tool():
    import versionhq as vhq

    rt = vhq.RagTool(url="https://github.com/chroma-core/chroma/issues/3233", query="What is the next action plan?")
    agent = vhq.Agent(role="RAG Tool Tester", tools=[rt])
    task = vhq.Task(description="return a simple response", can_use_agent_tools=True, tool_res_as_final=True)
    res = task.execute(agent=agent)

    assert res.raw is not None
    assert res.tool_output is not None
