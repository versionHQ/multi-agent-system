"""
Pytest use cases in the documentation.
"""

def test_quick_start():
    """quickstart/quickstart.md"""

    from versionhq import form_agent_network

    network = form_agent_network(
        task="YOUR AMAZING TASK OVERVIEW",
        expected_outcome="YOUR OUTCOME EXPECTATION",
    )
    res = network.launch()

    assert res.raw is not None


def test_doc_agent():
    """core/Agent.md"""

    def a():
        from versionhq import Agent
        agent = Agent(
            role="Marketing Analyst",
            goal="Coping with price competition in saturated markets"
        )

        assert agent.id and agent.role == "Marketing Analyst" and agent.goal == "Coping with price competition in saturated markets" and agent.backstory is not None

    def b():
        from versionhq import Agent
        agent = Agent(
            role="Marketing Analyst",
            goal="Coping with price competition in saturated markets",
            llm="gemini-2.0",
        )

        from versionhq.llm.model import LLM
        assert "gemini-2.0" in agent.llm.model and isinstance(agent.llm, LLM)


    def c_1():
        from versionhq import Agent
        agent = Agent(
            role="Marketing Analyst",
            goal="Coping with price competition in saturated markets"
        )

        assert agent.backstory == "You are an expert marketing analyst with relevant skillsets and abilities to query relevant information from the given knowledge sources. Leveraging these, you will identify competitive solutions to achieve the following goal: coping with price competition in saturated markets."


    def c_2():
        from versionhq import Agent

        agent = Agent(
            role="Marketing Analyst",
            goal="Coping with increased price competition in saturated markets.",
            backstory="You are a marketing analyst for a company in a saturated market. The market is becoming increasingly price-competitive, and your company's profit margins are shrinking. Your primary goal is to develop and implement strategies to help your company maintain its market share and profitability in this challenging environment."
        )
        assert agent.backstory == "You are a marketing analyst for a company in a saturated market. The market is becoming increasingly price-competitive, and your company's profit margins are shrinking. Your primary goal is to develop and implement strategies to help your company maintain its market share and profitability in this challenging environment."


    def e_1():
        import json
        from typing import Dict, Any
        from versionhq import Agent

        def format_response(res: str = None) -> str | Dict[str, Any]:
            try:
                r = json.dumps(eval(res))
                formatted_res = json.loads(r)
                return formatted_res
            except:
                return res

        agent = Agent(
            role="Marketing Analyst",
            goal="Coping with increased price competition in saturated markets.",
            callbacks=[format_response]
        )

        import litellm
        assert litellm.callbacks == agent.callbacks


    def e_2():
        import json
        from typing import Dict, Any
        from versionhq import Agent, Task

        def assessment(res: str) -> str:
            try:
                sub_agent = Agent(role="Validator", goal="Validate the given solutions.")
                task = Task(description=f"""Assess the given solution based on feasibilities and fits to client's strategies, then refine the solution if necessary.
        Solution: {res}
        """)
                r = task.sync_execute(agent=sub_agent)
                return r.raw

            except:
                return res

        def format_response(res: str = None) -> str | Dict[str, Any]:
            try:
                r = json.dumps(eval(res))
                formatted_res = json.loads(r)
                return formatted_res
            except:
                return res

        agent = Agent(
            role="Marketing Analyst",
            goal="Build solutions to address increased price competition in saturated markets",
            callbacks=[assessment, format_response]
        )

        import litellm
        assert litellm.callbacks == agent.callbacks and len(agent.callbacks) == 2


    def f():
        from versionhq import Agent

        llm_config = dict(
            temperature=1,
            top_p=0.1,
            n=1,
            stream=False,
            stream_options=None,
            stop="test",
            max_completion_tokens=10000,
            dummy="I am dummy"
        )

        agent = Agent(
            role="Marketing Analyst",
            goal="Coping with increased price competition in saturated markets.",
            respect_context_window=False,
            max_tokens=3000,
            max_execution_time=60,
            max_rpm=5,
            llm_config=llm_config
        )

        assert agent.llm.max_tokens == 3000
        assert agent.llm.temperature == 1
        assert agent.llm.top_p == 0.1
        assert agent.llm.n == 1
        assert agent.llm.stop=="test"


    def g():
        from versionhq import Agent
        from versionhq.task.model import Task
        from versionhq.knowledge.source import StringKnowledgeSource

        content = "Kuriko's favorite color is gold, and she enjoy Japanese food."
        string_source = StringKnowledgeSource(content=content)

        agent = Agent(
            role="Information Agent",
            goal="Provide information based on knowledge sources",
            knowledge_sources=[string_source,]
        )

        task = Task(
            description="Answer the following question: What is Kuriko's favorite color?"
        )

        res = task.execute_sync(agent=agent)
        assert "gold" in res.raw.lower()


    def h():
        from versionhq import Agent

        agent = Agent(
            role="Researcher",
            goal="You research about math.",
            use_memory=True
        )

        from versionhq.memory.model import ShortTermMemory, LongTermMemory
        assert isinstance(agent.short_term_memory, ShortTermMemory) and isinstance(agent.long_term_memory, LongTermMemory)


    def z():
        from versionhq import Agent

        agent = Agent(
            config=dict(
                role="Marketing Analyst",
                goal="Coping with increased price competition in saturated markets.",
            )
        )

        assert agent.role == "Marketing Analyst" and agent.goal == "Coping with increased price competition in saturated markets."

    a(); b(); c_1(); c_2(); e_1(); e_2(); f(); g(); h(); z()
