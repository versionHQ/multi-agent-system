"""Test use cases in docs/core/agents.md"""

def test_docs_core_agent_a():
    import versionhq as vhq
    agent = vhq.Agent(
        role="Marketing Analyst",
        goal="Coping with price competition in saturated markets",
    )

    assert agent.id and agent.role == "Marketing Analyst" and agent.goal == "Coping with price competition in saturated markets" and agent.backstory is not None

    res = agent.start(context="Planning a new campaign promotion starting this summer")

    assert isinstance(res, vhq.TaskOutput)
    assert res.json_dict


def test_docs_core_agent_b1():
    from versionhq import Agent
    agent = Agent(role="Marketing Analyst", llm="gemini-2.0")

    from versionhq.llm.model import LLM
    assert "gemini-2.0" in agent.llm.model and isinstance(agent.llm, LLM)


def test_docs_core_agent_b3():
    import versionhq as vhq

    agent = vhq.Agent(
        role="Marketing Analyst",
        goal="Coping with price competition in saturated markets"
    )

    tool = vhq.Tool(func=lambda x: x)
    agent.update(
        tools=[tool], # adding tools
        goal="my new goal", # updating the goal (this will trigger the update of developer_prompt)
        max_rpm=3, # default = 1
        knowledge_sources=["testing", "testing2"], # adding knowledge sources (this will trigger the storage creation)
        memory_config={"user_id": "0000"}, # adding memories
        llm="gemini-2.0", # updating LLM (Valid llm_config will be inherited to the new model.)
        use_developer_prompt=False,
        dummy="I am dummy" # <- invalid field - automatically ignored
    )

    from versionhq.agent.rpm_controller import RPMController
    assert agent.tools == [tool]
    assert agent.goal == "my new goal" and agent.role == "Marketing Analyst" and "my new goal" in agent.backstory
    assert isinstance(agent._rpm_controller, RPMController)
    assert isinstance(agent._knowledge, vhq.Knowledge) and agent.knowledge_sources == ["testing", "testing2"]
    assert isinstance(agent.user_memory, vhq.UserMemory)
    assert "gemini-2.0" in agent.llm.model and agent.llm.provider == "gemini"
    assert agent.use_developer_prompt == False
    assert isinstance(agent.llm, vhq.LLM)


def test_docs_core_agent_c1():
    from versionhq import Agent
    agent = Agent(
        role="Marketing Analyst",
        goal="Coping with price competition in saturated markets"
    )

    assert agent.backstory == "You are an expert marketing analyst with relevant skills and abilities to query relevant information from the given knowledge sources. Leveraging these, you will identify competitive solutions to achieve the following goal: coping with price competition in saturated markets."


def test_docs_core_agent_c2():
    from versionhq import Agent

    agent = Agent(
        role="Marketing Analyst",
        goal="Coping with increased price competition in saturated markets.",
        backstory="You are a marketing analyst for a company in a saturated market. The market is becoming increasingly price-competitive, and your company's profit margins are shrinking. Your primary goal is to develop and implement strategies to help your company maintain its market share and profitability in this challenging environment."
    )
    assert agent.backstory == "You are a marketing analyst for a company in a saturated market. The market is becoming increasingly price-competitive, and your company's profit margins are shrinking. Your primary goal is to develop and implement strategies to help your company maintain its market share and profitability in this challenging environment."


def test_docs_core_agent_e1():
    import json
    from typing import Dict, Any
    import versionhq as vhq

    def format_response(res: str = None) -> str | Dict[str, Any]:
        try:
            r = json.dumps(eval(res))
            formatted_res = json.loads(r)
            return formatted_res
        except:
            return res

    agent = vhq.Agent(
        role="Marketing Analyst",
        goal="Coping with increased price competition in saturated markets.",
        callbacks=[format_response]
    )

    import litellm
    assert litellm.callbacks == agent.callbacks


def test_docs_core_agent_e2():
    import json
    from typing import Dict, Any
    import versionhq as vhq


    def assessment(res: str) -> str:
        try:
            sub_agent = vhq.Agent(role="Validator", goal="Validate the given solutions.")
            task = vhq.Task(description=f"""Assess the given solution based on feasibilities and fits to client's strategies, then refine the solution if necessary.
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

    agent = vhq.Agent(
        role="Marketing Analyst",
        goal="Build solutions to address increased price competition in saturated markets",
        callbacks=[assessment, format_response]
    )

    import litellm
    assert litellm.callbacks == agent.callbacks and len(agent.callbacks) == 2


def test_docs_core_agent_f():
    import versionhq as vhq

    agent = vhq.Agent(
        role="Marketing Analyst",
        respect_context_window=False,
        max_execution_time=60,
        max_rpm=5,
        llm_config=dict(
                temperature=1,
                top_p=0.1,
                n=1,
                stop="answer",
                dummy="I am dummy" # <- invalid field will be ignored automatically.
            )
        )
    assert isinstance(agent.llm, vhq.LLM)
    assert agent.llm.llm_config["temperature"] == 1
    assert agent.llm.llm_config["top_p"] == 0.1
    assert agent.llm.llm_config["n"] == 1
    assert agent.llm.llm_config["stop"] == "answer"


def test_docs_core_agent_g():
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

    res = task.execute(agent=agent)
    assert "gold" in res.raw.lower()


def test_docs_core_agent_h():
    from versionhq import Agent

    agent = Agent(
        role="Researcher",
        goal="You research about math.",
        with_memory=True
    )

    from versionhq.memory.model import ShortTermMemory, LongTermMemory
    assert isinstance(agent.short_term_memory, ShortTermMemory) and isinstance(agent.long_term_memory, LongTermMemory)


def test_docs_core_agent_z():
    from versionhq import Agent

    agent = Agent(
        config=dict(
            role="Marketing Analyst",
            goal="Coping with increased price competition in saturated markets.",
        )
    )

    assert agent.role == "Marketing Analyst" and agent.goal == "Coping with increased price competition in saturated markets."
