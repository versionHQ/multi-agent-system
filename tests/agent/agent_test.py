import os
from unittest.mock import patch
from typing import Callable, Any

from versionhq.agent.model import Agent
from versionhq.agent.TEMPLATES.Backstory import BACKSTORY_SHORT, BACKSTORY_FULL
from versionhq.llm.model import DEFAULT_MODEL_NAME, LLM
from versionhq.tool.model import Tool
from versionhq.tool.decorator import tool

MODEL_NAME = os.environ.get("DEFAULT_MODEL_NAME", "gpt-3.5-turbo")


def test_build_agent_with_minimal_input():
    agent = Agent(role="analyst")

    assert agent.role == "analyst"
    assert agent.backstory == BACKSTORY_SHORT.format(role=agent.role.lower(), goal="")
    assert isinstance(agent.llm, LLM)
    assert agent.llm.model == DEFAULT_MODEL_NAME
    # assert agent.llm.api_key == LITELLM_API_KEY
    assert agent.tools == []
    assert isinstance(agent.func_calling_llm, LLM)


def test_build_agent_from_config():
    agent = Agent(config=dict(role="analyst", goal="analyze the company's website and retrieve the product overview"))

    assert agent.role == "analyst"
    assert agent.backstory == BACKSTORY_SHORT.format(role=agent.role.lower(), goal=agent.goal.lower())
    assert isinstance(agent.llm, LLM)
    assert agent.llm.model == DEFAULT_MODEL_NAME
    # assert agent.llm.api_key == LITELLM_API_KEY
    assert agent.tools == []
    assert isinstance(agent.func_calling_llm, LLM)


def test_build_agent_with_backstory():
    agent = Agent(
        role="analyst",
        goal="analyze the company's website and retrieve the product overview",
        backstory="You are competitive analysts who have abundand knowledge in marketing, product management."
    )

    assert agent.role == "analyst"
    assert agent.goal == "analyze the company's website and retrieve the product overview"
    assert agent.backstory == "You are competitive analysts who have abundand knowledge in marketing, product management."
    assert isinstance(agent.llm, LLM)
    assert agent.llm.model == DEFAULT_MODEL_NAME
    # assert agent.llm.api_key == LITELLM_API_KEY
    assert agent.tools == []
    assert isinstance(agent.func_calling_llm, LLM)


def test_build_agent():
    agent = Agent(
        role="analyst",
        goal="analyze the company's website and retrieve the product overview",
        skills=["financial analysis", "product management", ]
    )

    assert agent.role == "analyst"
    assert agent.backstory == BACKSTORY_FULL.format(role=agent.role.lower(), goal=agent.goal.lower(), skills=", ".join([item for item in agent.skills]), tools="")
    assert isinstance(agent.llm, LLM)
    assert agent.llm.model == DEFAULT_MODEL_NAME
    # assert agent.llm.api_key == LITELLM_API_KEY
    assert agent.tools == []
    assert isinstance(agent.func_calling_llm, LLM)


def test_build_agent_with_llm():
    agent = Agent(
        role="analyst",
        goal="analyze the company's website and retrieve the product overview",
        skills=["financial analysis", "product management", ],
        llm="gpt-4o"
    )

    assert agent.role == "analyst"
    assert agent.role in agent.backstory
    assert agent.goal in agent.backstory
    assert [item in agent.backstory for item in agent.skills]
    assert isinstance(agent.llm, LLM)
    assert agent.llm.model == "gpt-4o"
    # assert agent.llm.api_key == LITELLM_API_KEY
    assert agent.tools == []
    assert isinstance(agent.func_calling_llm, LLM)


def test_build_agent_with_llm_config():
    def dummy_func() -> str:
        return "dummy"

    llm_params = dict(deployment_name="gemini-1.5", max_tokens=4000, logprobs=False, abc="dummy key")
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
        role="analyst",
        goal="run test on llm instance",
        llm=llm_params,
        callbacks=[dummy_func],
        llm_config=llm_config
    )

    assert isinstance(agent.llm, LLM)
    assert isinstance(agent.func_calling_llm, LLM)
    assert agent.llm.model == "gemini/gemini-1.5-flash"
    assert agent.llm.callbacks == [dummy_func]

    import litellm
    valid_params = litellm.get_supported_openai_params(model="gemini/gemini-1.5-flash")
    config = llm_params.update(llm_config)
    for key in valid_params:
        if config and [k for k in config.keys() if k == key]:
            assert getattr(agent.llm, key) == config[key] or agent.llm.llm_config[key] == config[key]


def test_build_agent_with_llm_object():
    def dummy_func() -> str:
        return "dummy"

    llm = LLM(model="gemini-1.5", llm_config=dict(max_tokens=4000, logprobs=False))
    agent = Agent(role="analyst", llm=llm, callbacks=[dummy_func])

    assert isinstance(agent.llm, LLM)
    assert isinstance(agent.func_calling_llm, LLM)
    assert agent.llm.model == "gemini/gemini-1.5-flash"
    # assert agent.llm.api_key is not None
    assert agent.llm.llm_config["max_tokens"] == 4000
    assert agent.llm.llm_config["logprobs"] == False
    assert agent.llm.callbacks == [dummy_func]


def test_build_agent_with_llm_and_func_llm_config():
    def dummy_func() -> str:
        return "dummy"

    llm_params = dict(deployment_name="gemini-1.5", max_tokens=4000, logprobs=False, abc="dummy key")
    agent = Agent(role="analyst", func_calling_llm=llm_params, callbacks=[dummy_func])

    assert isinstance(agent.llm, LLM) and agent.llm.model == DEFAULT_MODEL_NAME
    assert isinstance(agent.func_calling_llm, LLM)
    assert agent.func_calling_llm.model == "gemini/gemini-1.5-flash" if agent.func_calling_llm._supports_function_calling() else DEFAULT_MODEL_NAME


def test_build_agent_with_llm_and_func_llm_object():
    def dummy_func() -> str:
        return "dummy"

    llm = LLM(model="gemini-1.5", llm_config=dict(max_tokens=4000, logprobs=False))
    agent = Agent(
        role="analyst",
        goal="analyze the company's website and retrieve the product overview",
        llm=llm,
        func_calling_llm=llm,
        llm_config=dict(),
        callbacks=[dummy_func]
    )
    assert isinstance(agent.llm, LLM) and isinstance(agent.func_calling_llm, LLM)
    assert agent.func_calling_llm.model == "gemini/gemini-1.5-flash" if agent.func_calling_llm._supports_function_calling() else DEFAULT_MODEL_NAME
    assert agent.func_calling_llm.llm_config["max_tokens"] == 4000
    assert agent.func_calling_llm.llm_config["logprobs"] == False
    assert agent.func_calling_llm.callbacks == [dummy_func]
    assert isinstance(agent.func_calling_llm, LLM)


def test_agent_with_tools():
    def empty_func():
        return "empty function"

    def custom_tool(query: str) -> str:
        return query

    agent = Agent(role="demo", goal="test a tool", tools=[dict(name="tool 1", func=empty_func), lambda x: x, custom_tool,])

    assert agent.tools[0]._run() ==  "empty function"
    assert agent.tools[0].name == "tool 1"
    assert agent.tools[1]._run(x="hey") == "hey"
    assert agent.tools[1].name == "random_func"
    assert agent.tools[2]._run(query="hey") == "hey"
    assert agent.tools[2].name == "custom_tool"


def test_agent_with_custom_tools():
    def send_message(message: str) -> str:
        return message + "_demo"

    class CustomTool(Tool):
        name: str = "custom tool"
        func: Callable[..., Any]

    tool = CustomTool(func=send_message)
    agent = Agent(role="demo", goal="test a tool", tools=[tool])

    assert agent.tools[0] is tool
    assert agent.tools[0]._run(message="hi") == "hi_demo"
    assert agent.tools[0].name == "custom tool"


# @pytest.mark.vcr(filter_headers=["authorization"])
def test_agent_custom_max_iterations():
    from versionhq.task.model import Task

    @tool
    def get_final_answer() -> int:
        """Get the final answer but don't give it yet, just re-use this tool non-stop."""
        return 42

    agent = Agent(role="demo", goal="test goal", maxit=1, allow_delegation=False, tools=[get_final_answer])

    with patch.object(LLM, "call", wraps=LLM(model=DEFAULT_MODEL_NAME).call) as private_mock:
        task = Task(
            description="The final answer is 42. But don't give it yet, instead keep using the `get_final_answer` tool.",
            can_use_agent_tools=True
        )
        agent.execute_task(task=task)
        assert private_mock.call_count == 1


def test_agent_with_knowledge_sources():
    from versionhq.knowledge.source import StringKnowledgeSource
    from versionhq.knowledge.source_docling import DoclingSource
    from versionhq.task.model import Task

    content = "Kuriko's favorite color is gold, and she enjoy Japanese food."
    string_source = StringKnowledgeSource(content=content)
    html = "https://github.blog/security/vulnerability-research/cybersecurity-researchers-digital-detectives-in-a-connected-world/"
    knowledge_sources = [content, string_source, html,]

    agent = Agent(role="Information Agent", goal="Provide information based on knowledge sources", knowledge_sources=knowledge_sources)

    assert agent._knowledge.collection_name == agent.key
    assert [isinstance(item, StringKnowledgeSource | DoclingSource) for item in agent.knowledge_sources]
    assert agent._knowledge.embedder_config == agent.embedder_config
    assert agent._knowledge.storage and agent._knowledge.storage.embedding_function
    assert agent._knowledge.storage.app is not None
    assert agent._knowledge.storage.collection_name is not None

    task = Task(description="Answer the following question: What is Kuriko's favorite color?")

    with patch("versionhq.knowledge.storage.KnowledgeStorage") as MockKnowledge:
        mock_knowledge_instance = MockKnowledge.return_value
        mock_knowledge_instance.query.return_value = [{ "content": content }]
        res = task.execute(agent=agent)
        assert "gold" in res.raw.lower()


def test_using_contextual_memory():
    from versionhq.task.model import Task
    from versionhq.task.evaluation import Evaluation
    from versionhq.storage.rag_storage import RAGStorage
    from versionhq.storage.ltm_sqlite_storage import LTMSQLiteStorage

    agent = Agent(role="Researcher", goal="You research about math.", with_memory=True)
    assert agent.short_term_memory.storage and isinstance(agent.short_term_memory.storage, RAGStorage) and agent.short_term_memory.storage.type == "stm"
    assert agent.long_term_memory.storage and isinstance(agent.long_term_memory.storage, LTMSQLiteStorage)

    task = Task(description="Research a topic to teach a kid aged 6 about math.", should_evaluate=True)
    res = task.execute(agent=agent)
    assert isinstance(res.evaluation, Evaluation) and res.evaluation.suggestion_summary and res.evaluation.aggregate_score is not None


def test_disabled_memory():
    from unittest.mock import patch
    from versionhq.task.model import Task
    from versionhq.memory.contextual_memory import ContextualMemory

    agent = Agent(role="Researcher", goal="You research about math.", with_memory=False)
    task = Task(description="Research a topic to teach a kid aged 6 about math.")

    with patch.object(ContextualMemory, "build_context_for_task") as contextual_mem:
        task.execute(agent=agent)
        contextual_mem.assert_not_called()


def test_agent_with_memory_config():
    from versionhq.storage.ltm_sqlite_storage import LTMSQLiteStorage

    agent_1 = Agent(role="Researcher", goal="You research about math.", with_memory=True, memory_config=dict(provider="mem0"))
    agent_2 = Agent(role="Researcher", goal="You research about math.", with_memory=True, memory_config=dict(provider="mem0", user_id="123"))


    assert agent_1.short_term_memory is not None
    assert agent_1.short_term_memory.memory_provider == "mem0" and agent_1.short_term_memory.storage.memory_type == "stm"
    assert agent_1.long_term_memory and isinstance(agent_1.long_term_memory.storage, LTMSQLiteStorage)
    assert agent_1.user_memory is None

    assert agent_2.short_term_memory is not None
    assert agent_2.short_term_memory.memory_provider == "mem0" and agent_2.short_term_memory.storage.memory_type == "stm"
    assert agent_2.long_term_memory and isinstance(agent_2.long_term_memory.storage, LTMSQLiteStorage)
    assert agent_2.user_memory and agent_2.user_memory.storage and agent_2.user_memory.storage.memory_type == "user"


def test_updating_llm():
    import versionhq as vhq

    agent = vhq.Agent(role="Researcher", goal="You research about math.")
    agent._update_llm(llm="gemini-2.0")
    assert isinstance(agent.llm, vhq.LLM) and "gemini-2.0" in agent.llm.model

    agent._update_llm(llm_config=dict(max_tokens=10000))
    assert agent.llm.llm_config["max_tokens"] == 10000

    agent._update_llm(llm="deepseek", llm_config=dict(max_tokens=500))
    assert isinstance(agent.llm, vhq.LLM) and "deepseek-r1" in agent.llm.model and agent.llm.llm_config["max_tokens"] == 500


def test_start_with_tools():
    import versionhq as vhq

    def demo_func() -> str:
        return "demo"

    my_tool = vhq.Tool(func=demo_func)

    agent = vhq.Agent(
        role="Tool Handler",
        goal="efficiently use the given tools",
        tools=[my_tool, ]
    )

    res = agent.start(tool_res_as_final=True)
    assert res.tool_output == "demo"
