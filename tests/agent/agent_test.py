import os
from unittest import mock
from unittest.mock import patch
import pytest
from typing import Callable, Any

from versionhq.agent.model import Agent
from versionhq.agent.TEMPLATES.Backstory import BACKSTORY_SHORT, BACKSTORY_FULL
from versionhq.llm.model import LLM, DEFAULT_MODEL_NAME
from versionhq.tool.model import Tool
from versionhq.tool.decorator import tool

MODEL_NAME = os.environ.get("DEFAULT_MODEL_NAME", "gpt-3.5-turbo")
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY")


def test_build_agent_with_minimal_input():
    agent = Agent(
        role="analyst",
        goal="analyze the company's website and retrieve the product overview"
    )

    assert agent.role == "analyst"
    assert agent.backstory == BACKSTORY_SHORT.format(role=agent.role.lower(), goal=agent.goal.lower())
    assert isinstance(agent.llm, LLM)
    assert agent.llm.model == DEFAULT_MODEL_NAME
    assert agent.llm.api_key == LITELLM_API_KEY
    assert agent.tools == []


def test_build_agent_from_config():
    agent = Agent(config=dict(role="analyst", goal="analyze the company's website and retrieve the product overview"))

    assert agent.role == "analyst"
    assert agent.backstory == BACKSTORY_SHORT.format(role=agent.role.lower(), goal=agent.goal.lower())
    assert isinstance(agent.llm, LLM)
    assert agent.llm.model == DEFAULT_MODEL_NAME
    assert agent.llm.api_key == LITELLM_API_KEY
    assert agent.tools == []


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
    assert agent.llm.api_key == LITELLM_API_KEY
    assert agent.tools == []


def test_build_agent():
    agent = Agent(
        role="analyst",
        goal="analyze the company's website and retrieve the product overview",
        skillsets=["financial analysis", "product management", ]
    )

    assert agent.role == "analyst"
    assert agent.backstory == BACKSTORY_FULL.format(role=agent.role.lower(), goal=agent.goal.lower(), skills=", ".join([item for item in agent.skillsets]), tools="")
    assert isinstance(agent.llm, LLM)
    assert agent.llm.model == DEFAULT_MODEL_NAME
    assert agent.llm.api_key == LITELLM_API_KEY
    assert agent.tools == []


def test_build_agent_with_llm():
    agent = Agent(
        role="analyst",
        goal="analyze the company's website and retrieve the product overview",
        skillsets=["financial analysis", "product management", ],
        llm="gpt-4o"
    )

    assert agent.role == "analyst"
    assert agent.role in agent.backstory
    assert agent.goal in agent.backstory
    assert [item in agent.backstory for item in agent.skillsets]
    assert isinstance(agent.llm, LLM)
    assert agent.llm.model == "gpt-4o"
    assert agent.llm.api_key == LITELLM_API_KEY
    assert agent.tools == []


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
    assert agent.llm.model == "gemini/gemini-1.5-flash"
    assert agent.llm.api_key is not None
    assert agent.llm.max_tokens == 4000
    assert agent.llm.logprobs == False
    assert [hasattr(agent.llm, k) and v for k, v in llm_config.items() if v is not None]
    assert agent.llm.callbacks == [dummy_func]


def test_build_agent_with_llm_instance():
    def dummy_func() -> str:
        return "dummy"

    llm = LLM(model="gemini-1.5", max_tokens=4000, logprobs=False)
    agent = Agent(
        role="analyst",
        goal="analyze the company's website and retrieve the product overview",
        llm=llm,
        max_tokens=3000,
        callbacks=[dummy_func],
    )
    assert isinstance(agent.llm, LLM)
    assert agent.llm.model == "gemini/gemini-1.5-flash"
    assert agent.llm.api_key is not None
    assert agent.llm.max_tokens == 3000
    assert agent.llm.logprobs == False
    assert agent.llm.callbacks == [dummy_func]


def test_build_agent_with_llm_and_func_llm_config():
    def dummy_func() -> str:
        return "dummy"

    llm_params = dict(deployment_name="gemini-1.5", max_tokens=4000, logprobs=False, abc="dummy key")
    agent = Agent(
        role="analyst",
        goal="analyze the company's website and retrieve the product overview",
        function_calling_llm=llm_params,
        callbacks=[dummy_func]
    )

    assert isinstance(agent.llm, LLM) and isinstance(agent.function_calling_llm, LLM)
    assert agent.llm.model == DEFAULT_MODEL_NAME
    assert agent.function_calling_llm.model == "gemini/gemini-1.5-flash" if agent.function_calling_llm._supports_function_calling() else DEFAULT_MODEL_NAME
    assert agent.function_calling_llm.api_key is not None
    assert agent.function_calling_llm.max_tokens == 4000
    assert agent.function_calling_llm.logprobs == False
    assert agent.function_calling_llm.callbacks == [dummy_func]


def test_build_agent_with_llm_and_func_llm_instance():
    def dummy_func() -> str:
        return "dummy"

    llm = LLM(model="gemini-1.5", max_tokens=4000, logprobs=False)
    agent = Agent(
        role="analyst",
        goal="analyze the company's website and retrieve the product overview",
        llm=llm,
        function_calling_llm=llm,
        llm_config=dict(),
        max_tokens=3000,
        callbacks=[dummy_func]
    )
    assert isinstance(agent.llm, LLM) and isinstance(agent.function_calling_llm, LLM)
    assert agent.function_calling_llm.model == "gemini/gemini-1.5-flash" if agent.function_calling_llm._supports_function_calling() else DEFAULT_MODEL_NAME
    assert agent.function_calling_llm.api_key is not None
    assert agent.function_calling_llm.max_tokens == 3000
    assert agent.function_calling_llm.logprobs == False
    assert agent.function_calling_llm.callbacks == [dummy_func]


def test_agent_with_random_dict_tools():
    def empty_func():
        return "empty function"

    agent = Agent(role="demo", goal="test a tool", tools=[dict(name="tool 1", func=empty_func), ])

    assert [tool._run() == "empty function" for tool in agent.tools]
    assert agent.tools[0].name == "tool 1"


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
    from versionhq.task.model import Task

    content = "Kuriko's favorite color is gold, and she enjoy Japanese food."
    string_source = StringKnowledgeSource(content=content)

    agent = Agent(role="Information Agent", goal="Provide information based on knowledge sources", knowledge_sources=[string_source])
    assert agent._knowledge.sources == [string_source] and agent._knowledge.embedder_config == agent.embedder_config

    task = Task(description="Answer the following question: What is Kuriko's favorite color?")

    with patch("versionhq.knowledge.storage.KnowledgeStorage") as MockKnowledge:
        mock_knowledge_instance = MockKnowledge.return_value
        mock_knowledge_instance.sources = [string_source, ]
        mock_knowledge_instance.query.return_value = [{ "content": content }]

        res = task.execute_sync(agent=agent)
        assert "gold" in res.raw.lower()



def test_using_contextual_memory():
    from unittest.mock import patch
    from versionhq.task.model import Task
    from versionhq.task.evaluate import Evaluation
    from versionhq.memory.contextual_memory import ContextualMemory
    from versionhq.storage.rag_storage import RAGStorage
    from versionhq.storage.ltm_sqlite_storage import LTMSQLiteStorage

    agent = Agent(role="Researcher", goal="You research about math.", use_memory=True)
    assert agent.short_term_memory.storage and isinstance(agent.short_term_memory.storage, RAGStorage) and agent.short_term_memory.storage.type == "stm"
    assert agent.long_term_memory.storage and isinstance(agent.long_term_memory.storage, LTMSQLiteStorage)

    task = Task(description="Research a topic to teach a kid aged 6 about math.")

    with patch.object(ContextualMemory, "build_context_for_task") as contextual_mem:
        res = task.execute_sync(agent=agent)
        assert isinstance(res.evaluation, Evaluation) and res.evaluation.suggestion_summary is not None
        assert res.evaluation.aggregate_score is not None
        contextual_mem.assert_called_once()


def test_disabled_memory_using_contextual_memory():
    from unittest.mock import patch
    from versionhq.task.model import Task
    from versionhq.memory.contextual_memory import ContextualMemory

    agent = Agent(role="Researcher", goal="You research about math.", use_memory=False)
    assert agent.short_term_memory is None
    assert agent.long_term_memory is None
    assert agent.user_memory is None

    task = Task(description="Research a topic to teach a kid aged 6 about math.")

    with patch.object(ContextualMemory, "build_context_for_task") as contextual_mem:
        task.execute_sync(agent=agent)
        contextual_mem.assert_not_called()


def test_agent_with_memory_config():
    from versionhq.storage.ltm_sqlite_storage import LTMSQLiteStorage

    agent_1 = Agent(role="Researcher", goal="You research about math.", use_memory=True, memory_config=dict(provider="mem0"))
    agent_2 = Agent(role="Researcher", goal="You research about math.", use_memory=True, memory_config=dict(provider="mem0", user_id="123"))


    assert agent_1.short_term_memory is not None
    assert agent_1.short_term_memory.memory_provider == "mem0" and agent_1.short_term_memory.storage.memory_type == "stm"
    assert agent_1.long_term_memory and isinstance(agent_1.long_term_memory.storage, LTMSQLiteStorage)
    assert agent_1.user_memory is None

    assert agent_2.short_term_memory is not None
    assert agent_2.short_term_memory.memory_provider == "mem0" and agent_2.short_term_memory.storage.memory_type == "stm"
    assert agent_2.long_term_memory and isinstance(agent_2.long_term_memory.storage, LTMSQLiteStorage)
    assert agent_2.user_memory and agent_2.user_memory.storage and agent_2.user_memory.storage.memory_type == "user"


if __name__ == "__main__":
    # test_agent_with_memory_config()
    # test_disabled_memory_using_contextual_memory()
    test_agent_with_random_dict_tools()

# embedder_config
