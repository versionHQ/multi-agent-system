import os
import pytest

from versionhq.agent.model import Agent
from versionhq.llm.model import LLM, DEFAULT_MODEL_NAME
from versionhq.tool.model import Tool

MODEL_NAME = os.environ.get("DEFAULT_MODEL_NAME", "gpt-3.5-turbo")
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY")


def test_build_agent_with_minimal_input():
    agent = Agent(
        role="analyst",
        goal="analyze the company's website and retrieve the product overview",
    )

    assert agent.role == "analyst"
    assert agent.role in agent.backstory
    assert agent.goal in agent.backstory
    assert isinstance(agent.llm, LLM)
    assert agent.llm.model == "gpt-3.5-turbo"
    assert agent.llm.api_key == LITELLM_API_KEY
    assert agent.tools == []


def test_build_agent_with_backstory():
    agent = Agent(
        role="analyst",
        goal="analyze the company's website and retrieve the product overview",
        backstory="You are competitive analysts who have abundand knowledge in marketing, product management."
    )

    assert agent.role == "analyst"
    assert agent.backstory == "You are competitive analysts who have abundand knowledge in marketing, product management."
    assert isinstance(agent.llm, LLM)
    assert agent.llm.model == "gpt-3.5-turbo"
    assert agent.llm.api_key == LITELLM_API_KEY
    assert agent.tools == []


def test_build_agent():
    agent = Agent(
        role="analyst",
        goal="analyze the company's website and retrieve the product overview",
        knowledge="competitor products",
        skillsets=["financial analysis", "product management", ]
    )

    assert agent.role == "analyst"
    assert agent.role in agent.backstory
    assert agent.goal in agent.backstory
    assert agent.knowledge in agent.backstory
    for item in agent.skillsets:
        assert item in agent.backstory
    assert isinstance(agent.llm, LLM)
    assert agent.llm.model == "gpt-3.5-turbo"
    assert agent.llm.api_key == LITELLM_API_KEY
    assert agent.tools == []


def test_build_agent_with_llm():
    agent = Agent(
        role="analyst",
        goal="analyze the company's website and retrieve the product overview",
        knowledge="competitor products",
        skillsets=["financial analysis", "product management", ],
        llm="gpt-4o"
    )
    assert agent.role == "analyst"
    assert agent.role in agent.backstory
    assert agent.goal in agent.backstory
    assert agent.knowledge in agent.backstory
    for item in agent.skillsets:
        assert item in agent.backstory
    assert isinstance(agent.llm, LLM)
    assert agent.llm.model == "gpt-4o"
    assert agent.llm.api_key == LITELLM_API_KEY
    assert agent.tools == []


def test_build_agent_with_llm_config():
    def dummy_func() -> str:
        return "dummy"

    llm_params = dict(deployment_name="gemini-1.5", max_tokens=4000, logprobs=False, abc="dummy key")
    agent = Agent(
        role="analyst",
        goal="analyze the company's website and retrieve the product overview",
        llm=llm_params,
        step_callback=dummy_func,
    )

    assert isinstance(agent.llm, LLM)
    assert agent.llm.model == "gemini/gemini-1.5-flash"
    assert agent.llm.api_key is not None
    assert agent.llm.max_tokens == 4000
    assert agent.llm.logprobs == False
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
        step_callback=dummy_func,
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
        step_callback=dummy_func,
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
        max_tokens=3000,
        step_callback=dummy_func,
    )

    assert isinstance(agent.llm, LLM) and isinstance(agent.function_calling_llm, LLM)
    assert agent.function_calling_llm.model == "gemini/gemini-1.5-flash" if agent.function_calling_llm._supports_function_calling() else DEFAULT_MODEL_NAME
    assert agent.function_calling_llm.api_key is not None
    assert agent.function_calling_llm.max_tokens == 3000
    assert agent.function_calling_llm.logprobs == False
    assert agent.function_calling_llm.callbacks == [dummy_func]



def test_agent_with_random_str_tools():
    agent = Agent(role="demo", goal="test a tool", tools=["random tool 1", "random tool 2",])

    assert [isinstance(tool, Tool) for tool in agent.tools]
    assert [tool.run() is not None for tool in agent.tools]
    assert [tool.run() == "empty function" for tool in agent.tools]
    assert agent.tools[0].name == "random tool 1"
    assert agent.tools[1].name == "random tool 2"


def test_agent_with_random_dict_tools():
    def empty_func():
        return "empty function"
    agent = Agent(role="demo", goal="test a tool", tools=[dict(name="tool 1", function=empty_func), ])

    assert [tool.run() == "empty function" for tool in agent.tools]
    assert agent.tools[0].name == "tool 1"


def test_agent_with_custom_tools():
    class CustomTool(Tool):
        name: str = "custom tool"

        def _run(self, agent_role: str) -> str:
            return agent_role

    tool = CustomTool()
    agent = Agent(role="demo", goal="test a tool", tools=[tool])

    assert agent.tools[0] is tool
    assert agent.tools[0].run(agent_role=agent.role) == "demo"
    assert agent.tools[0].name == "custom tool"
