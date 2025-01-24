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
    assert agent.backstory == BACKSTORY_SHORT.format(role=agent.role, goal=agent.goal)
    assert isinstance(agent.llm, LLM)
    assert agent.llm.model == DEFAULT_MODEL_NAME
    assert agent.llm.api_key == LITELLM_API_KEY
    assert agent.tools == []


def test_build_agent_from_config():
    agent = Agent(config=dict(role="analyst", goal="analyze the company's website and retrieve the product overview"))

    assert agent.role == "analyst"
    assert agent.backstory == BACKSTORY_SHORT.format(role=agent.role, goal=agent.goal)
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
        knowledge="competitor products",
        skillsets=["financial analysis", "product management", ]
    )

    assert agent.role == "analyst"
    assert agent.backstory == BACKSTORY_FULL.format(
        role=agent.role, goal=agent.goal, knowledge=agent.knowledge, skillsets=", ".join([item for item in agent.skillsets]), rag_tool_overview="")
    assert isinstance(agent.llm, LLM)
    assert agent.llm.model == DEFAULT_MODEL_NAME
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

    with patch.object(
        LLM, "call", wraps=LLM(model=DEFAULT_MODEL_NAME).call
    ) as private_mock:
        task = Task(
            description="The final answer is 42. But don't give it yet, instead keep using the `get_final_answer` tool.",
            can_use_agent_tools=True
        )
        agent.execute_task(task=task)
        assert private_mock.call_count == 1
