import os
import pytest
from dotenv import load_dotenv

from versionhq.agent.model import Agent
from versionhq.llm.model import LLM


load_dotenv(override=True)
MODEL_NAME = os.environ.get("LITELLM_MODEL_NAME", "gpt-3.5-turbo")
LLM_API_KEY = os.environ.get("LITELLM_API_KEY")


def build_agent_with_minimal_input():
    agent = Agent(
        role="analyst",
        goal="analyze the company's website and retrieve the product overview",
    )

    assert agent.role == "analyst"
    assert agent.role in agent.backstory
    assert agent.goal in agent.backstory
    assert isinstance(agent.llm, LLM)
    assert agent.llm.model == "gpt-3.5-turbo"
    assert agent.llm.api_key == LLM_API_KEY
    assert agent.tools == []


def build_agent_with_backstory():
    agent = Agent(
        role="analyst",
        goal="analyze the company's website and retrieve the product overview",
        backstory="You are competitive analysts who have abundand knowledge in marketing, product management."
    )

    assert agent.role == "analyst"
    assert agent.backstory == "You are competitive analysts who have abundand knowledge in marketing, product management."
    assert isinstance(agent.llm, LLM)
    assert agent.llm.model == "gpt-3.5-turbo"
    assert agent.llm.api_key == LLM_API_KEY
    assert agent.tools == []


def build_agent():
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
    assert agent.llm.api_key == LLM_API_KEY
    assert agent.tools == []


def build_agent_with_llm():
    agent = Agent(
        role="analyst",
        goal="analyze the company's website and retrieve the product overview",
        knowledge="competitor products",
        skillsets=["financial analysis", "product management", ],
        llm="gtp-4o"
    )

    assert agent.role == "analyst"
    assert agent.role in agent.backstory
    assert agent.goal in agent.backstory
    assert agent.knowledge in agent.backstory
    for item in agent.skillsets:
        assert item in agent.backstory
    assert isinstance(agent.llm, LLM)
    assert agent.llm.model == "gpt-4o"
    assert agent.llm.api_key == LLM_API_KEY
    assert agent.tools == []
