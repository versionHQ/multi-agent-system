import pytest
from typing import Any, Callable

from versionhq.agent.model import Agent
from versionhq.llm.model import LLM
from versionhq.task.model import Task
from versionhq.tool.model import Tool

from tests import Demo, demo_response_fields

"""
Test connection to the llm via endpoint provider or litellm.
"""

def set_agent(llm: str, **kwargs) -> Agent:
    agent = Agent(role="LLM Connection Tester", llm=llm, maxit=1, max_retry_limit=1, llm_config=dict(max_tokens=3000), **kwargs)
    return agent

@pytest.fixture(scope='module')
def simple_task():
    return Task(description="write a random poem.")


@pytest.fixture(scope='module')
def tool_task():
    class DemoTool(Tool):
        func: Callable[..., Any] = lambda x: "Demo"
    return Task(description="Simply execute the given tools.", tools=[DemoTool,], tool_res_as_final=True)


@pytest.fixture(scope='module')
def schema_task():
    return Task(description="Generate a random value that adheres strictly to the specified data type.", response_schema=Demo)


@pytest.fixture(scope='module')
def res_field_task():
    return Task(description="return random values strictly following the given response format.", response_schema=demo_response_fields)


def _test_con_bedrock(simple_task, tool_task, schema_task, res_field_task):
    llms_to_test = [
        "bedrock/converse/us.meta.llama3-3-70b-instruct-v1:0",
        "bedrock/us.meta.llama3-2-11b-instruct-v1:0",
        "bedrock/mistral.mistral-7b-instruct-v0:2",
        "bedrock/amazon.titan-text-lite-v1",
    ]
    agents = [set_agent(llm=llm) for llm in llms_to_test]

    for agent in agents:
        assert isinstance(agent.llm, LLM)
        assert agent.llm.provider == "bedrock"
        assert agent.llm._init_model_name and agent.llm.provider and agent.llm.llm_config["max_tokens"] == agent.llm_config["max_tokens"]

        res_1 = simple_task.execute(agent=agent, context="running a test")
        assert res_1.raw is not None

        res_2 = tool_task.execute(agent=agent, context="running a test")
        assert res_2.tool_output is not None

        res_3 = schema_task.execute(agent=agent, context="running a test")
        assert [
            getattr(res_3.pydantic, k) and v.annotation == Demo.model_fields[k].annotation
            for k, v in res_3.pydantic.model_fields.items()
        ]

        res_4 = res_field_task.execute(agent=agent, context="running a test")
        assert [v and type(v) == res_field_task.response_schema[i].data_type for i, (k, v) in enumerate(res_4.json_dict.items())]


def _test_con_openai(simple_task, tool_task, schema_task, res_field_task):
    llms_to_test = [
        "gpt-4.1",
        "o3-mini",
        "o3-mini-2025-01-31",
    ]
    agents = [set_agent(llm=llm, provider="openai") for llm in llms_to_test]

    for agent in agents:
        assert isinstance(agent.llm, LLM)
        assert agent.llm.provider == "openai"
        assert agent.llm._init_model_name and agent.llm.provider and agent.llm.llm_config["max_tokens"] == agent.llm_config["max_tokens"]

        res_1 = simple_task.execute(agent=agent, context="running a test")
        assert res_1.raw is not None

        res_2 = tool_task.execute(agent=agent, context="running a test")
        assert res_2.tool_output is not None

        res_3 = schema_task.execute(agent=agent, context="running a test")
        assert [
            getattr(res_3.pydantic, k) and v.annotation == Demo.model_fields[k].annotation
            for k, v in res_3.pydantic.model_fields.items()
        ]

        res_4 = res_field_task.execute(agent=agent, context="running a test")
        assert [v and type(v) == res_field_task.response_schema[i].data_type for i, (k, v) in enumerate(res_4.json_dict.items())]


def _test_con_gemini(simple_task, tool_task, schema_task, res_field_task):
    llms_to_test = [
        "gemini/gemini-2.5-pro-exp-03-25"
        "gemini/gemini-2.0-flash",
        "gemini/gemini-2.0-flash-thinking-exp",
        "gemini/gemini-2.0-flash-lite-preview-02-05",
        "gemini/gemini-2.0-flash-exp",
    ]

    agents = [set_agent(llm=llm) for llm in llms_to_test]

    for agent in agents:
        assert isinstance(agent.llm, LLM)
        assert agent.llm.provider == "gemini"
        assert agent.llm._init_model_name and agent.llm.provider and agent.llm.llm_config["max_tokens"] == agent.llm_config["max_tokens"]

        res_1 = simple_task.execute(agent=agent, context="running a test")
        assert res_1.raw is not None

        res_2 = tool_task.execute(agent=agent, context="running a test")
        assert res_2.tool_output is not None

        schema_task = Task(description="Generate a random value that adheres strictly to the specified data type.", response_schema=Demo)
        res_3 = schema_task.execute(agent=agent, context="running a test")
        assert [
            getattr(res_3.pydantic, k) and v.annotation == Demo.model_fields[k].annotation
            for k, v in res_3.pydantic.model_fields.items()
        ]

        res_4 = res_field_task.execute(agent=agent, context="running a test")
        assert [v and type(v) == res_field_task.response_schema[i].data_type for i, (k, v) in enumerate(res_4.json_dict.items())]


def _test_con_azure(simple_task, tool_task, schema_task, res_field_task):
    llms_to_test = [
        "azure/Phi-4-mini-instruct",
        "azure_ai/Phi-4-mini-instruct",
        "azure_ai/DeepSeek-V3",
    ]
    agents = [set_agent(llm=llm) for llm in llms_to_test]

    for agent in agents:
        assert isinstance(agent.llm, LLM)
        assert "azure" in agent.llm.provider
        assert agent.llm._init_model_name and agent.llm.provider and agent.llm.llm_config["max_tokens"] == agent.llm_config["max_tokens"]

        res_1 = simple_task.execute(agent=agent, context="running a test")
        assert res_1.raw is not None

        res_2 = tool_task.execute(agent=agent, context="running a test")
        assert res_2.tool_output is not None

        res_3 = schema_task.execute(agent=agent, context="running a test")
        if res_3.pydantic:
            assert [
                getattr(res_3.pydantic, k) and v.annotation == Demo.model_fields[k].annotation
                for k, v in res_3.pydantic.model_fields.items()
            ]

        res_4 = res_field_task.execute(agent=agent, context="running a test")
        if res_4.json_dict:
            assert [v and type(v) == res_field_task.response_schema[i].data_type for i, (k, v) in enumerate(res_4.json_dict.items())]
