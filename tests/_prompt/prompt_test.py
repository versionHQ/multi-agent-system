import pytest

from versionhq.task.model import Task
from tests import demo_response_fields


@pytest.fixture(scope='module')
def sub_task():
    return Task(description="write a random poem.")


@pytest.fixture(scope='module')
def main_task():
    from pathlib import Path

    current_dir = Path(__file__).parent.parent
    file_path = current_dir / "_sample/screenshot.png"
    audio_path = current_dir / "_sample/sample.mp3"

    return Task(
        description="return random values strictly following the given response format.",
        response_fields=demo_response_fields,
        tools=[lambda x: x, lambda y: y],
        can_use_agent_tools=True,
        image=str(file_path),
        file=str(file_path),
        audio=str(audio_path),
        should_evaluate=True,
        eval_criteria=["test1", "test2"],
    )


@pytest.fixture(scope='module')
def rag_tool():
    import versionhq as vhq
    rag_tool = vhq.RagTool(url="https://github.com/chroma-core/chroma/issues/3233", query="What is the next action plan?")
    return rag_tool


def test_draft_prompt(main_task, sub_task, rag_tool):
    import versionhq as vhq
    from versionhq._prompt.model import Prompt

    agent = vhq.Agent(llm="gemini-2.0", role="Content Interpretator")
    user_prompt, dev_prompt, messages = Prompt(task=main_task, agent=agent, context=["test", "test2", sub_task]).format_core(rag_tools=[rag_tool])

    assert user_prompt is not None
    assert dev_prompt is not None
    assert messages[0]["role"] == "user"
    assert isinstance(messages[0]["content"], list)
    assert messages[1]["role"] == "developer"
    assert messages[1]["content"] == agent.backstory


def test_draft_prompt_with_tools(main_task, sub_task, rag_tool):
    import versionhq as vhq
    from versionhq._prompt.model import Prompt

    agent = vhq.Agent(llm="gemini-2.0", role="Content Interpretator", tools=[rag_tool])
    user_prompt, dev_prompt, messages = Prompt(task=main_task, agent=agent, context=["test", "test2", sub_task]).format_core()

    assert user_prompt is not None
    assert dev_prompt is not None
    assert messages[0]["role"] == "user"
    assert isinstance(messages[0]["content"], list)
    assert messages[1]["role"] == "developer"
    assert messages[1]["content"] == agent.backstory
