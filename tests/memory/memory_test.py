from unittest.mock import patch

import pytest

from versionhq.agent.model import Agent
from versionhq.task.model import Task
from versionhq.memory.model import ShortTermMemory, ShortTermMemoryItem, LongTermMemory, LongTermMemoryItem


@pytest.fixture
def short_term_memory():
    """Fixture to create a ShortTermMemory instance"""
    agent = Agent(
        role="Demo",
        goal="Search relevant data and provide results",
    )

    task = Task(
        description="Perform a search on specific topics.",
        expected_output="A list of relevant URLs based on the search query.",
        agent=agent,
    )
    return ShortTermMemory(agent=agent)


def test_save_and_search_stm(short_term_memory):
    memory = ShortTermMemoryItem(
        data="""test value test value test value test value test value test value
        test value test value test value test value test value test value
        test value test value test value test value test value test value""",
        agent="test_agent",
        metadata={ "task": "test_task" },
    )

    with patch.object(ShortTermMemory, "save") as mock_save:
        short_term_memory.save(
            value=memory.data,
            metadata=memory.metadata,
            agent=memory.agent,
        )

        mock_save.assert_called_once_with(
            value=memory.data,
            metadata=memory.metadata,
            agent=memory.agent,
        )

    expected_result = [
        {
            "context": memory.data,
            "metadata": {"agent": "test_agent"},
            "score": 0.95,
        }
    ]
    with patch.object(ShortTermMemory, "search", return_value=expected_result):
        find = short_term_memory.search("test value", score_threshold=0.01)[0]
        assert find["context"] == memory.data, "Data value mismatch."
        assert find["metadata"]["agent"] == "test_agent", "Agent value mismatch."




@pytest.fixture
def long_term_memory():
    """Fixture to create a LongTermMemory instance"""
    return LongTermMemory()


def test_save_and_search_ltm(long_term_memory):
    memory = LongTermMemoryItem(
        agent="test_agent",
        task="test_task",
        datetime="test_datetime",
        quality=0.5,
        metadata={"task": "test_task", "quality": 0.5},
    )
    long_term_memory.save(memory)
    find = long_term_memory.search("test_task", latest_n = 5)[0]
    assert find["score"] == 0.5
    assert find["datetime"] == "test_datetime"
    assert find["metadata"]["agent"] == "test_agent"
    assert find["metadata"]["quality"] == 0.5
    assert find["metadata"]["task"] == "test_task"
