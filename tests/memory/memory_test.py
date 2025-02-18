from unittest.mock import patch
import pytest

from versionhq.agent.model import Agent
from versionhq.memory.model import ShortTermMemory, LongTermMemory, MemoryItem, MemoryMetadata, MemoryData



@pytest.fixture
def short_term_memory():
    """Fixture to create a ShortTermMemory instance"""
    agent = Agent(role="Pytest Demo", goal="Search relevant data and provide results")
    return ShortTermMemory(agent=agent)


def test_save_and_search_stm(short_term_memory):
    memory = MemoryItem(
        data=MemoryData(
            agent="test_agent",
            config=dict(val="""test value test value test value test value test value test value test value testvalue test value test value test value test value test value test value cat test value test value test value test value""")
        ),
        metadata=MemoryMetadata(config={ "task": "test_task" })
    )

    with patch.object(ShortTermMemory, "save") as mock_save:
        short_term_memory.save(value=memory.data, metadata=memory.metadata)
        mock_save.assert_called_once_with(value=memory.data, metadata=memory.metadata)

    expected_result = [
        {
            "context": str(memory.data),
            "metadata": { "task": "test_task" },
            "score": 0.95,
        }
    ]
    with patch.object(ShortTermMemory, "search", return_value=expected_result):
        find = short_term_memory.search("test value", score_threshold=0.01)[0]
        assert find["context"] == str(memory.data), "Data value mismatch."
        assert find["metadata"]["task"] == "test_task", "Metadata value mismatch."


@pytest.fixture
def long_term_memory():
    """Fixture to create a LongTermMemory instance"""
    return LongTermMemory()


def test_save_and_search_ltm(long_term_memory):
    memory_data = MemoryData(agent="test_agent", task_description="test_task", task_output="test_output")
    memory_metadata = MemoryMetadata(eval_criteria="test", score=0.5, config={"task": "test_task", "quality": 0.5})

    long_term_memory.save(data=memory_data, metadata=memory_metadata)
    find = long_term_memory.search(query="test_task", latest_n=3)[0]

    assert long_term_memory.storage is not None
    assert find["data"]["agent"] == "test_agent"
    assert find["data"]["task_description"] == "test_task"
    assert find["data"]["task_output"] == "test_output"
    assert find["metadata"]["eval_criteria"] == "test"
    assert find["metadata"]["score"] == 0.5
    assert find["metadata"]["task"] == "test_task"
    assert find["metadata"]["quality"] == "0.5"
