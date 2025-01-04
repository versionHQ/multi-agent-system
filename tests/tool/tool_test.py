import os
import pytest
from unittest.mock import patch
from typing import Dict, Any, Callable, List

from versionhq.tool.model import Tool, ToolSet
from versionhq.tool.tool_handler import ToolHandler
from versionhq.tool.decorator import tool


def test_create_custom_tool():
    class CustomTool(Tool):
        name: str = "custom tool"
        goal: str = "test a custom tool class"

        def _run(self, sentence: str) -> str:
            return sentence

    tool = CustomTool()

    assert tool.name == "custom tool"
    assert tool.description == "Tool Name: custom tool\nTool Arguments: {'sentence': {'description': None, 'type': 'str'}}\nGoal: test a custom tool class"
    assert tool.args_schema.model_json_schema()["properties"] == {"sentence": { "title": "Sentence", "type": "string" }}
    assert tool.run(sentence="I am a sentence.") == "I am a sentence."


def test_run_tool_with_abstract_function():
    def function_to_test(test_list: List) -> int:
        return len(test_list)

    class CustomTool(Tool):
        name: str = "custom tool"
        goal: str = "test a custom tool class"

        def _run(self, test_list: List) -> int:
            return function_to_test(test_list)

    tool = CustomTool()
    result = tool.run(test_list=["demo 0", "demo 1", "demo 2"])

    assert tool.name is "custom tool"
    assert result == 3


def test_run_tool_with_function():
    def function_to_test(test_list: List) -> int:
        return len(test_list)

    tool = Tool(name="tool", goal="run a function", function=function_to_test)
    result = tool.run(test_list=["demo 0", "demo 1", "demo 2"])

    assert tool.name is "tool"
    assert result == 3


def test_setup_cache_function():
    class CustomTool(Tool):
        name: str = "custom tool"
        goal: str = "test a custom tool"
        cache_function: Callable = lambda: False

        def _run(self, sentence: str) -> str:
            return sentence

    tool = CustomTool()
    assert not tool.cache_function()


def test_default_cache_function():
    class CustomTool(Tool):
        name: str = "custom tool"
        goal: str = "test a custom tool"

        def _run(self, sentence: str) -> str:
            return sentence

    tool = CustomTool()
    assert tool.cache_function()


def test_tool_annotation():
    @tool("demo")
    def my_tool(test_words: str) -> str:
        """Test a tool decorator."""
        return test_words

    assert my_tool.name == "demo"
    assert my_tool.description == "Tool Name: demo\nTool Arguments: {'test_words': {'description': None, 'type': 'str'}}\nGoal: None"
    assert my_tool.args_schema.model_json_schema()["properties"] == { "test_words": {"title": "Test Words", "type": "string"}}
    assert my_tool.function("testing") == "testing"


def test_tool_handler_turning_off_cache():
    def empty_func():
        return "empty function"

    tool = Tool(name="demo", function=empty_func, should_cache=False)
    tool.run()

    assert isinstance(tool.tool_handler, ToolHandler)
    assert tool.tool_handler.should_cache == False
    assert tool.tool_handler.last_used_tool is None


def test_tool_handler_with_cache():
    def empty_func():
        return "empty function"

    tool = Tool(name="demo", function=empty_func, should_cache=True)
    tool.run()

    assert isinstance(tool.tool_handler, ToolHandler)
    assert tool.tool_handler.should_cache == True
    assert tool.tool_handler.last_used_tool == ToolSet(tool=tool, kwargs={})


if __name__ == "__main__":
    test_tool_handler_with_cache()
