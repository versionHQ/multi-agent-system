import os
import pytest
from unittest.mock import patch
from typing import Dict, Any, Callable, List, Optional

from versionhq.tool.model import Tool, ToolSet, BaseTool
from versionhq.tool.tool_handler import ToolHandler
from versionhq.tool.cache_handler import CacheHandler
from versionhq.tool.decorator import tool


def test_create_custom_tool():
    class CustomTool(Tool):
        name: str = "custom tool"
        goal: str = "test a custom tool class"
        func: Callable[..., Any] = lambda x: f"{x}-demo"

    tool = CustomTool()

    assert tool.name == "custom tool"
    assert tool.description == "Tool: custom tool\nArgs: {'x': {'description': '', 'type': 'Any'}}"
    assert tool._run(x="I am a sentence") == "I am a sentence-demo"


def test_run_tool_with_abstract_function():
    def function_to_test(test_list: List) -> int:
        return len(test_list)

    class CustomTool(Tool):
        name: str = "custom tool"
        func: Callable[..., Any] = function_to_test

    tool = CustomTool()
    result = tool._run(["demo 0", "demo 1", "demo 2"])

    assert tool.name == "custom tool"
    assert result == 3


def test_run_tool_with_function():
    def function_to_test(test_list: List) -> int:
        return len(test_list)

    tool = Tool(name="tool", func=function_to_test)
    result = tool._run(test_list=["demo 0", "demo 1", "demo 2"])

    assert tool.name == "tool"
    assert result == 3


def test_setup_cache_function():
    class CustomTool(Tool):
        name: str = "custom tool"
        goal: str = "test a custom tool"
        cache_function: Callable = lambda: False
        func: Callable = lambda sentence: sentence

    tool = CustomTool()
    assert not tool.cache_function()


def test_default_cache_function():
    class CustomTool(Tool):
        name: str = "custom tool"
        func: Callable[..., Any] = lambda x: x

    tool = CustomTool()
    assert tool.cache_function()


def test_tool_annotation():
    @tool("demo")
    def my_tool(test_words: str) -> str:
        """Test a tool decorator."""
        return test_words

    assert my_tool.name == "demo"
    assert my_tool.description == "Tool: demo\nArgs: {'test_words': {'description': '', 'type': 'str'}}"
    assert my_tool.args_schema.model_json_schema() == {'properties': {'test_words': {'title': 'Test Words', 'type': 'string'}}, 'required': ['test_words'], 'title': 'My_ToolSchema', 'type': 'object'}
    assert my_tool.func("testing") == "testing"


def test_tool_handler_turning_off_cache():
    def empty_func():
        return "empty function"

    tool = Tool(name="demo", func=empty_func, should_cache=False)
    tool.run()

    assert isinstance(tool.tool_handler, ToolHandler)
    assert tool.tool_handler.should_cache == False
    assert tool.tool_handler.last_used_tool is None


def test_tool_handler_with_cache():
    def empty_func():
        return "empty function"

    tool = Tool(name="demo", func=empty_func, should_cache=True)
    res = tool.run()

    assert res == "empty function"
    assert isinstance(tool.tool_handler, ToolHandler)
    assert tool.tool_handler.should_cache == True
    assert tool.tool_handler.last_used_tool == ToolSet(tool=tool, kwargs={})
    assert tool.tool_handler.cache.read(tool_name=tool.name, input=str({})) == res


def test_custom_tool_with_cache():
    class CustomTool(Tool):
        name: str = "custom tool"
        func: Optional[Callable] = None

    def demo_func(demo_list: List[Any]) -> int:
        return len(demo_list)

    tool = CustomTool(func=demo_func)
    res = tool.run(params=dict(demo_list=["demo1", "demo2"]))

    assert isinstance(tool.tool_handler, ToolHandler)
    assert tool.tool_handler.should_cache == True
    assert tool.tool_handler.last_used_tool == ToolSet(tool=tool, kwargs=dict(demo_list=["demo1", "demo2"]))



def test_cache_tool():
    my_tool = Tool(
        name="my tool",
        func=lambda x: x + 2,
        cache_handler=CacheHandler()
    )
    res = my_tool._run(x=3)
    my_tool.cache_handler.add(tool_name=my_tool.name, input=str({"x": 3}), output=res)

    assert my_tool.cache_handler.read(tool_name=my_tool.name, input=str({"x": 3})) == 5
    assert my_tool.cache_handler._cache[f"{my_tool.name}-{str({"x": 3})}"] == 5
