from typing import Optional, Any
from pydantic import InstanceOf

from versionhq.tool.model import ToolSet
from versionhq.tool.cache_handler import CacheHandler, CacheTool


class ToolHandler:
    """
    Record the tool usage by ToolSet instance with cache and error recording.
    """

    last_used_tool: InstanceOf[ToolSet]
    cache: InstanceOf[CacheHandler] = CacheHandler()
    error: Optional[str]
    should_cache: bool

    def __init__(self, last_used_tool: InstanceOf[ToolSet] = None, cache_handler: InstanceOf[CacheHandler] = None, should_cache: bool = True):
        self.last_used_tool = last_used_tool
        self.cache = cache_handler if cache_handler else CacheHandler()
        self.should_cache = should_cache


    def record_last_tool_used(self, last_used_tool: InstanceOf[ToolSet], output: str, should_cache: bool = True) -> None:
        self.last_used_tool = last_used_tool

        if should_cache:
            self.cache = CacheHandler()
            self.cache.add(tool_name=last_used_tool.tool.name, input=str(last_used_tool.kwargs), output=output)


    def has_called_before(self, tool_set: InstanceOf[ToolSet] = None) -> bool:
        if tool_set is None or not self.last_used_tool:
            return False

        if tool_set := self.last_used_tool:
            return bool((tool_set.tool.name == self.last_used_tool.tool.name) and (tool_set.kwargs == self.last_used_tool.kwargs))
