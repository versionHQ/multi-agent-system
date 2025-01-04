from typing import Optional, Any
from pydantic import InstanceOf

from versionhq.tool.model import ToolSet, InstructorToolSet
from versionhq._utils.cache_handler import CacheHandler


class ToolHandler:
    """
    Record the tool usage by ToolSet instance with cache and error recording.
    """

    last_used_tool: InstanceOf[ToolSet] | InstanceOf[InstructorToolSet]
    cache: Optional[CacheHandler]
    error: Optional[str]

    def __init__(
        self,
        last_used_tool: InstanceOf[ToolSet] | InstanceOf[InstructorToolSet] = None,
        cache_handler: Optional[CacheHandler] = None
    ):
        self.cache = cache_handler
        self.last_used_tool = last_used_tool


    def record_last_tool_used(
        self,
        last_used_tool: InstanceOf[ToolSet] | InstanceOf[InstructorToolSet],
        output: str,
        should_cache: bool = True,
    ) -> Any:

        self.last_used_tool = last_used_tool

        from versionhq.tool.model import CacheTool
        if self.cache and should_cache and last_used_tool.tool.name != CacheTool().name:
            self.cache.add(tool=last_used_tool.tool.name, input=last_used_tool.kwargs, output=output)


    def has_called_before(self, tool_called: ToolSet = None) -> bool:
        if tool_called is None or not self.last_used_tool:
            return False

        if tool_called := self.last_used_tool:
            return bool((tool_called.tool.name == self.last_used_tool.tool.name) and (tool_called.kwargs == self.last_used_tool.kwargs))
