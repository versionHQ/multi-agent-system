from typing import Any, Optional
from pydantic import InstanceOf
from versionhq.tool.model import ToolCalled, InstructorToolCalled, CacheTool
from versionhq._utils.cache_handler import CacheHandler


class ToolHandler:
    """
    Record the tool usage by ToolCalled instance with cache and error recording.
    Use as a callback function.
    """

    last_used_tool: ToolCalled = {}
    cache: Optional[CacheHandler]
    error: Optional[str]

    def __init__(self, cache_handler: Optional[CacheHandler] = None):
        """
        Initialize the callback handler.
        """

        self.cache = cache_handler
        self.last_used_tool = {}

    def record_last_tool_used(
        self,
        last_used_tool: InstanceOf[ToolCalled] | InstanceOf[InstructorToolCalled],
        output: str,
        should_cache: bool = True,
    ) -> Any:
        self.last_used_tool = last_used_tool

        if self.cache and should_cache and last_used_tool.tool_name != CacheTool().name:
            self.cache.add(
                last_used_tool.tool_name,
                input.last_used_tool.arguments,
                output=output,
            )

    def has_called_before(self, tool_called: ToolCalled = None) -> bool:
        if tool_called is None or not self.last_used_tool:
            return False

        if tool_called := self.last_used_tool:
            return bool(
                (tool_called.tool.name == self.last_used_tool.tool.name)
                and (tool_called.arguments == self.last_used_tool.arguments)
            )
