from typing import Any, Dict, Optional

from pydantic import BaseModel, PrivateAttr, Field, InstanceOf


class CacheHandler(BaseModel):
    """
    A class to add or read cache
    """

    _cache: Dict[str, Any] = PrivateAttr(default_factory=dict)

    def add(self, tool_name: str, input: str, output: Any) -> None:
        self._cache[f"{tool_name}-{input}"] = output

    def read(self, tool_name: str, input: str) -> Optional[str]:
        return self._cache.get(f"{tool_name}-{input}")



class CacheTool(BaseModel):
    """
    A cache tool to read the cached result.
    """

    name: str = "Cache Tool"
    cache_handler: InstanceOf[CacheHandler] = Field(default_factory=CacheHandler)

    def read_cache(self, key):
        split = key.split("tool:")
        tool = split[1].split("|input:")[0].strip()
        tool_input = split[1].split("|input:")[1].strip()
        return self.cache_handler.read(tool, tool_input)

    def tool(self):
        return Tool(
            func=self.read_cache,
            name=self.name,
            description="Read from cache"
        )
