from typing import Any, Dict, Optional

from pydantic import BaseModel, PrivateAttr


class CacheHandler(BaseModel):
    _cache: Dict[str, Any] = PrivateAttr(default_factory=dict)

    def add(self, tool: str, input: str, output: Any) -> None:
        self._cache[f"{tool}-{input}"] = output

    def read(self, tool: str, input: str) -> Optional[str]:
        return self._cache.get(f"{tool}-{input}")
