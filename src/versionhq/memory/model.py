from typing import Any, Dict, List, Optional

from versionhq.storage.rag_storage import RAGStorage


class Memory:
    """
    Base class for memory.
    """

    def __init__(self, storage: RAGStorage):
        self.storage = storage


    def save(self, value: Any, metadata: Optional[Dict[str, Any]] = None, agent: Optional[str] = None) -> None:
        metadata = metadata or {}

        if agent:
            metadata["agent"] = agent
        self.storage.save(value, metadata)


    def search(self, query: str, limit: int = 3, score_threshold: float = 0.35) -> List[Any]:
        return self.storage.search(query=query, limit=limit, score_threshold=score_threshold)



class ShortTermMemoryItem:
    def __init__(
        self,
        data: Any,
        agent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.data = data
        self.agent = agent
        self.metadata = metadata if metadata is not None else {}


class ShortTermMemory(Memory):
    """
    A class for managing transient data related to immediate tasks and interactions.
    """

    def __init__(self, agent = None, embedder_config: Dict[str, Any] = None, storage=None, path=None):
        if hasattr(agent, "memory_config") and agent.memory_config is not None:
            self.memory_provider = agent.memory_config.get("provider")
        else:
            self.memory_provider = None

        if self.memory_provider == "mem0":
            try:
                from versionhq.storage.mem0_storage import Mem0Storage
            except ImportError:
                raise ImportError("Mem0 is not installed. Please install it with `uv pip install mem0ai`.")

            storage = Mem0Storage(type="stm", agent=agent)
        else:
            storage = storage if storage else RAGStorage(type="stm", embedder_config=embedder_config, agents=[agent,], path=path)

        super().__init__(storage)


    def save(self, value: Any, metadata: Optional[Dict[str, Any]] = None, agent: Optional[str] = None) -> None:
        item = ShortTermMemoryItem(data=value, metadata=metadata, agent=agent)
        if self.memory_provider == "mem0":
            item.data = f"Remember the following insights from Agent run: {item.data}"

        super().save(value=item.data, metadata=item.metadata, agent=item.agent)


    def search(self, query: str, limit: int = 3, score_threshold: float = 0.35,):
        return self.storage.search(query=query, limit=limit, score_threshold=score_threshold)

    def reset(self) -> None:
        try:
            self.storage.reset()
        except Exception as e:
            raise Exception(f"An error occurred while resetting the short-term memory: {str(e)}")



class UserMemoryItem:
    def __init__(self, data: Any, user: str, metadata: Optional[Dict[str, Any]] = None):
        self.data = data
        self.user = user
        self.metadata = metadata if metadata is not None else {}


class UserMemory(Memory):
    """
    UserMemory class for handling user memory storage and retrieval.
    """

    def __init__(self, agent=None, user_id=None):
        try:
            from versionhq.storage.mem0_storage import Mem0Storage
        except ImportError:
            raise ImportError("Mem0 is not installed. Please install it with `uv pip install mem0ai`.")

        if not user_id:
            raise ValueError("Need User Id to create UserMemory.")

        else:
            storage = Mem0Storage(type="user", agent=agent, user_id=user_id)
            super().__init__(storage)


    def save(self, value: str, metadata: Optional[Dict[str, Any]] = None, agent: Optional[str] = None) -> None:
        data = f"Remember the details about the user: {value}"
        super().save(value=data, metadata=metadata, agent=agent)


    def search(self, query: str, limit: int = 3, score_threshold: float = 0.35):
        results = self.storage.search(query=query, limit=limit, score_threshold=score_threshold)
        return results
