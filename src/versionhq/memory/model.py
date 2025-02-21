import datetime
from typing import Any, Dict, List, Optional

from versionhq.storage.rag_storage import RAGStorage
from versionhq.storage.ltm_sqlite_storage import LTMSQLiteStorage
from versionhq._utils.logger import Logger


class MemoryData:
    """
    A class to store structured data to store in the memory.
    """
    def __init__(
        self,
        agent: Optional[str] = None, # task execution agent (core)
        task_description: Optional[str] = None,
        task_output: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.agent = agent
        self.task_description = task_description
        self.task_output = task_output

        if config:
            for k, v in config.items():
                setattr(self, k, str(v))



class MemoryMetadata:
    """
     A class to store structured metadata to store in the memory.
    """

    def __init__(
        self,
        task_description: str = None,
        eval_criteria: Optional[str] = None,
        score: Optional[int | float] = None,
        suggestion: Optional[str] = None,
        eval_by: Optional[str] = None, # task evaluator agent
        config: Optional[Dict[str, Any]] = None
    ):
        self.task_description = task_description
        self.eval_criteria = eval_criteria
        self.score = score
        self.suggestion = suggestion
        self.eval_by = eval_by

        if config:
            for k, v in config.items():
                setattr(self, k, str(v))


class Memory:
    """
    Base class for memory.
    """

    def __init__(self, storage: RAGStorage):
        self.storage = storage


    def save(
            self,
            data: MemoryData | Dict[str, Any],
            metadata: Optional[MemoryMetadata | Dict[str, Any]] = None,
            agent: Optional[str] = None
        ) -> None:

        """
        Create a dict for data and metadata without empty values before storing them in the given storage.
        """

        if not data:
            Logger(verbose=True).log(level="error", message="Missing data to store. Add either dict or MemoryData object", color="red")
            return None

        metadata_dict = metadata if isinstance(metadata, dict) else metadata.__dict__ if isinstance(metadata, MemoryMetadata) else dict()
        metadata_dict = {k: v for k, v in metadata_dict.items() if v} # remove empty values
        data_dict = data if isinstance(data, dict) else data.__dict__ if isinstance(data, MemoryData) else dict()
        data_dict = {k: v for k, v in data_dict.items() if v}

        if agent and data_dict["agent"] is None:
            data_dict["agent"] = agent

        if metadata_dict:
            self.storage.save(data=data_dict, metadata=metadata_dict)
        else:
            self.storage.save(data=data_dict)


    def search(self, query: str, limit: int = 3, score_threshold: float = 0.35) -> List[Any]:
        return self.storage.search(query=query, limit=limit, score_threshold=score_threshold)


class MemoryItem:
    """
    A class to store item to be saved in either long term memory or short term memory.
    """

    def __init__(self, data: MemoryData = None, metadata: Optional[MemoryMetadata] = None):
        self.data = data
        self.metadata = metadata if metadata is not None else {}


class ShortTermMemory(Memory):
    """
    A Pydantic class to store agents' short-term memories.
    - Type: stm
    - Storage: Mem0Storage | RAGStorage
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
                try:
                    import os
                    os.system("uv add mem0ai --optional mem0ai")

                    from versionhq.storage.mem0_storage import Mem0Storage
                except:
                    raise ImportError("Mem0 is not installed. Please install it with `$ uv add versionhq[mem0ai]`.")

            storage = Mem0Storage(type="stm", agent=agent)
        else:
            storage = storage if storage else RAGStorage(type="stm", embedder_config=embedder_config, agents=[agent,], path=path)

        super().__init__(storage)


    def save(
            self,
            task_description: str = None,
            task_output: str = None,
            agent: Optional[str] = None,
            data: Optional[MemoryData] = None,
            metadata: Optional[MemoryMetadata] = None
        ) -> None:

        data = data if data else MemoryData(task_description=task_description, task_output=task_output, agent=agent)
        item = MemoryItem(data=data, metadata=metadata)

        if self.memory_provider == "mem0":
            item.data.task_output = f"Remember the following insights from Agent run: {item.data.task_output}"

        super().save(data=item.data.__dict__, metadata=item.metadata.__dict__ if item.metadata else {})


    def search(self, query: str, limit: int = 3, score_threshold: float = 0.35,):
        return self.storage.search(query=query, limit=limit, score_threshold=score_threshold)


    def reset(self) -> None:
        try:
            self.storage.reset()
        except Exception as e:
            raise Exception(f"An error occurred while resetting the short-term memory: {str(e)}")


class LongTermMemory(Memory):
    """
    A Pydantic class for storing agents' long-term memories. Query task outputs for the evaluation.
    - Type: ltm
    - Storage: LTMSQLiteStorage | RAGStorage
    """

    def __init__(self, storage=None, path=None):
        if not storage:
            storage = LTMSQLiteStorage(db_path=path) if path else LTMSQLiteStorage()

        super().__init__(storage)


    def save(
            self,
            task_description: str = None,
            task_output: str = None,
            agent: Optional[str] = None,
            data: Optional[MemoryData] = None,
            metadata: Optional[MemoryMetadata] = None
        ) -> None:

        data = data if data else MemoryData(task_description=task_description, task_output=task_output, agent=agent)
        item = MemoryItem(data=data, metadata=metadata)
        super().save(data=item.data, metadata=item.metadata)


    def search(self, query: str, latest_n: int = 3) -> List[Dict[str, Any]]:
        """
        Query the storage and return the results up to latest_n.
        """
        return self.storage.load(query=query, latest_n=latest_n)


    def reset(self) -> None:
        self.storage.reset()



class UserMemoryItem:
    def __init__(self, data: Any, user: str, metadata: Optional[Dict[str, Any]] = None):
        self.data = data
        self.user = user
        self.metadata = metadata if metadata is not None else {} # can be stored last purchased item, comm related to the user


class UserMemory(Memory):
    """
    UserMemory class for handling user memory storage and retrieval.
    - Type: user
    - Storage: Mem0Storage
    - Requirements: `user_id` in metadata
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
