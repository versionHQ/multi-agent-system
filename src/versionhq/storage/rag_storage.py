import contextlib
import io
import logging
import os
import shutil
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from chromadb.api import ClientAPI

from versionhq.knowledge.embedding import EmbeddingConfigurator
from versionhq._utils.vars import MAX_FILE_NAME_LENGTH
from versionhq.storage.utils import fetch_db_storage_path


@contextlib.contextmanager
def suppress_logging(
    logger_name="chromadb.segment.impl.vector.local_persistent_hnsw",
    level=logging.ERROR,
):
    logger = logging.getLogger(logger_name)
    original_level = logger.getEffectiveLevel()
    logger.setLevel(level)
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
        contextlib.suppress(UserWarning),
    ):
        yield
    logger.setLevel(original_level)


class BaseRAGStorage(ABC):
    """
    Base class for RAG-based Storage implementations.
    """

    app: Any | None = None

    def __init__(
        self,
        type: str,
        allow_reset: bool = True,
        embedder_config: Optional[Any] = None,
        agents: List[Any] = None,
    ):
        self.type = type
        self.allow_reset = allow_reset
        self.embedder_config = embedder_config
        self.agents = agents

    def _initialize_agents(self) -> str:
        if self.agents:
            return "_".join([agent.key for agent in self.agents])
        return ""

    # @abstractmethod
    # def _sanitize_role(self, role: str) -> str:
    #     """Sanitizes agent roles to ensure valid directory names."""
    #     pass

    @abstractmethod
    def save(self, value: Any, metadata: Dict[str, Any]) -> None:
        """Save a value with metadata to the storage."""
        pass

    @abstractmethod
    def search(self, query: str, limit: int = 3, filter: Optional[dict] = None, score_threshold: float = 0.35) -> List[Any]:
        """Search for entries in the storage."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the storage."""
        pass

    @abstractmethod
    def _generate_embedding(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Any:
        """Generate an embedding for the given text and metadata."""
        pass

    @abstractmethod
    def _initialize_app(self):
        """Initialize the vector db."""
        pass

    def setup_config(self, config: Dict[str, Any]):
        """Setup the config of the storage."""
        pass

    def initialize_client(self):
        """Initialize the client of the storage. This should setup the app and the db collection"""
        pass



class RAGStorage(BaseRAGStorage):
    """
    Extends Storage to handle embeddings for memory entries, improving
    search efficiency.
    """

    app: ClientAPI | None = None

    def __init__(self, type, allow_reset=True, embedder_config=None, agents=list(), path=None):
        super().__init__(type, allow_reset, embedder_config, agents)
        agents = agents
        agents = [agent.key for agent in agents]
        agents = "_".join(agents)

        self.agents = agents
        self.storage_file_name = self._build_storage_file_name(type, agents)
        self.type = type
        self.allow_reset = allow_reset
        self.path = path
        self._initialize_app()


    def _set_embedder_config(self):
        configurator = EmbeddingConfigurator()
        self.embedder_config = configurator.configure_embedder(self.embedder_config)


    def _initialize_app(self) -> None:
        import chromadb
        from chromadb.config import Settings

        self._set_embedder_config()
        try:
            chroma_client = chromadb.PersistentClient(
                path=self.path if self.path else self.storage_file_name,
                settings=Settings(allow_reset=self.allow_reset),
            )
            self.app = chroma_client
            self.collection = self.app.get_collection(name=self.type, embedding_function=self.embedder_config)
        except Exception:
            if self.app:
                self.collection = self.app.create_collection(name=self.type, embedding_function=self.embedder_config)


    # def _sanitize_role(self, role: str) -> str:
    #     """
    #     Sanitizes agent roles to ensure valid directory names.
    #     """
    #     return role.replace("\n", "").replace(" ", "_").replace("/", "_")


    def _build_storage_file_name(self, type: str, file_name: str) -> str:
        """
        Ensures file name does not exceed max allowed by OS
        """
        base_path = f"{fetch_db_storage_path()}/{type}"

        if len(file_name) > MAX_FILE_NAME_LENGTH:
            logging.warning(
                f"Trimming file name from {len(file_name)} to {MAX_FILE_NAME_LENGTH} characters."
            )
            file_name = file_name[:MAX_FILE_NAME_LENGTH]

        return f"{base_path}/{file_name}"


    def _create_default_embedding_function(self):
        from chromadb.utils.embedding_functions.openai_embedding_function import (
            OpenAIEmbeddingFunction,
        )

        return OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small"
        )


    def _generate_embedding(self, text: str, metadata: Optional[Dict[str, Any]]) -> None:
        if not hasattr(self, "app") or not hasattr(self, "collection"):
            self._initialize_app()

        if metadata:
            self.collection.add(documents=[text], metadatas=[metadata, ], ids=[str(uuid.uuid4())])

        else:
            self.collection.add(documents=[text], ids=[str(uuid.uuid4())])


    def save(self, data: Dict[str, Any] | str, metadata: Optional[Dict[str, Any]] = dict()) -> None:
        if not hasattr(self, "app") or not hasattr(self, "collection"):
            self._initialize_app()
        try:
            self._generate_embedding(text=str(data), metadata=metadata)
        except Exception as e:
            logging.error(f"Error during {self.type} save: {str(e)}")


    def search(self, query: str, limit: int = 3, filter: Optional[dict] = None, score_threshold: float = 0.35) -> List[Any]:
        if not hasattr(self, "app"):
            self._initialize_app()

        try:
            with suppress_logging():
                response = self.collection.query(query_texts=query, n_results=limit)

            results = []
            for i in range(len(response["ids"][0])):
                result = {
                    "id": response["ids"][0][i],
                    "metadata": response["metadatas"][0][i],
                    "context": response["documents"][0][i],
                    "score": response["distances"][0][i],
                }
                if result["score"] >= score_threshold:
                    results.append(result)

            return results
        except Exception as e:
            logging.error(f"Error during {self.type} search: {str(e)}")
            return []


    def reset(self) -> None:
        try:
            if self.app:
                self.app.reset()
                shutil.rmtree(f"{fetch_db_storage_path()}/{self.type}")
                self.app = None
                self.collection = None
        except Exception as e:
            if "attempt to write a readonly database" in str(e):
                pass
            else:
                raise Exception(f"An error occurred while resetting the {self.type} memory: {e}")
