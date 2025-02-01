from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import contextlib
import hashlib
import io
import logging
import os
import shutil

import chromadb
import chromadb.errors
from chromadb.api import ClientAPI
from chromadb.api.types import OneOrMany
from chromadb.config import Settings

from versionhq._utils.logger import Logger
from versionhq._utils.vars import KNOWLEDGE_DIRECTORY
from versionhq.storage.utils import fetch_db_storage_path
from versionhq.knowledge.embedding import EmbeddingConfigurator


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



class BaseKnowledgeStorage(ABC):
    """
    Abstract base class for knowledge storage implementations.
    """

    @abstractmethod
    def search(self, query: List[str], limit: int = 3, filter: Optional[dict] = None, score_threshold: float = 0.35) -> List[Dict[str, Any]]:
        """Search for documents in the knowledge base."""
        pass

    @abstractmethod
    def save(self, documents: List[str], metadata: Dict[str, Any] | List[Dict[str, Any]]) -> None:
        """Save documents to the knowledge base."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the knowledge base."""
        pass



class KnowledgeStorage(BaseKnowledgeStorage):
    """
    A class to store ChromaDB Storage vals that handles embeddings, ChromaClient, and Collection.
    """

    collection: Optional[chromadb.Collection] = None
    collection_name: Optional[str] = "knowledge"
    app: Optional[ClientAPI] = None
    embedding_function: Optional[Any] = None # store ChromaDB's EmbeddingFunction instance
    embedder_config: Optional[Dict[str, Any]] = None # store config dict for embedding_function


    def __init__(self, embedder_config: Optional[Dict[str, Any]] = None, collection_name: Optional[str] = None):
        self.collection_name = collection_name
        self.embedder_config = embedder_config
        self.initialize_knowledge_storage()


    def _create_default_embedding_function(self) -> Any:
        from chromadb.utils.embedding_functions.openai_embedding_function import OpenAIEmbeddingFunction

        return OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small"
        )


    def _set_embedding_function(self, embedder_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Set the embedding configuration for the knowledge storage.
        """
        self.embedding_function = EmbeddingConfigurator().configure_embedder(embedder_config) if embedder_config else self._create_default_embedding_function()


    def initialize_knowledge_storage(self):
        """
        Create ChromaClinent, set up the embedding function using `embedder_config`, and get or create Collection.
        """
        base_path = os.path.join(fetch_db_storage_path(), "knowledge")
        chroma_client = chromadb.PersistentClient(path=base_path, settings=Settings(allow_reset=True))
        self.app = chroma_client

        self._set_embedding_function(self.embedder_config)

        try:
            collection_name = f"knowledge_{self.collection_name}" if self.collection_name else "knowledge"
            if self.app:
                self.collection = self.app.get_or_create_collection(name=collection_name, embedding_function=self.embedding_function)
            else:
                raise Exception("Vector Database Client not initialized")
        except Exception:
            raise Exception("Failed to create or get collection")


    def search(self, query: List[str], limit: int = 3, filter: Optional[dict] = None, score_threshold: float = 0.35) -> List[Dict[str, Any]]:
        with suppress_logging():
            if self.collection:
                fetched = self.collection.query(query_texts=query, n_results=limit, where=filter)
                results = []
                for i in range(len(fetched["ids"][0])):
                    result = {
                        "id": fetched["ids"][0][i],
                        "metadata": fetched["metadatas"][0][i],
                        "context": fetched["documents"][0][i],
                        "score": fetched["distances"][0][i],
                    }
                    if result["score"] >= score_threshold:
                        results.append(result)
                return results
            else:
                raise Exception("Collection not initialized")


    def save(self, documents: List[str], metadata: Optional[Dict[str, Any] | List[Dict[str, Any]]] = None) -> None:
        if not self.collection:
            self.initialize_knowledge_storage()
            # raise Exception("Collection not initialized")

        try:
            unique_docs = {}
            for i, doc in enumerate(documents):
                if doc:
                    doc = doc
                    if isinstance(doc, list):
                        doc = doc[0]

                    doc_id = hashlib.sha256(str(doc).encode("utf-8")).hexdigest()
                    doc_metadata = None
                    if metadata:
                        if isinstance(metadata, list):
                            doc_metadata = metadata[i]
                        else:
                            doc_metadata = metadata
                    unique_docs[doc_id] = (doc, doc_metadata)

            filtered_docs = []
            filtered_metadata = []
            filtered_ids = []

            for doc_id, (doc, meta) in unique_docs.items():
                if doc_id and doc:
                    filtered_docs.append(doc)
                    filtered_metadata.append(meta)
                    filtered_ids.append(doc_id)

            final_metadata: Optional[OneOrMany[chromadb.Metadata]] = (
                None if all(m is None for m in filtered_metadata) else filtered_metadata
            )

            if filtered_docs:
                self.collection.upsert(documents=filtered_docs, metadatas=final_metadata, ids=filtered_ids)

        except chromadb.errors.InvalidDimensionException as e:
            Logger(verbose=True).log(
                level="error",
                message="Embedding dimension mismatch. This usually happens when mixing different embedding models.",
                color="red",
            )
            raise ValueError("Embedding dimension mismatch. Make sure you're using the same embedding model across all operations with this collection.") from e

        except Exception as e:
            Logger(verbose=True).log(level="error", message=f"Failed to upsert documents: {str(e)}", color="red")
            raise


    def reset(self):
        base_path = os.path.join(fetch_db_storage_path(), KNOWLEDGE_DIRECTORY)
        if not self.app:
            self.app = chromadb.PersistentClient(path=base_path, settings=Settings(allow_reset=True))
        self.app.reset()
        shutil.rmtree(base_path)
        self.app = None
        self.collection = None
