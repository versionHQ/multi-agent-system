import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from versionhq.knowledge.storage import KnowledgeStorage
from versionhq.knowledge.source import BaseKnowledgeSource


class Knowledge(BaseModel):
    """
    Knowlede class for collection of sources and setup for the vector store to query relevant context.
    """
    sources: List[BaseKnowledgeSource] = Field(default_factory=list)
    model_config = ConfigDict(arbitrary_types_allowed=True)
    storage: KnowledgeStorage = Field(default_factory=KnowledgeStorage)
    embedder_config: Optional[Dict[str, Any]] = None
    collection_name: Optional[str] = None

    def __init__(
        self,
        collection_name: str,
        sources: List[BaseKnowledgeSource],
        embedder_config: Optional[Dict[str, Any]] = None,
        storage: Optional[KnowledgeStorage] = None,
        **data,
    ):
        super().__init__(**data)


        if storage:
            self.storage = storage
        else:
            self.storage = KnowledgeStorage(embedder_config=embedder_config, collection_name=collection_name)

        self.storage._set_embedding_function(embedder_config=embedder_config)
        self.storage.initialize_knowledge_storage()

        self.sources = sources
        for source in sources:
            source.storage = self.storage
            source.add()


    def query(self, query: List[str], limit: int = 3) -> List[Dict[str, Any]]:
        """
        Query across all knowledge sources to find the most relevant information.
        Returns the top_k most relevant chunks.
        """

        results = self.storage.search(query, limit)
        return results


    def _add_sources(self):
        for source in self.sources:
            source.storage = self.storage
            source.add()
