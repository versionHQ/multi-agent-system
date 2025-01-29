import csv
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from versionhq.knowledge.storage import KnowledgeStorage
from versionhq._utils.vars import KNOWLEDGE_DIRECTORY
from versionhq._utils.logger import Logger


class BaseKnowledgeSource(BaseModel, ABC):
    """
    Abstract base class for knowledge sources: csv, json, excel, pdf, string, and docling.
    """

    chunk_size: int = 4000
    chunk_overlap: int = 200
    chunks: List[str] = Field(default_factory=list)
    chunk_embeddings: List[np.ndarray] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)
    storage: Optional[KnowledgeStorage] = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)  # Currently unused
    collection_name: Optional[str] = Field(default=None)

    @abstractmethod
    def validate_content(self) -> Any:
        """Load and preprocess content from the source."""
        pass

    @abstractmethod
    def add(self) -> None:
        """Process content, chunk it, compute embeddings, and save them."""
        pass

    def get_embeddings(self) -> List[np.ndarray]:
        """Return the list of embeddings for the chunks."""
        return self.chunk_embeddings

    def _chunk_text(self, text: str) -> List[str]:
        """
        Utility method to split text into chunks.
        """

        return [
            text[i : i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
        ]

    def _save_documents(self):
        """
        Save the documents to the storage.
        This method should be called after the chunks and embeddings are generated.
        """
        if self.storage:
            self.storage.save(self.chunks)
        else:
            raise ValueError("No storage found to save documents.")



class StringKnowledgeSource(BaseKnowledgeSource):
    """
    A knowledge source that stores and queries plain text content using embeddings.
    """

    content: str = Field(...)
    collection_name: Optional[str] = Field(default=None)

    def model_post_init(self, _):
        """Post-initialization method to validate content."""
        self.validate_content()

    def validate_content(self):
        """Validate string content."""
        if not isinstance(self.content, str):
            raise ValueError("StringKnowledgeSource only accepts string content")

    def add(self) -> None:
        """
        Add string content to the knowledge source, chunk it, compute embeddings, and save them.
        """
        new_chunks = self._chunk_text(self.content)
        self.chunks.extend(new_chunks)
        self._save_documents()


    def _chunk_text(self, text: str) -> List[str]:
        """
        Utility method to split text into chunks.
        """
        return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]



class BaseFileKnowledgeSource(BaseKnowledgeSource, ABC):
    """Base class for knowledge sources that load content from files."""

    _logger: Logger = Logger(verbose=True)
    file_paths: Optional[Path | List[Path] | str | List[str]] = Field(default_factory=list)
    content: Dict[Path, str] = Field(init=False, default_factory=dict)
    storage: Optional[KnowledgeStorage] = Field(default=None)
    safe_file_paths: List[Path] = Field(default_factory=list, description="store a list of `Path` objects from self.file_paths")


    @field_validator("file_paths", mode="before")
    def validate_file_path(cls, v, info):
        """
        Validate if at least one valid file path is provided.
        """
        if v is None and info.data.get("file_paths") is None:
            raise ValueError("Either file_path or file_paths must be provided")
        return v


    def model_post_init(self, _) -> None:
        """
        Post-initialization method to load content.
        """
        self.safe_file_paths = self._process_file_paths()
        self.validate_content()
        self.content = self.load_content()


    @abstractmethod
    def load_content(self) -> Dict[Path, str]:
        """
        Load and preprocess file content. Should be overridden by subclasses.
        Assume that the file path is relative to the project root in the knowledge directory.
        """
        pass


    def validate_content(self):
        """
        Validate the given file paths.
        """
        for path in self.safe_file_paths:
            if not path.exists():
                self._logger.log(
                    "error",
                    f"File not found: {path}. Try adding sources to the knowledge directory. If it's inside the knowledge directory, use the relative path.",
                    color="red",
                )
                raise FileNotFoundError(f"File not found: {path}")
            if not path.is_file():
                self._logger.log("error", f"Path is not a file: {path}", color="red")


    def _save_documents(self):
        if self.storage:
            self.storage.save(self.chunks)
        else:
            raise ValueError("No storage found to save documents.")


    def convert_to_path(self, path: Path | str) -> Path:
        """
        Convert a path to a Path object.
        """
        return Path(KNOWLEDGE_DIRECTORY + "/" + path) if isinstance(path, str) else path


    def _process_file_paths(self) -> List[Path]:
        """
        Convert file_path to a list of Path objects.
        """

        if self.file_paths is None:
            raise ValueError("Your source must be provided with a file_paths: []")

        path_list: List[Path | str] = [self.file_paths] if isinstance(self.file_paths, (str, Path)) else list(self.file_paths) if isinstance(self.file_paths, list) else []

        if not path_list:
            raise ValueError(
                "file_path/file_paths must be a Path, str, or a list of these types"
            )

        return [self.convert_to_path(path) for path in path_list]



class TextFileKnowledgeSource(BaseFileKnowledgeSource):
    """
    A knowledge source class that stores and queries text file content using embeddings.
    """

    def load_content(self) -> Dict[Path, str]:
        """
        Load and preprocess text file content.
        """

        content = {}
        for path in self.safe_file_paths:
            path = self.convert_to_path(path)
            with open(path, "r", encoding="utf-8") as f:
                content[path] = f.read()
        return content


    def add(self) -> None:
        """
        Add text file content to the knowledge source, chunk it, compute embeddings, and save the embeddings.
        """
        for _, text in self.content.items():
            new_chunks = self._chunk_text(text)
            self.chunks.extend(new_chunks)
        self._save_documents()


    def _chunk_text(self, text: str) -> List[str]:
        """
        Utility method to split text into chunks.
        """
        return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]



class PDFKnowledgeSource(BaseFileKnowledgeSource):
    """
    A knowledge source class that stores and queries PDF file content using embeddings.
    """

    def load_content(self) -> Dict[Path, str]:
        """
        Load and preprocess PDF file content.
        """
        pdfplumber = self._import_pdfplumber()
        content = {}
        for path in self.safe_file_paths:
            text = ""
            path = self.convert_to_path(path)
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            content[path] = text
        return content


    def _import_pdfplumber(self):
        """
        Dynamically import pdfplumber.
        """
        try:
            import pdfplumber
            return pdfplumber
        except ImportError:
            raise ImportError("pdfplumber is not installed. Please install it with: pip install pdfplumber")


    def add(self) -> None:
        """
        Add PDF file content to the knowledge source, chunk it, compute embeddings, and save the embeddings.
        """
        for _, text in self.content.items():
            new_chunks = self._chunk_text(text)
            self.chunks.extend(new_chunks)
        self._save_documents()


    def _chunk_text(self, text: str) -> List[str]:
        """
        Utility method to split text into chunks.
        """
        return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]



class CSVKnowledgeSource(BaseFileKnowledgeSource):
    """
    A knowledge source class that stores and queries CSV file content using embeddings.
    """

    def load_content(self) -> Dict[Path, str]:
        """
        Load and preprocess CSV file content.
        """
        content_dict = {}
        for file_path in self.safe_file_paths:
            with open(file_path, "r", encoding="utf-8") as csvfile:
                reader = csv.reader(csvfile)
                content = ""
                for row in reader:
                    content += " ".join(row) + "\n"
                content_dict[file_path] = content

        return content_dict


    def add(self) -> None:
        """
        Add CSV file content to the knowledge source, chunk it, compute embeddings,
        and save the embeddings.
        """
        content_str = str(self.content) if isinstance(self.content, dict) else self.content
        new_chunks = self._chunk_text(content_str)
        self.chunks.extend(new_chunks)
        self._save_documents()


    def _chunk_text(self, text: str) -> List[str]:
        """
        Utility method to split text into chunks.
        """
        return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]



class JSONKnowledgeSource(BaseFileKnowledgeSource):
    """
    A knowledge source class that stores and queries JSON file content using embeddings.
    """

    def load_content(self) -> Dict[Path, str]:
        """
        Load and preprocess JSON file content.
        """
        content: Dict[Path, str] = {}
        for path in self.safe_file_paths:
            path = self.convert_to_path(path)
            with open(path, "r", encoding="utf-8") as json_file:
                data = json.load(json_file)
            content[path] = self._json_to_text(data)
        return content

    def _json_to_text(self, data: Any, level: int = 0) -> str:
        """
        Recursively convert JSON data to a text representation.
        """
        text = ""
        indent = "  " * level
        if isinstance(data, dict):
            for key, value in data.items():
                text += f"{indent}{key}: {self._json_to_text(value, level + 1)}\n"
        elif isinstance(data, list):
            for item in data:
                text += f"{indent}- {self._json_to_text(item, level + 1)}\n"
        else:
            text += f"{str(data)}"
        return text


    def add(self) -> None:
        """
        Add JSON file content to the knowledge source, chunk it, compute embeddings, and save the embeddings.
        """
        content_str = str(self.content) if isinstance(self.content, dict) else self.content
        new_chunks = self._chunk_text(content_str)
        self.chunks.extend(new_chunks)
        self._save_documents()


    def _chunk_text(self, text: str) -> List[str]:
        """
        Utility method to split text into chunks.
        """
        return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]



class ExcelKnowledgeSource(BaseFileKnowledgeSource):
    """
    A knowledge source that stores and queries Excel file content using embeddings.
    """

    def load_content(self) -> Dict[Path, str]:
        """
        Load and preprocess Excel file content.
        """

        pd = self._import_dependencies()
        content_dict = {}
        for file_path in self.safe_file_paths:
            file_path = self.convert_to_path(file_path)
            df = pd.read_excel(file_path)
            content = df.to_csv(index=False)
            content_dict[file_path] = content
        return content_dict

    def _import_dependencies(self):
        """
        Dynamically import dependencies.
        """
        try:
            import pandas as pd
            return pd
        except ImportError as e:
            missing_package = str(e).split()[-1]
            raise ImportError(
                f"{missing_package} is not installed. Please install it with: pip install {missing_package}"
            )

    def add(self) -> None:
        """
        Add Excel file content to the knowledge source, chunk it, compute embeddings, and save the embeddings.
        """
        content_str = "\n".join(str(value) for value in self.content.values()) if isinstance(self.content, dict) else str(self.content)
        new_chunks = self._chunk_text(content_str)
        self.chunks.extend(new_chunks)
        self._save_documents()


    def _chunk_text(self, text: str) -> List[str]:
        """
        Utility method to split text into chunks.
        """
        return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]
