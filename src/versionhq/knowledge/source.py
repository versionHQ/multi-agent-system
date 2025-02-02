import csv
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from versionhq.knowledge.storage import KnowledgeStorage
from versionhq.storage.utils import fetch_db_storage_path
from versionhq._utils.vars import KNOWLEDGE_DIRECTORY
from versionhq._utils.logger import Logger


class BaseKnowledgeSource(BaseModel, ABC):
    """
    Abstract base class for knowledge sources: csv, json, excel, pdf, string, and docling.
    """
    _logger: Logger = Logger(verbose=True)

    chunk_size: int = 3000
    chunk_overlap: int = 200
    chunks: List[str] = Field(default_factory=list)
    chunk_embeddings: List[np.ndarray] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)
    storage: Optional[KnowledgeStorage] = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    collection_name: Optional[str] = Field(default=None)


    @abstractmethod
    def validate_content(self, **kwargs) -> Any:
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
        return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]


    def _save_documents(self) -> None:
        """
        Save the documents to the given (or newly created) storage on ChromaDB.
        This method should be called after the chunks and embeddings are generated.
        """
        # if not self.chunks or self.chunk_embeddings:
        #     self._logger.log(level="warning", message="Chunks or chunk embeddings are missing. Save docs after creating them.", color="yellow")
        #     return

        try:
            if self.storage:
                self.storage.save(documents=self.chunks, metadata=self.metadata)

            else:
                storage = KnowledgeStorage(collection_name=self.collection_name) if self.collection_name else KnowledgeStorage()
                storage.initialize_knowledge_storage()
                self.storage = storage
                self.storage.save(documents=self.chunks, metadata=self.metadata)

        except:
            self._logger.log(level="error", message="No storage found or created to save the documents.", color="red")
            return
            # raise ValueError("No storage found to save documents.")



class StringKnowledgeSource(BaseKnowledgeSource):
    """
    A knowledge source that stores and queries plain text content using embeddings.
    """

    content: str = Field(...)
    collection_name: Optional[str] = Field(default=None)

    def model_post_init(self, _):
        """Post-initialization method to validate content."""
        self.validate_content()
        self._save_documents()


    def validate_content(self):
        """Validate string content."""
        if not isinstance(self.content, str):
            raise ValueError("StringKnowledgeSource only accepts string content")


    def add(self) -> None:
        """
        Add string content to the knowledge source, chunk it, compute embeddings, and save them.
        """
        new_chunks = self._chunk_text(text=self.content)
        self.chunks.extend(new_chunks)
        self._save_documents()



class BaseFileKnowledgeSource(BaseKnowledgeSource, ABC):
    """Base class for knowledge sources that load content from files."""

    file_paths: Optional[Path | List[Path] | str | List[str]] = Field(default_factory=list)
    content: Dict[Path, str] = Field(init=False, default_factory=dict)
    storage: Optional[KnowledgeStorage] = Field(default=None)
    valid_file_paths: List[Path] = Field(default_factory=list, description="store a list of `Path` objects from self.file_paths")


    @field_validator("file_paths", mode="before")
    def validate_file_path(cls, v, info):
        """
        Validate if at least one valid file path is provided.
        """
        if v is None and info.data.get("file_paths") is None:
            raise ValueError("Either file_path or file_paths must be provided")
        return v


    def validate_content(self, path: str | Path) -> List[Path]:
        """
        Convert the given path to a Path object, and validate if the path exists and refers to a file.)
        """

        path_instance = Path(KNOWLEDGE_DIRECTORY + "/" + path) if isinstance(path, str) else path

        if not path_instance.exists():
            abs_path = fetch_db_storage_path()
            path_instance = Path(abs_path + "/" + KNOWLEDGE_DIRECTORY + "/" + path) if isinstance(path, str) else path

            if not path_instance.exists():
                self._logger.log(level="error", message="File path not found.", color="red")
                raise ValueError()

            elif not path_instance.is_file():
                self._logger.log(level="error", message="Non-file object was given.", color="red")
                raise ValueError()

        elif not path_instance.is_file():
            self._logger.log(level="error", message="Non-file object was given.", color="red")
            raise ValueError()

        return path_instance



    def _process_file_paths(self) -> List[Path]:
        """
        Convert file_path to a list of Path objects.
        """
        if not self.file_paths:
            self._logger.log(level="error", message="Missing file paths.", color="red")
            raise ValueError("Missing file paths.")


        path_list: List[Path | str] = [self.file_paths] if isinstance(self.file_paths, (str, Path)) else list(self.file_paths) if isinstance(self.file_paths, list) else []
        valid_path_list = list()

        if not path_list:
            self._logger.log(level="error", message="Missing valid file paths.", color="red")
            raise ValueError("Your source must be provided with file_paths: []")

        for item in path_list:
            valid_path = self.validate_content(item)
            if valid_path:
                valid_path_list.append(valid_path)

        return valid_path_list


    def model_post_init(self, _) -> None:
        """
        Post-initialization method to load content.
        """
        self.valid_file_paths = self._process_file_paths()
        self.content = self.load_content()
        self._save_documents()


    @abstractmethod
    def load_content(self) -> Dict[Path, str]:
        """
        Load and preprocess file content. Should be overridden by subclasses.
        Assume that the file path is relative to the project root in the knowledge directory.
        """
        pass



class TextFileKnowledgeSource(BaseFileKnowledgeSource):
    """
    A knowledge source class that stores and queries text file content using embeddings.
    """

    def load_content(self) -> Dict[Path, str]:
        """
        Load and preprocess text file content.
        """
        content = {}
        for path in self.valid_file_paths:
            path = self.validate_content(path=path)
            with open(path, "r", encoding="utf-8") as f:
                content[path] = f.read()
        return content


    def add(self) -> None:
        """
        Add text file content to the knowledge source, chunk it, compute embeddings, and save the embeddings.
        """
        for _, text in self.content.items():
            new_chunks = self._chunk_text(text=text)
            self.chunks.extend(new_chunks)

        self._save_documents()



class PDFKnowledgeSource(BaseFileKnowledgeSource):
    """
    A knowledge source class that stores and queries PDF file content using embeddings.
    """

    def load_content(self) -> Dict[Path, str]:
        """
        Load and preprocess PDF file content.
        """
        self._import_pdfplumber()

        import pdfplumber

        content = {}
        for path in self.valid_file_paths:
            text = ""
            path = self.validate_content(path)
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
        except ImportError:
            try:
                import os
                os.system("uv add pdfplumber --optional pdfplumber")
            except:
                raise ImportError("pdfplumber is not installed. Please install it with: uv add pdfplumber")


    def add(self) -> None:
        """
        Add PDF file content to the knowledge source, chunk it, compute embeddings, and save the embeddings.
        """
        for _, text in self.content.items():
            new_chunks = self._chunk_text(text=text)
            self.chunks.extend(new_chunks)

        self._save_documents()




class CSVKnowledgeSource(BaseFileKnowledgeSource):
    """
    A knowledge source class that stores and queries CSV file content using embeddings.
    """

    def load_content(self) -> Dict[Path, str]:
        """
        Load and preprocess CSV file content.
        """
        content_dict = {}
        for file_path in self.valid_file_paths:
            with open(file_path, "r", encoding="utf-8") as csvfile:
                reader = csv.reader(csvfile)
                content = ""
                for row in reader:
                    content += " ".join(row) + "\n"
                content_dict[file_path] = content

        return content_dict


    def add(self) -> None:
        """
        Add CSV file content to the knowledge source, chunk it, compute embeddings, and save the embeddings.
        """
        content_str = str(self.content) if isinstance(self.content, dict) else self.content
        new_chunks = self._chunk_text(text=content_str)
        self.chunks.extend(new_chunks)
        self._save_documents()



class JSONKnowledgeSource(BaseFileKnowledgeSource):
    """
    A knowledge source class that stores and queries JSON file content using embeddings.
    """

    def load_content(self) -> Dict[Path, str]:
        """
        Load and preprocess JSON file content.
        """
        content: Dict[Path, str] = {}
        for path in self.valid_file_paths:
            path = self.validate_content(path)
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
        new_chunks = self._chunk_text(text=content_str)
        self.chunks.extend(new_chunks)
        self._save_documents()



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
        for file_path in self.valid_file_paths:
            file_path = self.validate_content(file_path)
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
            try:
                import os
                os.system("uv add pandas --optional pandas")
                import pandas as pd
                return pd
            except:
                missing_package = str(e).split()[-1]
                raise ImportError(
                    f"{missing_package} is not installed. Please install it with: pip install {missing_package}"
                )


    def add(self) -> None:
        """
        Add Excel file content to the knowledge source, chunk it, compute embeddings, and save the embeddings.
        """
        content_str = "\n".join(str(value) for value in self.content.values()) if isinstance(self.content, dict) else str(self.content)
        new_chunks = self._chunk_text(text=content_str)
        self.chunks.extend(new_chunks)
        self._save_documents()
