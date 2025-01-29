from pathlib import Path
from typing import Iterator, List, Optional
from urllib.parse import urlparse

try:
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter
    from docling.exceptions import ConversionError
    from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker
    from docling_core.types.doc.document import DoclingDocument
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False

from pydantic import Field

from versionhq.knowledge.source import BaseKnowledgeSource
from versionhq._utils.vars import KNOWLEDGE_DIRECTORY
from versionhq._utils.logger import Logger


class DoclingSource(BaseKnowledgeSource):
    """
    Default docling class for converting documents to markdown or json.
    Support PDF, DOCX, TXT, XLSX, PPTX, MD, Images, and HTML files without any additional dependencies.
    """

    def __init__(self, *args, **kwargs):
        if not DOCLING_AVAILABLE:
            raise ImportError("The docling package is required. Please install the package using: $ uv add docling.")

        super().__init__(*args, **kwargs)

    _logger: Logger = Logger(verbose=True)
    file_paths: List[Path | str] = Field(default_factory=list)
    chunks: List[str] = Field(default_factory=list)
    safe_file_paths: List[Path | str] = Field(default_factory=list)
    content: List["DoclingDocument"] = Field(default_factory=list)
    document_converter: "DocumentConverter" = Field(
        default_factory=lambda: DocumentConverter(
            allowed_formats=[
                InputFormat.MD,
                InputFormat.ASCIIDOC,
                InputFormat.PDF,
                InputFormat.DOCX,
                InputFormat.HTML,
                InputFormat.IMAGE,
                InputFormat.XLSX,
                InputFormat.PPTX,
            ]
        )
    )

    def model_post_init(self, _) -> None:
        self.safe_file_paths = self.validate_content()
        self.content = self._load_content()


    def _load_content(self) -> List["DoclingDocument"]:
        try:
            return self._convert_source_to_docling_documents()
        except ConversionError as e:
            self._logger.log(
                level="error",
                message=f"Error loading content: {str(e)}. Supported formats: {self.document_converter.allowed_formats}",
                color="red",
            )
            raise e
        except Exception as e:
            self._logger.log(level="error", message=f"Error loading content: {e}", color="red")
            raise e


    def add(self) -> None:
        if self.content is None:
            return
        for doc in self.content:
            new_chunks_iterable = self._chunk_doc(doc)
            self.chunks.extend(list(new_chunks_iterable))
        self._save_documents()


    def _convert_source_to_docling_documents(self) -> List["DoclingDocument"]:
        conv_results_iter = self.document_converter.convert_all(self.safe_file_paths)
        return [result.document for result in conv_results_iter]


    def _chunk_doc(self, doc: "DoclingDocument") -> Iterator[str]:
        chunker = HierarchicalChunker()
        for chunk in chunker.chunk(doc):
            yield chunk.text


    def validate_content(self) -> List[Path | str]:
        processed_paths: List[Path | str] = []
        for path in self.file_paths:
            if isinstance(path, str):
                if path.startswith(("http://", "https://")):
                    try:
                        if self._validate_url(path):
                            processed_paths.append(path)
                        else:
                            raise ValueError(f"Invalid URL format: {path}")
                    except Exception as e:
                        raise ValueError(f"Invalid URL: {path}. Error: {str(e)}")
                else:
                    local_path = Path(KNOWLEDGE_DIRECTORY + "/" + path)
                    if local_path.exists():
                        processed_paths.append(local_path)
                    else:
                        raise FileNotFoundError(f"File not found: {local_path}")
            else:
                if isinstance(path, Path):
                    processed_paths.append(path)
        return processed_paths


    def _validate_url(self, url: str) -> bool:
        try:
            result = urlparse(url)
            return all(
                [
                    result.scheme in ("http", "https"),
                    result.netloc,
                    len(result.netloc.split(".")) >= 2,  # Ensure domain has TLD
                ]
            )
        except Exception:
            return False
