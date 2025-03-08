from pathlib import Path
from typing import Iterator, List

try:
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter
    from docling.exceptions import ConversionError
    from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker
    from docling_core.types.doc.document import DoclingDocument
    DOCLING_AVAILABLE = True
# except ImportError:
    # import envoy
    # envoy.run("uv add docling --optional docling")
    # DOCLING_AVAILABLE = True
except:
    DOCLING_AVAILABLE = False

from pydantic import Field

from versionhq.knowledge.source import BaseKnowledgeSource
from versionhq.storage.utils import fetch_db_storage_path
from versionhq._utils import KNOWLEDGE_DIRECTORY, is_valid_url


class DoclingSource(BaseKnowledgeSource):
    """
    Default docling class for converting documents to markdown or json.
    Support PDF, DOCX, TXT, XLSX, PPTX, MD, Images, and HTML files without any additional dependencies.
    """

    file_paths: List[Path | str] = Field(default_factory=list)
    valid_file_paths: List[Path | str] = Field(default_factory=list)
    content: List["DoclingDocument"] = Field(default_factory=list)
    document_converter: "DocumentConverter" = Field(default_factory=lambda: DocumentConverter(
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
    ))

    def __init__(self, *args, **kwargs):
        if DOCLING_AVAILABLE:
            from docling.datamodel.base_models import InputFormat
            from docling.document_converter import DocumentConverter
            from docling.exceptions import ConversionError
            from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker
            from docling_core.types.doc.document import DoclingDocument

            super().__init__(*args, **kwargs)

        else:
            raise ImportError("The docling package is required. Please install the package using: $ uv add versionhq[docling]")
        # else:
        #     super().__init__(*args, **kwargs)


    def _convert_source_to_docling_documents(self) -> List["DoclingDocument"]:
        conv_results_iter = self.document_converter.convert_all(self.valid_file_paths)
        return [result.document for result in conv_results_iter]


    def _load_content(self) -> List["DoclingDocument"]:
        try:
            return self._convert_source_to_docling_documents()
        except ConversionError as e:
            self._logger.log(level="error", message=f"Error loading content: {str(e)}. Supported formats: {self.document_converter.allowed_formats}", color="red")
            raise e
        except Exception as e:
            self._logger.log(level="error", message=f"Error loading content: {str(e)}", color="red")
            raise e


    def _chunk_doc(self, doc: "DoclingDocument") -> Iterator[str]:
        chunker = HierarchicalChunker()
        for chunk in chunker.chunk(doc):
            yield chunk.text


    def model_post_init(self, _) -> None:
        self.valid_file_paths = self.validate_content()
        self.content.extend(self._load_content())


    def validate_content(self) -> List[Path | str]:
        processed_paths: List[Path | str] = []
        for path in self.file_paths:
            if isinstance(path, str):
                if path.startswith(("http://", "https://")):
                    try:
                        if is_valid_url(path):
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
                        local_path = Path(fetch_db_storage_path() + "/" + KNOWLEDGE_DIRECTORY + "/" + path) # try with abs. path
                        if local_path.exists():
                            processed_paths.append(local_path)
                        else:
                            raise FileNotFoundError(f"File not found: {local_path}")
            else:
                if isinstance(path, Path):
                    processed_paths.append(path)
        return processed_paths


    def add(self) -> None:
        if self.content is None:
            self.model_post_init()

        if self.content:
            for doc in self.content:
                new_chunks_iterable = self._chunk_doc(doc)
                self.chunks.extend(list(new_chunks_iterable))
            self._save_documents()
