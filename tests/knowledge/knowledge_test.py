from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest

from versionhq import (
    StringKnowledgeSource,
    TextFileKnowledgeSource,
    PDFKnowledgeSource,
    CSVKnowledgeSource,
    JSONKnowledgeSource,
    ExcelKnowledgeSource,
    DoclingSource
)

str_content = "Kuriko's favorite color is black, and she likes Japanese food."
str_content_long = (
    "Kuriko drinks her coffee black every morning. It is her morning ritual."
    "Kuriko designs bridges."
    "Kuriko owns a fluffy cat. Her name is Luna."
    "Kuriko enjoys solving complex problems."
    "Kuriko's cat naps on her keyboard."
    "Kuriko loves the smell of freshly brewed coffee."
    "Kuriko works on a computer all day and her eyesights are getting poor."
    "Kuriko's apartment is filled with cat toys. She loves to spoil her cat."
    "Kuriko reads engineering journals."
    "Kuriko enjoys a peaceful morning at home."
    "Kuriko is a skilled problem-solver and good at math."
    "Luna loves to play fetch and naps on Kuriko's keyboard."
    "Kuriko works on building projects."
    "Kuriko enjoys trying new coffee blends."
    "Kuriko uses CAD software."
    "Kuriko recently bought 3 Android smartphones."
    "Kurko's cat is her best friend."
    "Kuriko works in a busy office with 100 people on the floor."
    "And she works long hours."
    "Luna is 10 months old."
    "She speaks 3 languages."
    "She is passionate about weight training, tracking weekly volume load to grow muscle."
    "Kuriko recently traveled across Latin America."
    "random sentence, random sentence, random sentence."
)
str_content_long_2 = (
    "J drinks his coffee black every morning."
    "J designs bridges."
    "J owns a fluffy cat. Her name is Luna."
    "J enjoys solving complex problems."
    "J's cat naps on his keyboard."
    "J loves the smell of freshly brewed coffee."
    "J works on a computer all day."
    "J reads engineering journals."
    "J is a skilled problem-solver and good at math."
    "Luna loves to play fetch and naps on J's keyboard."
    "J works on building projects."
    "J enjoys trying new coffee blends."
    "J uses CAD software."
    "J recently bought a laptop."
    "J works in a busy office with 100 people on the floor."
    "And he works long hours."
    "Luna is 10 months old."
    "J recently traveled across Southeast Asia."
)


@pytest.fixture(autouse=True)
def mock_vector_db():
    """
    Mock vector database operations. the query method to return a predefined response
    """

    with patch("versionhq.knowledge.storage.KnowledgeStorage") as mock:
        instance = mock.return_value
        instance.query.return_value = [
            {
                "context": "Kuriko's favorite color is black, and she likes Japanese food.",
                "score": 0.9,
            }
        ]
        instance.reset.return_value = None
        yield instance


@pytest.fixture(autouse=True)
def reset_knowledge_storage(mock_vector_db):
    """
    Fixture to reset knowledge storage before each test.
    """
    yield


def test_string_knowledge_source(mock_vector_db):
    """
    Create a plane text storage source and query them.
    """

    string_source = StringKnowledgeSource(content=str_content, metadata={ "preference": "personal" })
    mock_vector_db.sources = [string_source]
    mock_vector_db.query.return_value = [{ "context": str_content, "score": 0.9 }]

    query = "What is Kuriko's favorite color?"
    results = mock_vector_db.query(query)

    assert any("black" in result["context"].lower() for result in results)
    mock_vector_db.query.assert_called_once()


# @pytest.mark.vcr(filter_headers=["authorization"])
def test_2k_string_knowledge_source(mock_vector_db):
    string_source = StringKnowledgeSource(content=str_content_long, metadata={ "preference": "personal" })
    mock_vector_db.sources = [string_source]
    mock_vector_db.query.return_value = [{ "context": str_content_long, "score": 0.9 }]

    query = "What is Kuriko's cat name?"
    results = mock_vector_db.query(query)

    assert any("luna" in result["context"].lower() for result in results)
    mock_vector_db.query.assert_called_once()


def test_multiple_string_knowledge_sources(mock_vector_db):
    contents = [
        "Kuriko decided to take a stray kitten home.",
        "She enjoys quiet and peaceful morning.",
        "She named the kitten Luna."
    ]
    string_sources = [StringKnowledgeSource(content=content, metadata={ "preference": "personal" }) for content in contents]
    mock_vector_db.query.return_value = [{ "context": "Kuriko has a cat named Luna.", "score": 0.9 }]
    mock_vector_db.sources = string_sources

    query = "What is her cat's name?"
    results = mock_vector_db.query(query)

    assert any("luna" in result["context"].lower() for result in results)
    mock_vector_db.query.assert_called_once()


def test_multiple_2k_string_knowledge_sources(mock_vector_db):
    contents = [str_content_long * 2, str_content_long_2 * 2]
    string_sources = [StringKnowledgeSource(content=content, metadata={ "preference": "personal" }) for content in contents]
    mock_vector_db.sources = string_sources
    mock_vector_db.query.return_value = [{ "context": contents[1], "score": 0.9 }]

    query = "What did J buy recently?"
    results = mock_vector_db.query(query)

    assert any("laptop" in result["context"].lower() for result in results)
    mock_vector_db.query.assert_called_once()


def test_single_short_file_knowledge_source(mock_vector_db, tmpdir):
    """
    Create a text file with a short content and query
    """

    file_path = Path(tmpdir.join("short_file.txt"))
    with open(file_path, "w") as f:
        f.write(str_content)

    file_source = TextFileKnowledgeSource(file_paths=[file_path], metadata={ "preference": "personal" })
    mock_vector_db.sources = [file_source]
    mock_vector_db.query.return_value = [{"context": str_content, "score": 0.9}]

    query = "What color does Kuriko like?"
    results = mock_vector_db.query(query)

    assert any("black" in result["context"].lower() for result in results)
    mock_vector_db.query.assert_called_once()


def test_single_2k_character_file(mock_vector_db, tmpdir):
    file_path = Path(tmpdir.join("long_file.txt"))
    with open(file_path, "w") as f:
        f.write(str_content_long)

    file_source = TextFileKnowledgeSource(file_paths=[file_path], metadata={ "preference": "personal" })
    mock_vector_db.sources = [file_source]
    mock_vector_db.query.return_value = [{"context": str_content_long, "score": 0.9}]

    query = "Where did Kuriko travel recently?"
    results = mock_vector_db.query(query)

    assert any("latin america" in result["context"].lower() for result in results)
    mock_vector_db.query.assert_called_once()


def test_multiple_short_file_knowledge_sources(mock_vector_db, tmpdir):
    contents = [
        {
            "content": "John works as a software engineer.",
            "metadata": {"category": "profession", "source": "occupation"},
        },
        {
            "content": "John lives in San Francisco.",
            "metadata": {"category": "city", "source": "personal"},
        },
        {
            "content": "John enjoys cooking Italian food.",
            "metadata": {"category": "hobby", "source": "personal"},
        },
    ]
    file_paths = []
    for i, item in enumerate(contents):
        file_path = Path(tmpdir.join(f"file_{i}.txt"))
        with open(file_path, "w") as f:
            f.write(item["content"])
        file_paths.append((file_path, item["metadata"]))

    file_sources = [TextFileKnowledgeSource(file_paths=[path], metadata=metadata) for path, metadata in file_paths]
    mock_vector_db.sources = file_sources
    mock_vector_db.query.return_value = [{"context": "John lives in San Francisco.", "score": 0.9}]

    query = "What city does he reside in?"
    results = mock_vector_db.query(query)

    assert any("san francisco" in result["context"].lower() for result in results)
    mock_vector_db.query.assert_called_once()


def test_multiple_2k_character_files(mock_vector_db, tmpdir):
    contents = [str_content_long * 2, str_content_long_2 * 2]
    file_paths = []
    for i, content in enumerate(contents):
        file_path = Path(tmpdir.join(f"long_file_{i}.txt"))
        with open(file_path, "w") as f:
            f.write(content)
        file_paths.append(file_path)

    file_sources = [TextFileKnowledgeSource(file_paths=[path], metadata={"preference": "personal"}) for path in file_paths]
    mock_vector_db.sources = file_sources
    mock_vector_db.query.return_value = [{ "context": contents[1], "score": 0.9 }]

    query = "What is J's cat name?"
    results = mock_vector_db.query(query)

    assert any("luna" in result["context"].lower() for result in results)
    mock_vector_db.query.assert_called_once()


# @pytest.mark.vcr(filter_headers=["authorization"])
def test_hybrid_string_and_file_knowledge_source(mock_vector_db, tmpdir):
    str_contents = [
        "Kuriko decided to take a stray kitten home.",
        "She enjoys quiet and peaceful morning.",
        "She named the kitten Luna."
    ]
    string_sources = [StringKnowledgeSource(content=item, metadata={ "preference": "personal" }) for item in str_contents]

    file_contents = ["J works on a computer all day.","J reads engineering journals."]
    file_paths = []
    for i, content in enumerate(file_contents):
        file_path = Path(tmpdir.join(f"file_{i}.txt"))
        with open(file_path, "w") as f:
            f.write(content)
        file_paths.append(file_path)
    file_sources = [TextFileKnowledgeSource(file_paths=[item], metadata={"preference": "personal"}) for item in file_paths]

    mock_vector_db.sources = string_sources + file_sources
    mock_vector_db.query.return_value = [{"context": file_contents[1], "score": 0.9 }]

    query = "What does J read?"
    results = mock_vector_db.query(query)

    assert any("engineering journals" in result["context"].lower() for result in results)
    mock_vector_db.query.assert_called_once()


def test_pdf_knowledge_source(mock_vector_db):
    current_dir = Path(__file__).parent
    pdf_path = current_dir / "mock_report_compressed.pdf"
    pdf_source = PDFKnowledgeSource(file_paths=[pdf_path], metadata={ "preference": "personal" })
    assert pdf_source.valid_file_paths == [pdf_path]
    assert pdf_source.content is not None

    mock_vector_db.sources = [pdf_source]
    mock_vector_db.query.return_value = [{"context": "McKinsey faces a challenge of manually curating and tagging documents, and takes hierarchical classification approach to query their internal knowledge base.", "score": 0.9} ]

    query = "What methodology does McKinsey take?"
    results = mock_vector_db.query(query)

    assert any("hierarchical classification" in result["context"].lower() for result in results)
    mock_vector_db.query.assert_called_once()


# @pytest.mark.vcr(filter_headers=["authorization"])
def test_csv_knowledge_source(mock_vector_db, tmpdir):
    csv_content = [
        ["Name", "Age", "City"],
        ["Kuriko", "55", "New York"],
        ["Alice", "30", "Los Angeles"],
        ["J", "60", "Chicago"],
    ]
    csv_path = Path(tmpdir.join("data.csv"))
    with open(csv_path, "w", encoding="utf-8") as f:
        for row in csv_content:
            f.write(",".join(row) + "\n")
    csv_source = CSVKnowledgeSource(file_paths=[csv_path], metadata={"preference": "personal"})
    mock_vector_db.sources = [csv_source]
    mock_vector_db.query.return_value = [{"context": "Kuriko is 55 years old.", "score": 0.9}]

    query = "How old is Kuriko?"
    results = mock_vector_db.query(query)

    assert any("55" in result["context"] for result in results)
    mock_vector_db.query.assert_called_once()


def test_excel_knowledge_source(mock_vector_db, tmpdir):
    import pandas as pd
    excel_data = {
        "Name": ["Kuriko", "Alice", "J"],
        "Age": [55, 30, 60],
        "City": ["New York", "Los Angeles", "Chicago"],
    }
    df = pd.DataFrame(excel_data)
    excel_path = Path(tmpdir.join("data.xlsx"))
    df.to_excel(excel_path, index=False)

    excel_source = ExcelKnowledgeSource(file_paths=[excel_path], metadata={"preference": "personal"})
    mock_vector_db.sources = [excel_source]
    mock_vector_db.query.return_value = [{ "context": "Kuriko is 55 years old.", "score": 0.9 }]

    query = "How old is Kuriko?"
    results = mock_vector_db.query(query)

    assert any("55" in result["context"] for result in results)
    mock_vector_db.query.assert_called_once()


def test_json_knowledge_source(mock_vector_db, tmpdir):
    json_data = {
        "people": [
            {"name": "Kuriko", "age": 55, "city": "New York"},
            {"name": "Alice", "age": 30, "city": "Los Angeles"},
            {"name": "J", "age": 60, "city": "Chicago"},
        ]
    }
    json_path = Path(tmpdir.join("data.json"))
    with open(json_path, "w", encoding="utf-8") as f:
        import json
        json.dump(json_data, f)

    json_source = JSONKnowledgeSource(file_paths=[json_path], metadata={ "preference": "personal" })
    mock_vector_db.sources = [json_source]
    mock_vector_db.query.return_value = [{"context": "Alice lives in Los Angeles.", "score": 0.9}]

    query = "Where does Alice reside?"
    results = mock_vector_db.query(query)

    assert any("los angeles" in result["context"].lower() for result in results)
    mock_vector_db.query.assert_called_once()


def test_docling_source(mock_vector_db):
    docling_source = DoclingSource(file_paths=["https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",])
    mock_vector_db.sources = [docling_source]
    mock_vector_db.query.return_value = [
        {
            "context": "Reward hacking is a technique used to improve the performance of reinforcement learning agents.",
            "score": 0.9,
        }
    ]

    query = "What is reward hacking?"
    results = mock_vector_db.query(query)

    assert any("reward hacking" in result["context"].lower() for result in results)
    mock_vector_db.query.assert_called_once()


def test_multiple_docling_sources():
    urls: List[Path | str] = [
        "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
        "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    ]
    docling_source = DoclingSource(file_paths=urls)

    assert docling_source.file_paths == urls
    assert docling_source.content is not None
