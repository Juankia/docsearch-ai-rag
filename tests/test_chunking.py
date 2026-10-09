import pytest

from docsearch.config import ChunkingConfig
from docsearch.domain import DocType, LoadedDocument
from docsearch.ingestion.chunking import Chunker


def _doc(text: str, doc_type: DocType = DocType.TEXT) -> LoadedDocument:
    return LoadedDocument(doc_id="d1", source="d1.txt", doc_type=doc_type, text=text)


def test_chunks_respect_size_and_carry_metadata() -> None:
    chunker = Chunker({DocType.TEXT: ChunkingConfig(chunk_size=100, chunk_overlap=20)})
    text = " ".join(f"Sentence number {i} talks about topic {i}." for i in range(40))
    chunks = chunker.split(_doc(text))

    assert len(chunks) > 5
    assert all(len(c.text) <= 100 for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    first = chunks[0]
    assert first.chunk_id == "d1:0"
    assert first.metadata["doc_id"] == "d1"
    assert first.metadata["doc_type"] == "text"
    assert first.metadata["source"] == "d1.txt"


def test_overlap_repeats_text_between_neighbours() -> None:
    chunker = Chunker({DocType.TEXT: ChunkingConfig(chunk_size=80, chunk_overlap=30)})
    words = " ".join(f"w{i}" for i in range(100))
    chunks = chunker.split(_doc(words))
    tail_of_first = chunks[0].text.split()[-1]
    assert tail_of_first in chunks[1].text


def test_markdown_splits_on_headings_first() -> None:
    md = (
        "# Doc\n\n## Vacation\n"
        + "Vacation text. " * 20
        + "\n\n## Expenses\n"
        + "Expense text. " * 20
    )
    chunker = Chunker({DocType.MARKDOWN: ChunkingConfig(chunk_size=400, chunk_overlap=0)})
    chunks = chunker.split(_doc(md, DocType.MARKDOWN))
    assert any(c.text.startswith("## Expenses") for c in chunks)
    assert not any("Vacation" in c.text and "Expense" in c.text for c in chunks)


def test_per_type_configuration_is_applied() -> None:
    chunker = Chunker(
        {
            DocType.TEXT: ChunkingConfig(chunk_size=50, chunk_overlap=0),
            DocType.PDF: ChunkingConfig(chunk_size=500, chunk_overlap=0),
        }
    )
    text = "alpha beta gamma delta " * 30
    n_text = len(chunker.split(_doc(text, DocType.TEXT)))
    n_pdf = len(chunker.split(_doc(text, DocType.PDF)))
    assert n_text > n_pdf
    assert chunker.config_for(DocType.PDF).chunk_size == 500


def test_invalid_overlap_rejected() -> None:
    with pytest.raises(ValueError):
        Chunker({DocType.TEXT: ChunkingConfig(chunk_size=100, chunk_overlap=100)})
