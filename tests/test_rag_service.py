from pathlib import Path

from docsearch.config import Settings
from docsearch.container import build_service
from docsearch.evaluation import evaluate, ingest_directory, load_qa
from docsearch.rag.generation import ExtractiveGenerator, LLMGenerator
from docsearch.rag.prompting import REFUSAL, SYSTEM_PROMPT, extract_citations, is_refusal
from docsearch.rag.service import RAGService
from tests.fakes import FakeLLM

HANDBOOK = b"""# Handbook

## Vacation
Employees receive 22 days of paid vacation per year.

## Expenses
Expenses must be submitted within 30 days with a receipt.
"""


def test_ingest_then_answer_with_citation(service: RAGService, fake_llm: FakeLLM) -> None:
    result = service.ingest(HANDBOOK, "handbook.md")
    assert result.doc_id == "handbook"
    assert result.chunks >= 1
    assert result.replaced_chunks == 0

    answer = service.answer("How many days of paid vacation do employees receive?")
    assert answer.grounded
    assert answer.cited and answer.cited[0].startswith("handbook:")
    assert answer.sources[0].score >= service.min_score

    system, user = fake_llm.prompts[-1]
    assert system == SYSTEM_PROMPT
    assert "22 days of paid vacation" in user
    assert '<excerpt id="handbook:' in user


def test_reingest_replaces_instead_of_duplicating(service: RAGService) -> None:
    first = service.ingest(HANDBOOK, "handbook.md")
    second = service.ingest(HANDBOOK.replace(b"22 days", b"25 days"), "handbook.md")
    assert second.replaced_chunks == first.chunks
    assert service.store.count("handbook") == second.chunks
    texts = " ".join(h.chunk.text for h in service.retrieve("paid vacation days", k=10))
    assert "25 days" in texts and "22 days" not in texts


def test_refuses_without_calling_llm_when_nothing_relevant(
    service: RAGService, fake_llm: FakeLLM
) -> None:
    service.ingest(HANDBOOK, "handbook.md")
    answer = service.answer("quantum chromodynamics lattice gauge")
    assert answer.answer == REFUSAL
    assert not answer.grounded
    assert answer.extra["reason"] == "no_relevant_context"
    assert fake_llm.prompts == []


def test_llm_refusal_is_propagated(service: RAGService, fake_llm: FakeLLM) -> None:
    service.ingest(HANDBOOK, "handbook.md")
    fake_llm.reply = REFUSAL
    answer = service.answer("How many vacation days do employees receive?")
    assert answer.answer == REFUSAL
    assert not answer.grounded


def test_hallucinated_citations_are_dropped(service: RAGService, fake_llm: FakeLLM) -> None:
    service.ingest(HANDBOOK, "handbook.md")
    fake_llm.reply = "22 days [handbook:0] [made-up:99]"
    answer = service.answer("How many vacation days do employees receive?")
    assert answer.cited == ["handbook:0"]


def test_metadata_filter_restricts_retrieval(service: RAGService) -> None:
    service.ingest(HANDBOOK, "handbook.md", metadata={"department": "hr"})
    service.ingest(
        b"Vacation rentals are not covered by travel policy.",
        "travel.txt",
        metadata={"department": "finance"},
    )
    hits = service.retrieve("vacation", k=10, filters={"department": "finance"})
    assert hits and {h.chunk.doc_id for h in hits} == {"travel"}


def test_extract_citations_and_refusal_detection() -> None:
    text = "A [doc-1:0]. B [doc-1:0][faq_v2:12]. Not a cite [x]."
    assert extract_citations(text) == ["doc-1:0", "faq_v2:12"]
    assert extract_citations(text, allowed={"faq_v2:12"}) == ["faq_v2:12"]
    assert is_refusal(REFUSAL)
    assert is_refusal("")
    assert not is_refusal("The answer is 22 days [h:0].")


def test_llm_generator_falls_back_to_refusal_on_empty_reply(service: RAGService) -> None:
    service.ingest(HANDBOOK, "handbook.md")
    gen = LLMGenerator(FakeLLM(reply="   "))
    assert gen.generate("q", service.retrieve("vacation")) == REFUSAL


def test_extractive_generator_quotes_context(service: RAGService) -> None:
    service.ingest(HANDBOOK, "handbook.md")
    hits = service.retrieve("When must expenses be submitted?")
    text = ExtractiveGenerator().generate("When must expenses be submitted?", hits)
    assert "within 30 days" in text
    assert extract_citations(text)


def test_eval_on_samples_meets_baseline(samples_dir: Path) -> None:
    """Regression guard for retrieval quality with the offline hashing embedder."""
    svc = build_service(Settings(_env_file=None))  # type: ignore[call-arg]
    ingest_directory(svc, samples_dir)
    report = evaluate(svc, load_qa(samples_dir.parent / "eval" / "qa.jsonl"), ks=[1, 5])
    assert report.n_answerable >= 30
    assert report.hit_rate[5] >= 0.9
    assert report.mrr >= 0.8
    assert report.refusal_accuracy == 1.0
