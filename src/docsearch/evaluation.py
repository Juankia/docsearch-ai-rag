"""Retrieval evaluation: hit-rate@k and MRR over a small labelled Q&A set.

A retrieved chunk counts as relevant when it belongs to the expected document AND
contains the expected evidence span. Matching on evidence (not on chunk ids) keeps the
labels valid when chunking parameters change, which is exactly what we want to compare.

Out-of-scope questions (doc_id = null) measure the refusal gate instead: the question
should be refused because no chunk clears `min_score`.

Usage:
    python -m docsearch.evaluation                      # current settings
    python -m docsearch.evaluation --sweep              # compare chunk sizes
    python -m docsearch.evaluation --k 1 3 5 --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from docsearch.config import ChunkingConfig, Settings
from docsearch.container import build_service
from docsearch.domain import DocType, ScoredChunk
from docsearch.ingestion.chunking import Chunker
from docsearch.rag.service import RAGService

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLES = ROOT / "samples"
DEFAULT_QA = ROOT / "eval" / "qa.jsonl"


@dataclass(frozen=True)
class QAItem:
    id: str
    question: str
    doc_id: str | None
    evidence: str | None

    @property
    def answerable(self) -> bool:
        return self.doc_id is not None


@dataclass
class EvalReport:
    label: str
    n_answerable: int
    n_out_of_scope: int
    hit_rate: dict[int, float]
    mrr: float
    refusal_accuracy: float
    false_refusals: int
    chunks_indexed: int
    misses: list[str]


def load_qa(path: Path) -> list[QAItem]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(QAItem(**json.loads(line)))
    return items


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


def first_relevant_rank(item: QAItem, hits: Sequence[ScoredChunk]) -> int | None:
    """1-based rank of the first relevant chunk, or None."""
    assert item.evidence is not None
    evidence = _squash(item.evidence)
    for rank, hit in enumerate(hits, start=1):
        if hit.chunk.doc_id == item.doc_id and evidence in _squash(hit.chunk.text):
            return rank
    return None


def ingest_directory(service: RAGService, directory: Path) -> None:
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() in {".md", ".txt", ".pdf"}:
            service.ingest(path.read_bytes(), path.name)


def evaluate(
    service: RAGService, items: Sequence[QAItem], ks: Sequence[int], label: str = "default"
) -> EvalReport:
    max_k = max(ks)
    answerable = [i for i in items if i.answerable]
    oos = [i for i in items if not i.answerable]

    hits_at = dict.fromkeys(ks, 0)
    reciprocal_ranks: list[float] = []
    misses: list[str] = []
    false_refusals = 0
    for item in answerable:
        hits = service.retrieve(item.question, k=max_k)
        rank = first_relevant_rank(item, hits)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        for k in ks:
            if rank is not None and rank <= k:
                hits_at[k] += 1
        if rank is None:
            misses.append(item.id)
        if not hits or hits[0].score < service.min_score:
            false_refusals += 1

    refused = sum(
        1
        for item in oos
        if not (hits := service.retrieve(item.question, k=1)) or hits[0].score < service.min_score
    )
    n = max(len(answerable), 1)
    return EvalReport(
        label=label,
        n_answerable=len(answerable),
        n_out_of_scope=len(oos),
        hit_rate={k: hits_at[k] / n for k in ks},
        mrr=sum(reciprocal_ranks) / n,
        refusal_accuracy=refused / len(oos) if oos else 1.0,
        false_refusals=false_refusals,
        chunks_indexed=service.store.count(),
        misses=misses,
    )


def to_markdown(reports: Sequence[EvalReport], ks: Sequence[int]) -> str:
    head = ["Config", "Chunks", *[f"Hit@{k}" for k in ks], f"MRR@{max(ks)}", "OOS refused"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in reports:
        row = [
            r.label,
            str(r.chunks_indexed),
            *[f"{r.hit_rate[k]:.3f}" for k in ks],
            f"{r.mrr:.3f}",
            f"{r.refusal_accuracy * r.n_out_of_scope:.0f}/{r.n_out_of_scope}",
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _run(
    settings: Settings, samples: Path, items: list[QAItem], ks: list[int], label: str
) -> EvalReport:
    service = build_service(settings)
    ingest_directory(service, samples)
    return evaluate(service, items, ks, label)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--qa", type=Path, default=DEFAULT_QA)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--sweep", action="store_true", help="compare chunk-size presets")
    parser.add_argument("--json", action="store_true", help="print JSON instead of Markdown")
    args = parser.parse_args(argv)

    items = load_qa(args.qa)
    ks = sorted(set(args.k))
    base = Settings(vector_store="memory")  # eval always runs on a fresh in-memory index
    reports: list[EvalReport] = []
    if args.sweep:
        for size, overlap in [(300, 50), (500, 80), (700, 80), (1000, 150), (1500, 200)]:
            cfg = ChunkingConfig(chunk_size=size, chunk_overlap=overlap)
            settings = base.model_copy(update={"chunking": dict.fromkeys(DocType, cfg)})
            label = f"size={size}/overlap={overlap}"
            reports.append(_run(settings, args.samples, items, ks, label))
    else:
        label = f"{base.embedding_provider} + {base.vector_store} (per-type chunking)"
        reports.append(_run(base, args.samples, items, ks, label))

    if args.json:
        print(json.dumps([asdict(r) for r in reports], indent=2))
    else:
        print(to_markdown(reports, ks))
        for r in reports:
            if r.misses:
                print(f"\n[{r.label}] misses@{max(ks)}: {', '.join(r.misses)}", file=sys.stderr)
    return 0


__all__ = ["Chunker", "EvalReport", "QAItem", "evaluate", "load_qa", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
