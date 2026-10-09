"""FastAPI application: HTTP concerns only, all logic lives in RAGService."""

from __future__ import annotations

import json
import logging
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

from docsearch import __version__
from docsearch.api.schemas import (
    DocumentDeleted,
    DocumentIngested,
    DocumentSummary,
    Health,
    QueryRequest,
    QueryResponse,
    Source,
)
from docsearch.config import Settings, get_settings
from docsearch.domain import Metadata
from docsearch.errors import (
    DocumentNotFoundError,
    EmptyDocumentError,
    ProviderError,
    UnsupportedDocumentError,
)
from docsearch.rag.service import RAGService

logger = logging.getLogger(__name__)


def get_service(request: Request) -> RAGService:
    svc: RAGService = request.app.state.service
    return svc


# Module-level so FastAPI can resolve it from string annotations (PEP 563).
ServiceDep = Annotated[RAGService, Depends(get_service)]


def create_app(service: RAGService | None = None, settings: Settings | None = None) -> FastAPI:
    """App factory. Tests inject a service wired with fakes; production builds from env."""
    settings = settings or get_settings()
    if service is None:
        from docsearch.container import build_service

        service = build_service(settings)

    app = FastAPI(
        title="DocSearch AI - RAG API",
        version=__version__,
        description="Upload documents, then ask grounded questions with cited sources.",
    )
    app.state.service = service
    app.state.settings = settings

    @app.exception_handler(UnsupportedDocumentError)
    async def _unsupported(_: Request, exc: UnsupportedDocumentError) -> JSONResponse:
        return JSONResponse(status_code=415, content={"detail": str(exc)})

    @app.exception_handler(EmptyDocumentError)
    async def _empty(_: Request, exc: EmptyDocumentError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(DocumentNotFoundError)
    async def _not_found(_: Request, exc: DocumentNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(ProviderError)
    async def _provider(_: Request, exc: ProviderError) -> JSONResponse:
        logger.exception("provider failure")
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.get("/health", response_model=Health, tags=["ops"])
    def health(svc: ServiceDep) -> Health:
        docs = svc.store.list_documents()
        return Health(
            status="ok",
            version=__version__,
            vector_store=settings.vector_store,
            embedding_provider=settings.embedding_provider,
            llm_provider=svc.generator.name,
            documents=len(docs),
            chunks=sum(docs.values()),
        )

    @app.get("/documents", response_model=list[DocumentSummary], tags=["documents"])
    def list_documents(svc: ServiceDep) -> list[DocumentSummary]:
        return [
            DocumentSummary(doc_id=d, chunks=n)
            for d, n in sorted(svc.store.list_documents().items())
        ]

    @app.post(
        "/documents",
        response_model=DocumentIngested,
        status_code=status.HTTP_201_CREATED,
        tags=["documents"],
    )
    async def upload_document(
        svc: ServiceDep,
        file: Annotated[UploadFile, File(description="PDF, Markdown or plain-text file")],
        doc_id: Annotated[str | None, Form(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,127}$")] = None,
        metadata: Annotated[str | None, Form(description="JSON object of metadata")] = None,
    ) -> DocumentIngested:
        data = await file.read()
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail="File too large.")
        result = svc.ingest(
            data,
            file.filename or "upload.txt",
            content_type=file.content_type,
            doc_id=doc_id,
            metadata=_parse_metadata(metadata),
        )
        return DocumentIngested(
            doc_id=result.doc_id,
            source=result.source,
            doc_type=result.doc_type.value,
            chunks=result.chunks,
            replaced_chunks=result.replaced_chunks,
        )

    @app.delete("/documents/{doc_id}", response_model=DocumentDeleted, tags=["documents"])
    def delete_document(doc_id: str, svc: ServiceDep) -> DocumentDeleted:
        return DocumentDeleted(doc_id=doc_id, deleted_chunks=svc.delete(doc_id))

    @app.post("/query", response_model=QueryResponse, tags=["query"])
    def query(body: QueryRequest, svc: ServiceDep) -> QueryResponse:
        result = svc.answer(body.question, k=body.top_k, filters=body.filters)
        cited = set(result.cited)
        return QueryResponse(
            answer=result.answer,
            grounded=result.grounded,
            citations=result.cited,
            sources=[
                Source(
                    id=sc.chunk.chunk_id,
                    doc_id=sc.chunk.doc_id,
                    chunk_index=sc.chunk.chunk_index,
                    source=str(sc.chunk.metadata.get("source", sc.chunk.doc_id)),
                    score=round(sc.score, 4),
                    cited=sc.chunk.chunk_id in cited,
                    text=sc.chunk.text,
                )
                for sc in result.sources
            ],
            generator=result.extra.get("generator"),
            latency_ms=float(result.extra.get("latency_ms", 0.0)),
        )

    return app


def _parse_metadata(raw: str | None) -> Metadata:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"metadata is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict) or not all(
        isinstance(v, str | int | float | bool) for v in parsed.values()
    ):
        raise HTTPException(
            status_code=422, detail="metadata must be a flat JSON object of scalar values"
        )
    reserved = {"doc_id", "source", "doc_type", "chunk_index"} & parsed.keys()
    if reserved:
        raise HTTPException(status_code=422, detail=f"reserved metadata keys: {sorted(reserved)}")
    return parsed
