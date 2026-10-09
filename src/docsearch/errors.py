"""Domain-level exceptions, translated to HTTP errors by the API layer."""


class DocSearchError(Exception):
    """Base class for all application errors."""


class UnsupportedDocumentError(DocSearchError):
    """The uploaded file type cannot be ingested."""


class EmptyDocumentError(DocSearchError):
    """No extractable text was found in the uploaded file."""


class DocumentNotFoundError(DocSearchError):
    """The requested document id is not indexed."""


class ProviderError(DocSearchError):
    """An upstream embedding / LLM / vector-store provider failed or is misconfigured."""
