# types.py
from typing import Any, List, TypedDict, Protocol, Union
from pymupdf import Document  # type: ignore


class ChatMessage(TypedDict):
    role: str
    content: str


class LLMResult(TypedDict):
    query: str
    result: str
    citations: List[Any]
    source_documents: List[Any]


# Verwende ein Protocol mit keyword-only Parametern:
class RunLLM(Protocol):
    def __call__(self, *, query: str, chat_history: List[ChatMessage]) -> LLMResult: ...

LangchainResult = Union[dict[str, Any], Any]


class Annotation(TypedDict):
    """Type definition for PDF annotations"""

    page: str
    x: str
    y: str
    width: str
    height: str
    color: str
    type: str


class PdfDocument(TypedDict):
    document: Document
    annotation: List[List[Annotation]]
