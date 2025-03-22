#!/usr/bin/env python
# %%
from typing import List

from pymupdf import Document as MuDocument, open # type: ignore
import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_PATH = os.getenv("PROJECT_PATH", None)
print(f"PROJECT_PATH: {PROJECT_PATH}")


def get_project_path(relative_path: str = ".") -> str:
    if PROJECT_PATH is None:
        raise ValueError("PROJECT_PATH is not set")
    return os.path.join(PROJECT_PATH, relative_path)

def get_pdf_paths() -> List[str]:
    pdf_paths = os.getenv("PDF_PATHS")
    if not pdf_paths:
        raise ValueError("PDF_PATHS not set in environment variables")
    return [get_project_path(path.strip()) for path in pdf_paths.split(",")]


def load_pdf(doc: str) -> MuDocument:
    return open(doc)


def save_pdf(pdf_document: MuDocument, path: str) -> None:
    new_doc_path = path.rsplit(".pdf", 1)[0] + "_highlighted.pdf"
    os.makedirs(
        os.path.dirname(new_doc_path), exist_ok=True
    )  # Ensure the directory exists
    pdf_document.save(new_doc_path)  # type: ignore
    pdf_document.close()  # type: ignore
