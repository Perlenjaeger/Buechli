#!/usr/bin/env python
import os
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_postgres.vectorstores import PGVector
from dotenv import load_dotenv
from util.utils import get_pdf_paths
from typing import List

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def load_docs() -> List[Document]:
    pdf_paths = get_pdf_paths()
    documents: List[Document] = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(file_path=pdf_path)
        documents.extend(loader.load())
    return documents


def split_docs_to_texts(documents: List[Document]) -> List[Document]:
    print("splitting...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=70, length_function=len
    )
    texts: list[Document] = text_splitter.split_documents(documents)
    print(f"created {len(texts)} chunks")
    # enrich each text bloc with an id -> an id is necessary for the pgvector store
    for index, text in enumerate(texts):
        text.metadata["id"] = index  # type: ignore
    return texts


# TODO replace, upsert, skip, create
# replace: if index is not empty delete it and create new one
# upsert: if index is not empty add new documents to it
# skip: if index is not empty skip it
# create: if index does not exist create it
def ingest_pinecone(texts: list[Document]) -> None:
    print(f"Adding {len(texts)} texts to Pinecone")
    try:
        index_name: str = os.getenv("PINECONE_INDEX_NAME", "")
        vectorstore = PineconeVectorStore.from_documents(
            texts, embeddings, index_name=index_name
        )
        print(f"Successfully created vectorstore: {vectorstore}")

    except Exception as e:
        print(f"Failed to create Pinecone vectorstore: {str(e)}")
        raise  # Re-raise the exception to see the full stack trace


def ingest_faiss(texts: list[Document]):
    print("Ingesting...")

    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(os.getenv("VECTOR_STORE_INDEX_NAME", "faiss_index"))

    print("finish")


def ingest_pgvector(texts: List[Document]):
    print("Ingesting...")

    connection = "postgresql+psycopg://" + str(
        os.getenv("PGVECTOR_URL")
    )  # langchain psycopg Uses psycopg3!
    collection_name = str(
        os.getenv("VECTOR_STORE_INDEX_NAME")
    )  # Name of the collection in the database

    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection,
        use_jsonb=True,
    )

    vectorstore.add_documents(texts, ids=[text.metadata["id"] for text in texts])  # type: ignore

    print("finish")


def main():
    docs = load_docs()
    texts = split_docs_to_texts(docs)
    if os.getenv("VECTOR_STORE") == "pinecone":
        ingest_pinecone(texts)
    elif os.getenv("VECTOR_STORE") == "faiss":
        ingest_faiss(texts)
    elif os.getenv("VECTOR_STORE") == "pgvector":
        ingest_pgvector(texts)
    else:
        raise ValueError(
            "No valid vector store specified in environment variables. Please set VECTOR_STORE to 'pinecone', 'faiss', or 'pgvector'."
        )


if __name__ == "__main__":
    main()
