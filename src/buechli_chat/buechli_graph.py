#!/usr/bin/env python
# %%
import re
from io import TextIOWrapper
from pydantic import BaseModel, Field
import os
import string
from typing import Any, Callable, List, TypedDict, Dict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


from dotenv import load_dotenv, find_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from IPython.display import Image, display  # type: ignore
from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import FAISS
from langchain_postgres.vectorstores import PGVector

from util.utils import get_pdf_paths, load_pdf, save_pdf
from buechli_chat.types import ChatMessage, LLMResult, LangchainResult, RunLLM, Annotation
from numpy import uint
from pymupdf import Page, Document as MuDocument # type: ignore

from util.debug_cache import debug_stub_store


load_dotenv()


class Citation(BaseModel):
    source_id: int = Field(
        ...,
        description="The integer ID of a SPECIFIC source which justifies the answer.",
    )
    quote: str = Field(
        ...,
        description="The VERBATIM quote from the specified source that justifies the answer.",
    )


class QuotedAnswer(BaseModel):
    """Answer the user question based only on the given sources, and cite the sources used."""

    answer: str = Field(
        ...,
        description="The answer to the user question, which is based only on the given sources.",
    )
    citations: List[Citation] = Field(
        ..., description="Citations from the given sources that justify the answer."
    )


class State(TypedDict):
    question: str
    context: List[Document]
    answer: QuotedAnswer


def format_docs_with_id(docs: List[Document]) -> str:
    formatted = [
        f"Source ID: {i}\nArticle Page: {doc.metadata['page_label']}\nArticle Snippet: {doc.page_content}"  # type: ignore
        for i, doc in enumerate(docs)
    ]
    return "\n\n" + "\n\n".join(formatted)


def get_vectorestore():
    vectorstore: PineconeVectorStore | FAISS | PGVector | None = (
        None  # todo evtl abstraction drüber?
    )
    if os.getenv("VECTOR_STORE") == "pinecone":
        vectorstore = PineconeVectorStore(
            index_name=os.getenv("PINECONE_INDEX_NAME", ""),
            embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        )
    elif os.getenv("VECTOR_STORE") == "faiss":
        vectorstore = FAISS.load_local(
            os.getenv("VECTOR_STORE_INDEX_NAME", "faiss_index"),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    elif os.getenv("VECTOR_STORE") == "pgvector":
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
    else:
        raise Exception("No vector store found")

    return vectorstore


def retrieve(state: State):
    # Initialize vector store
    vectorstore = get_vectorestore()
    # Search for relevant documents
    retrieved_docs = vectorstore.similarity_search(state["question"], k=10)  # type: ignore
    return {"context": retrieved_docs}


def generate(state: State):  # type: ignore
    formatted_docs = format_docs_with_id(state["context"])
    messages = prompt.invoke(  # type: ignore
        {
            "question": state["question"],
            "context": formatted_docs,
            "min_citations": MIN_CITATIONS,
            "max_citations": MAX_CITATIONS,
        }
    )
    structured_llm = llm.with_structured_output(QuotedAnswer)  # type: ignore
    response = structured_llm.invoke(messages)  # type: ignore
    return {"answer": response}  # type: ignore


MIN_CITATIONS = int(os.getenv("MIN_CITATIONS", "2"))
MAX_CITATIONS = int(os.getenv("MAX_CITATIONS", "4"))


system_prompt = (
    "You're a helpful AI assistant. Given a user question "
    "and some article snippets, answer the user "
    "question with EXACTLY {min_citations} to {max_citations} citations."
    "If none of the snippets answer the question, just say EXACTLY: UNKNOWN"
    "\n\nHere are the article snippets: "
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(  # type: ignore
    [
        ("system", system_prompt),
        ("human", "{question}"),
    ]
)

llm = ChatOpenAI(
    model="gpt-4o-mini-2024-07-18",
    verbose=True,
    temperature=0,
    max_completion_tokens=1000,
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " "],
    keep_separator=False,
)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


# Define a generic type alias for chain creation functions
ChainCreator = Callable[..., Any]


@debug_stub_store
def _run_llm_impl(query: str, chat_history: List[ChatMessage]) -> LLMResult:
    # Run the graph with the query
    result = get_graph().invoke({"question": query})

    new_result: LLMResult = {
        "query": query,
        "result": result["answer"].answer,
        "citations": result["answer"].citations,
        "source_documents": result["context"],
    }
    return new_result


# Export the typed function
run_llm: RunLLM = _run_llm_impl


def get_graph() -> "CompiledStateGraph":
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])  # type: ignore
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()  # type: ignore
    return graph


def run_llm_operations(graph: "CompiledStateGraph", question: str) -> LangchainResult:
    result: LangchainResult = graph.invoke({"question": question})
    return result


def tee_print_write(file: TextIOWrapper, text: str):
    print(text)
    file.write(text + "\n")


def print_and_store_results(
    graph: "CompiledStateGraph", result: LangchainResult, output_file: str = "output.md"
):
    with open(output_file, "w") as file:
        tee_print_write(file, "# LangGraph Result Output")
        tee_print_write(file, "\n## Graph\n")

        # Skip graph visualization if pygraphviz is not available
        try:
            display(Image(graph.get_graph().draw_mermaid_png()))
            with open("graph.png", "wb") as f:
                f.write(graph.get_graph().draw_png())  # type: ignore
            file.write('![Execution Graph](graph.png "Execution Graph")\n')
        except ImportError:
            tee_print_write(
                file, "Graph visualization skipped - pygraphviz not available\n"
            )

        tee_print_write(file, "\n## Question\n")
        tee_print_write(file, result["question"])

        tee_print_write(file, "\n## Answer\n")
        tee_print_write(file, str(result["answer"].answer))

        tee_print_write(file, "\n### Citations\n")

        pdf_paths = get_pdf_paths()
        pdf_documents: Dict[str, MuDocument] = {pdf_path: load_pdf(pdf_path) for pdf_path in pdf_paths}
        for index, citation in enumerate(result["answer"].citations):
            tee_print_write(file, f"\n#### Quote {index}\n")
            tee_print_write(file, citation.quote)
            source_path: str = result["context"][citation.source_id].metadata["source"]
            tee_print_write(
                file,
                f"\n##### From: Source ID {citation.source_id} Page: {result['context'][citation.source_id].metadata['page_label']} Source: {source_path}\n",
            )
            tee_print_write(file, "\n#### Citation block context\n")
            tee_print_write(file, result["context"][citation.source_id].page_content)

            try:
                pdf_document: MuDocument = pdf_documents[source_path]
                pdf_document = highlight_text_in_pdf(
                    pdf_document,
                    citation.quote,
                    result["context"][citation.source_id].metadata["page"],
                )
                pdf_documents[source_path] = pdf_document
            except Exception as e:
                tee_print_write(file, f"\nError: {e}\n")

        for pdf_path, pdf_document in pdf_documents.items():
            save_pdf(pdf_document, pdf_path)

        print("\n# finish")

def highlight_text_in_pdf(
    pdf_document: MuDocument, text: str, page_number_zero_based: uint
) -> MuDocument:
    """Highlight the text in the pdf file
    uses pymupdf to highlight in yellow text segments given by the text on page_number
    """
# document.search_page_for
    # pdf_document.search_page_for(page_number, search_text)
    # remove punctuation from text
    # pdf_document.get_text_blocks
    page: Page = pdf_document.load_page(page_number_zero_based)  # type: ignore
    rectangles = highlight_rectangles_4_pdf(pdf_document, text, page_number_zero_based)
    for rectangular in rectangles:  # type: ignore
        page.add_highlight_annot(rectangular)  # type: ignore

    return pdf_document


def highlight_rectangles_4_pdf(
    pdf_document: MuDocument, text: str, page_number_zero_based: uint
) -> List[Any]:
    """Highlight the text in the pdf file
    uses pymupdf to highlight in yellow text segments given by the text on page_number
    """
    page: Page = pdf_document.load_page(page_number_zero_based)  # type: ignore

    # "([A-z]*-[A-z]*)" splits before Word-Word and after
    # "[.,;:!?]" splits by punctuation and removes the punctuations
    needle_parts = [
        punctuation
        for minus_concatenation in re.split(r"([A-z]*-[A-z]*)", text)
    for punctuation in re.split(r"[.;:!?]", minus_concatenation)
    ]
    rectangles: List[Any] = []
    for part in needle_parts:
        if not part:
            continue
        part = part.lstrip(string.whitespace).rstrip(string.whitespace)
        rectangle = page.search_for(part)  # type: ignore
        if rectangle:
            rectangles.extend(rectangle)  # type: ignore
        else:
            print(f"\nText '{part}' not found on page {page_number_zero_based}\n")
    return rectangles


def annotations_4_pdf(
    pdf_document: MuDocument, text: str, page_number_zero_based: uint
) -> List[Annotation]:
    """Highlight the text in the pdf file
    uses pymupdf to highlight in yellow text segments given by the text on page_number
    """
    annotations: List[Annotation] = []
    rectangles = highlight_rectangles_4_pdf(pdf_document, text, page_number_zero_based)
    for rectangle in rectangles:
        annotations.append(
            {
                "page": str(page_number_zero_based + 1),
                "x": str(rectangle[0]),
                "y": str(rectangle[1]),
                "width": str(rectangle[2] - rectangle[0]),
                "height": str(rectangle[3] - rectangle[1]),
                "color": "rgba(255, 0, 0, 1.0)",
                "type": "title",
            }
        )
    return annotations


def chat(question: str):
    print("hi")
    print(str(os.getenv("OPENAI_API_KEY"))[:5])
    print(f"PINECONE_INDEX_NAME: {os.getenv('PINECONE_INDEX_NAME')}")

    # Debug: Print which .env file is being loaded
    print(f"Loading .env from: {find_dotenv()}")

    graph = get_graph()
    result = run_llm_operations(graph, question)
    print_and_store_results(graph, result)


# %%
def main():
    res: LLMResult = run_llm(
        query="Give me the gist of ReAct in exactly 3 sentences", chat_history=[]
    )
    result: str = res.get("result", "No answer found")
    print(result)

    chat("Give me the gist of ReAct in exactly 3 sentences")


# %%
if __name__ == "__main__":
    main()
