from typing import Any, Dict, Protocol, List, cast
import streamlit as st
from dotenv import load_dotenv
from buechli_chat.types import ChatMessage, LLMResult, PdfDocument
from buechli_chat.buechli_graph import run_llm, annotations_4_pdf
from streamlit_pdf_viewer import pdf_viewer  # type: ignore
from util.utils import (
    get_project_path,
    load_pdf,
    get_pdf_paths,
)

load_dotenv()

# Configure the page style
st.set_page_config(page_title="Buechli Chat", layout="wide")

# Custom CSS for styling
st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)

st.markdown(
    """
<style>
    .st-emotion-cache-165fv6u{
        font-family: 'Orbitron', 'Segoe UI', sans-serif;
    }
    /* Main text and headers */
    .stMarkdown, .stText, .stTextInput, .stWidgetLabel, .stMarkdownContainer {
        font-family: 'Orbitron', 'Segoe UI', sans-serif;
        letter-spacing: 0.1rem;
        line-height: 1.6;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Orbitron', 'Segoe UI', sans-serif !important;
        color: #00f2c1 !important;
        text-transform: uppercase;
        letter-spacing: 0.15rem;
    }

    h1 {
        font-size: 3.5rem;
        line-height: 1.2;
        font-weight: 700;
    }

    h2 {
        font-size: 1.75rem;
        line-height: 1.2;
        margin-bottom: 1.5rem;
        font-weight: 500;
    }

    h3 {
        font-size: 1.375rem;
        line-height: 1.2;
        font-weight: 400;
    }
    
    /* Input fields */
    .stTextInput input {
        background-color: rgba(0, 242, 193, 0.1);
        border: 1px solid #00f2c1;
        color: #00f2c1;
        font-family: 'Orbitron', 'Segoe UI', sans-serif;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    
    /* Chat messages */
    .stChatMessage {
        background-color: rgba(0, 242, 193, 0.05);
        border: 1px solid #00f2c1;
        border-radius: 0.5rem;
        padding: 1rem;
    }
    
    /* Custom background */
    .stApp {
        background: linear-gradient(45deg, #000000, #001a1a);
    }
    
    /* Button styling */
    .stButton button {
        font-family: 'Orbitron', sans-serif;
        background-color: #00f2c1;
        color: black;
        border: none;
        border-radius: 1.25rem 0.5rem;
        padding: 1rem 2rem;
        min-width: 10rem;
        font-size: 1.2rem;
        line-height: 1.2;
        transition: all 0.1s ease;
        text-transform: uppercase;
        letter-spacing: 0.1rem;
    }

    .stButton button:hover {
        filter: drop-shadow(0px 0px 7px var(--color-primary));
        transform: translateY(-1px);
    }

    .stButton button:active {
        transform: translateY(1px);
    }

    /* Strong text */
    strong {
        color: #00f2c1;
        font-weight: 500;
    }

    /* Sidebar */
    .css-1d391kg {
        background-color: rgba(0, 0, 0, 0.8);
    }

    /* Progress bar styling */
    .stProgress > div > div > div {
        background-color: #005c4b !important;
        border-radius: 0.25rem;
        height: 0.5rem;
    }
    
    .stProgress > div > div {
        background-color: rgba(0, 92, 75, 0.2);
        border-radius: 0.25rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Add sidebar with parrot info
with st.sidebar:
    st.markdown("### LangChain Parrot")

    # Add parrot image using Streamlit's native image component
    st.image(
        get_project_path("assets/cyber-parrot-small.webp"),
        width=150,
    )


# Define protocol for streamlit chat message
class ChatMessageInterface(Protocol):
    def write(self, content: Any, unsafe_allow_html: bool = False) -> None: ...


# Initialize state and result
if "chat_history" not in st.session_state:
    chat_history: List[ChatMessage] = []
    st.session_state["chat_history"] = chat_history
    st.session_state["result"] = {
        "query": "",
        "result": "",
        "citations": [],
        "source_documents": [],
    }

# Get current result from session state
result: LLMResult = st.session_state["result"]

# Configure avatars
USER_AVATAR = "🧑‍💻"
ASSISTANT_AVATAR = get_project_path("assets/ls-parrot-tiny.png")

# Chat input at the top
# question = st.text_input("Ask a question about the paper:", key="question_input")

if "question_input" not in st.session_state:
    st.session_state.question_input = ""


def submit():
    st.session_state.question_input = st.session_state.widget
    st.session_state.widget = ""


st.text_input("Ask a question about the paper:", key="widget", on_change=submit)

question = st.session_state.question_input

# st.write(question_input)

if question:
    with st.spinner("Thinking..."):
        # Update result in session state
        st.session_state["result"] = run_llm(
            query=question, chat_history=st.session_state["chat_history"]
        )
        result = st.session_state["result"]

        # Add to chat history using list concatenation
        st.session_state["chat_history"] = st.session_state["chat_history"] + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": result},
        ]

        st.session_state["question_input"] = ""

# Group messages into QA pairs and display in reverse order
messages = st.session_state.chat_history
qa_pairs: List[tuple[ChatMessage, ChatMessage]] = list(
    zip(messages[::2], messages[1::2])
)  # Group into pairs

for qa_index, (user_msg, assistant_msg) in enumerate(reversed(qa_pairs)):
    # Display user question
    with st.chat_message(user_msg["role"], avatar=USER_AVATAR) as chat_msg:
        if chat_msg is not None:
            chat_msg_typed = cast(ChatMessageInterface, chat_msg)
            chat_msg_typed.write(user_msg["content"])
        else:
            st.write(user_msg["content"])  # type: ignore

    # Display assistant response
    with st.chat_message(assistant_msg["role"], avatar=ASSISTANT_AVATAR) as chat_msg:
        if chat_msg is not None:
            chat_msg_typed = cast(ChatMessageInterface, chat_msg)
            chat_msg_typed.write(str(assistant_msg["content"]["result"]))  # type: ignore
        else:
            st.write(assistant_msg["content"]["result"])  # type: ignore

        llm_result:LLMResult = assistant_msg["content"] # type: ignore

        # Show sources for this answer
        if "source_documents" in llm_result:
            with st.expander("View Sources"):
                for i, doc in enumerate(llm_result["source_documents"]):
                    if doc.metadata['title']:
                        st.markdown(
                            f"**Source {i+1}** (Page {doc.metadata['page_label']}) of \"{doc.metadata['title']}\":"
                        )
                    else:
                        st.markdown(
                            f"**Source {i+1}** (Page {doc.metadata['page_label']}):"
                        )
                    st.markdown(doc.page_content)

        # Show sources for this answer
        if "citations" in llm_result:
            pdf_paths = get_pdf_paths()
            with st.expander("View Citations", True):
                for i, citation in enumerate(llm_result["citations"]):
                    cite_page = int(
                        llm_result["source_documents"][citation.source_id].metadata[
                            "page_label"
                        ]
                    )
                    if llm_result['source_documents'][citation.source_id].metadata['title']:
                        st.markdown(
                            f"**Citation {i+1}** (Page {cite_page} of \"{llm_result['source_documents'][citation.source_id].metadata['title']}\":"
                        )
                    else:
                        st.markdown(
                            f"**Citation {i+1}** (Page {cite_page}):"
                        )
                    st.markdown(citation.quote)

            with st.expander("PDF"):
                pdf_documents: Dict[str, PdfDocument] = {}

                for cite in llm_result["citations"]:
                    source_path = llm_result["source_documents"][cite.source_id].metadata[
                        "source"
                    ]
                    if not pdf_documents.get(source_path):
                        try:
                            pdf_documents[source_path] = {
                                "document": load_pdf(source_path),
                                "annotation": [],
                            }
                        except Exception as e:
                            st.markdown(f"Error: {e}")
                            continue
                    document = pdf_documents[source_path]
                    document["annotation"] = document["annotation"] + annotations_4_pdf(  # type: ignore
                        document["document"],
                        cite.quote,
                        llm_result["source_documents"][cite.source_id].metadata["page"],
                    )
                    page_label = llm_result["source_documents"][cite.source_id].metadata[
                        "page_label"
                    ]
                    page = llm_result["source_documents"][cite.source_id].metadata["page"]

                for pdf_path, pdf_document in pdf_documents.items():
                    if pdf_document["annotation"]:
                        st.markdown(pdf_document["document"].metadata["title"])  # type: ignore
                        cite_pages = sorted(set([int(cite_page["page"]) for cite_page in pdf_document["annotation"]]))  # type: ignore
                        pdf_viewer(
                            input=pdf_path,
                            pages_to_render=cite_pages,
                            annotations=pdf_document["annotation"],
                            key=f"pdf_viewer_{qa_index}_{len(pdf_document['annotation'])}",  # Eindeutiger Key pro QA-Paar # type: ignore
                        )
                        st.markdown("---")
