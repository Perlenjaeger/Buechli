# Buechli Chat

A LangChain-based chat application that allows you to have conversations with PDF documents using various vector stores and LLMs.

## Features
- Chat with PDF documents using different LLMs
- Support for multiple vector stores (Pinecone, FAISS, PGVector)
- PDF text highlighting for citations
- Structured responses with source citations

## Prerequisites
- Python 3.11+
- Docker (for PGVector)
- pip
- VS Code or Cursor (recommended for development)
- Git
- Graphviz, for Ubuntu/Debian:
    ```bash
    sudo apt-get install python3-dev graphviz graphviz-dev
    ```

## Installation

1. **Install system dependencies (see Prerequisites)**

2. **Clone the repository**
```bash
git clone <repository-url>
cd buechli-chat
```

### Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

#### Check if the virtual environment is activated
```bash
which python
```
Output should point to the `XXX/Buechli/.venv/bin/python` directory.

### Install the package in development mode
```bash
pip install -e .
```

The project uses `pyproject.toml` for dependency management. Key dependencies include:
- langchain and related packages for LLM interactions
- Vector store backends (FAISS, Pinecone, PGVector)
- PDF processing tools (PyMuPDF)
- Graph visualization (pygraphviz)
- Development tools (black, ruff)

Install development dependencies with:
```bash
pip install -e .[dev]
```

4. **Set up environment variables**
Copy `env.example` to `.env` and fill in your values:
```bash
cp env.example .env
```

1. **Start the PGVector database** (if using PGVector)
```bash
docker compose up -d
```

1. **Place PDF Documents**
```bash
mkdir -p data
# Copy your PDF files into the data/ directory
```

## Usage

1. **Ingest documents**
```bash
buechli-ingest
```

2. **Start the chat web interface**
```bash
streamlit run src/buechli_chat/app.py
# or
buechli-web
```

Or use the CLI chat:
```bash
buechli-chat
```

### Web Interface Features
- Modern Streamlit UI with custom styling
- Chat history with question-answer pairs
- Source citations with page references
- Custom LangChain Parrot avatar
- Latest answers appear at the top
- Expandable source sections

### Project Structure
```
.
├── data/                  # PDF documents
├── src/
│   ├── buechli_chat/     
│   │   ├── app.py        # Streamlit web interface
│   │   ├── buechli_graph.py  # LangChain graph implementation
│   │   ├── types.py      # Shared type definitions
│   │   └── web.py        # Web entry point
│   ├── ingest/           # Document ingestion code
│   └── util/             # Utility functions
├── compose.yaml          # Docker compose for PGVector
├── pyproject.toml        # Project dependencies
└── .env                  # Environment variables
```

### Architecture
- Separation of concerns between UI (`app.py`) and logic (`buechli_graph.py`)
- Shared type definitions in `types.py`
- LangChain graph-based implementation for document retrieval and response generation
- Support for multiple vector stores (Pinecone, FAISS, PGVector)

## Development

### VS Code Setup

1. **Install Recommended Extensions**
The project includes recommended VS Code extensions in `.vscode/extensions.json`:
- Black Formatter for Python code formatting
- Ruff for linting
- Run on Save for automatic dependency installation

2. **Editor Settings**
The project includes preconfigured settings in `.vscode/settings.json`:
- Automatic code formatting on save
- Strict type checking
- Ruff linting configuration
- Custom run-on-save commands for dependency management

3. **Debug Configuration**
The project includes a preconfigured debug setup in `.vscode/launch.json` that:
- Sets the correct working directory
- Loads environment variables from `.env`
- Enables Python inline values during debugging

To start debugging:
- Open any Python file
- Set breakpoints
- Press F5 or use the Run and Debug sidebar

for debugging the chat:
- Open `src/buechli_chat/app.py`
- Set breakpoints
- use the streamlit debug configuration
- Press F5 or use the Run and Debug sidebar

for debugging without the llm calls:
- take a function from `src/buechli_chat/buechli_graph.py` and add the `@debug_stub_store` decorator
- the decorator stores the function return value on the disk and on next call reads the return value from the disk

### Automatic Dependency Installation

The project uses the Run on Save extension to automatically install dependencies when `pyproject.toml` is modified. This ensures your development environment stays in sync with project requirements.

### Code Quality

- **Formatting**: Black formatter runs automatically on save
- **Linting**: Ruff provides real-time code analysis
- **Type Checking**: Strict type checking is enabled by default

### Troubleshooting

Common issues and solutions:

1. **Run on Save not working**
   - Ensure you're using a workspace
   - Check VS Code extension is installed
   - Verify `.vscode/settings.json` is present

2. **Vector Store Connection Issues**
   - For PGVector: Ensure Docker is running
   - For Pinecone: Verify API key and environment
   - For FAISS: Check index file permissions
