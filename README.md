# NASA Bioscience Knowledge Engine

The NASA Bioscience Knowledge Engine is a sophisticated RAG (Retrieval-Augmented Generation) system designed to provide intelligent answers from a vast collection of NASA bioscience research papers. It combines a powerful AI backend for processing and a modern web frontend for user interaction.

## Table of Contents

- Features
- Architecture
- Tech Stack
- Project Structure
- Setup and Installation
- Configuration
- Running the Application
- Usage
- API Endpoints
- Documentation
- Testing
- Contributing
- License

## Features

### AI Engine (Backend)
- **Developed AI Engine**: The core AI engine is fully developed and functional, providing the intelligence for the knowledge engine.
- **Optimized RAG Pipeline**: Efficiently retrieves and synthesizes information to answer complex queries.
- **Advanced Document Retrieval**: Uses FAISS with `mpnet` embeddings for fast and accurate initial document retrieval.
- **Contriever Reranking**: Employs a `BAAI/bge-reranker-base` model to refine search results for higher relevance.
- **Hybrid Answer Generation**:
    - **Summarization**: Leverages `google/flan-t5-large` for abstractive summarization, with an extractive fallback.
    - **Abstractive**: Utilizes powerful LLMs like GPT-4o-mini or a local Mistral-based model for generating human-like, citation-aware answers.
    - **Extractive Fallback**: Provides direct, grounded snippets from source documents when generative models are unavailable or fail.
- **GPU Acceleration**: Supports CUDA for accelerated model inference, with a fallback to CPU.
- **Citation-Aware**: All generated answers include inline citations, linking back to the source documents.

### React App (Frontend)
- The frontend is currently under development. It is intended to provide a modern, responsive user interface for interacting with the knowledge engine.

## Architecture

The application is a monorepo composed of two main components:

1.  **`AI_engine` (Backend)**: This is the core of the project, a Python-based service that houses the fully developed AI engine and exposes an API for its RAG pipeline. It handles:
    - Embedding and indexing of research papers.
    - Receiving user queries.
    - Executing the RAG pipeline (retrieve, rerank, generate).
    - Returning a structured response with the answer and sources.

2.  **`frontend`**: A React single-page application (SPA) that provides the user interface. It:
    - Sends user queries to the backend API.
    - Is currently under development to display the generated answer and the source documents in a user-friendly format.

## Tech Stack

| Component | Technology |
|---|---|
| **AI Backend** | Python, PyTorch, Hugging Face Transformers, Sentence-Transformers, FAISS, NumPy |
| **Frontend** | React, JavaScript/TypeScript, HTML/CSS (Under Development) |
| **Models** | `all-mpnet-base-v2` (Embeddings), `BAAI/bge-reranker-base` (Reranking), `google/flan-t5-large` (Summarization), GPT-4o-mini / `mistralai/Mistral-7B-Instruct-v0.2` (Generation) |

## Project Structure

```
NASA-Bioscience-Knowledge-Engine/
├── AI_engine/
│   ├── modules/
│   │   ├── rag_engine.py       # Core RAG pipeline logic
│   │   ├── embedding_store.py  # FAISS index and chunk management
│   │   └── ...
│   ├── utils/
│   ├── config.py             # Backend configuration
│   └── requirements.txt      # Python dependencies
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── README.md             # Frontend-specific README
└── README.md                 # Main project README (this file)
```

## Setup and Installation

### Prerequisites
- Python 3.8+
- Node.js and npm
- (Optional) NVIDIA GPU with CUDA drivers for hardware acceleration.

### Backend (`AI_engine`)

1.  **Navigate to the AI engine directory:**
    ```bash
    cd AI_engine
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Frontend

1.  **Navigate to the frontend directory:**
    ```bash
    cd ../frontend
    ```

2.  **Install Node.js dependencies:**
    ```bash
    npm install
    ```

## Configuration

The backend requires certain environment variables. Create a `.env` file in the `AI_engine` directory and add the following:

```env
# To use a local LLM from Hugging Face
USE_LLM=True
OPENAI_MODEL="mistralai/Mistral-7B-Instruct-v0.2" # Or another compatible model
HUGGING_FACE_TOKEN="your_hf_token_here"

# To use OpenAI's API (ensure USE_LLM is False or unset)
# OPENAI_API_KEY="your_openai_key_here"

# Set device for Hugging Face models ('cuda', 'cpu', or 'auto')
HF_DEVICE=auto
```

## Running the Application

1.  **Start the Backend Server:**
    - In your terminal, from the `AI_engine` directory (with the virtual environment activated), start the application server (e.g., using Flask or FastAPI).
    ```bash
    # Example with a hypothetical run.py using Flask
    python run.py
    ```

2.  **Start the Frontend Development Server:**
    - In a new terminal, from the `frontend` directory:
    ```bash
    npm start
    ```
    - Open http://localhost:3000 in your browser to view the application.

## Usage

Once both the backend and frontend are running, you can use the web interface at `http://localhost:3000` to interact with the knowledge engine.

1.  Enter a question about NASA bioscience research into the search box.
2.  Press Enter or click the "Ask" button.
3.  The system will process your query, retrieve relevant documents, and generate a concise, citation-backed answer.
4.  The answer will be displayed, along with the source documents used to generate it.

### Command-Line Interface (CLI)

You can also test the AI engine directly from the command line. Navigate to the `AI_engine` directory and run the `rag_engine.py` script:

```bash
cd AI_engine
python -m modules.rag_engine
```

This will prompt you to enter a query. The script will then execute the full RAG pipeline and print the generated answer and a list of the top-ranked source documents.

## API Endpoints

The backend exposes the following API endpoint for the frontend to consume.

### `POST /api/query`

Submits a query to the RAG pipeline.

-   **Request Body:**
    ```json
    {
      "query": "What are the effects of microgravity on the human cardiovascular system?"
    }
    ```
-   **Success Response (200):**
    ```json
    {
      "query": "What are the effects of microgravity on the human cardiovascular system?",
      "answer": "Microgravity exposure leads to cardiovascular deconditioning, including a reduction in heart size and changes in arterial pressure [1]. These effects are a key concern for long-duration spaceflight [2].",
      "retrieved_chunks": [
        {
          "chunk": "...",
          "title": "Cardiovascular Deconditioning in Space",
          "paper_id": "12345",
          "score": 0.89
        },
        {
          "chunk": "...",
          "title": "Arterial Pressure Regulation in Microgravity",
          "paper_id": "67890",
          "score": 0.85
        }
      ]
    }
    ```

## Example Output

An example of a report generated by the AI engine can be found here. This PDF demonstrates the kind of output you can expect from a query, including the RAG-generated answer, a summary of the context, and source citations.

*   [Sample Report (PDF)](AI_engine/data/exports/nasa_engine_results_20251005_162709.pdf)

## Testing

To run tests for the AI engine, you can use `pytest`:

```bash
cd AI_engine
pytest
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any bugs, feature requests, or improvements.

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some feature'`).
5.  Push to the branch (`git push origin feature/your-feature-name`).
6.  Open a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
