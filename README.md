# Research Paper RAG Assistant

An end-to-end deployed Retrieval-Augmented Generation (RAG) application for question answering over research papers.

The system combines PDF processing, semantic embeddings, FAISS vector search, Cross-Encoder reranking, and a local FLAN-T5 language model. It is exposed through a FastAPI REST API, containerized with Docker, deployed on AWS EC2, and automated with GitHub Actions CI/CD.

## Live Demo

- **API:** http://15.252.13.92:8000
- **Swagger UI:** http://15.252.13.92:8000/docs

Open Swagger UI, expand `POST /ask`, click **Try it out**, and submit a question about *Attention Is All You Need*.

## Project Overview

The application processes a research paper, converts its content into searchable chunks, generates vector embeddings, stores them in a FAISS index, retrieves relevant passages for a user query, reranks them, and generates an answer from the most relevant context.

## Architecture

```text
Research Paper PDF
        |
        v
PDF Text Extraction (PyMuPDF)
        |
        v
Sentence-Aware Chunking (NLTK)
        |
        v
Embeddings (all-MiniLM-L6-v2)
        |
        v
FAISS Vector Store (IndexFlatIP)
        |
        v
User Question -> Query Embedding -> FAISS Top 10
        |
        v
Cross-Encoder Reranking -> Top 3
        |
        v
FLAN-T5
        |
        v
Grounded Answer + Sources
```

## Key Features

- PDF text extraction using PyMuPDF with page-level source preservation
- Sentence-aware chunking (~500 characters) with sentence overlap
- Semantic embeddings using Sentence Transformers
- FAISS vector search with cosine similarity using normalized vectors
- Two-stage retrieval: FAISS top 10, Cross-Encoder reranking to top 3
- Local answer generation using FLAN-T5 without an external LLM API
- Source page, similarity, and rerank score in API responses
- FastAPI REST API and Docker containerization
- AWS ECR and EC2 deployment with GitHub Actions

## Technology Stack

| Category | Technology |
|---|---|
| Language | Python |
| PDF Processing | PyMuPDF |
| Text Processing | NLTK |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector Search | FAISS |
| Reranking | Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) |
| LLM | Google FLAN-T5 (`flan-t5-base`) |
| API | FastAPI, Uvicorn |
| Containerization | Docker |
| Cloud | AWS EC2, AWS ECR |
| CI/CD | GitHub Actions |

## Project Structure

```text
research-paper-rag/
├── data/
│   ├── papers/attention_is_all_you_need.pdf
│   └── index/ (faiss_index.bin, chunks.pkl)
├── src/
│   ├── ingestion.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── storage.py
│   ├── retrieval.py
│   ├── reranker.py
│   ├── llm.py
│   ├── pipeline.py
│   └── build_index.py
├── tests/
├── app.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## RAG Pipeline

1. **Ingestion:** PyMuPDF extracts text per page, keeping document name and page number.
2. **Chunking:** NLTK splits text into sentences, grouped into ~500-character chunks with sentence overlap. Page information is preserved on every chunk.
3. **Embeddings:** Each chunk becomes a 384-dimensional vector using `all-MiniLM-L6-v2`.
4. **Vector storage:** Normalized embeddings are stored in a FAISS `IndexFlatIP` index. Because the vectors are normalized, inner product corresponds to cosine similarity.
5. **Retrieval:** The question is embedded and FAISS returns the top 10 chunks.
6. **Reranking:** `cross-encoder/ms-marco-MiniLM-L-6-v2` scores each question-passage pair and the top 3 are kept.
7. **Generation:** `google/flan-t5-base` answers using only the retrieved context. If the answer is not in the context, it states that the answer is not available in the provided paper.

## Local Setup

```bash
git clone https://github.com/Vineetyadav07-42/research-paper-rag.git
cd research-paper-rag

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt_tab')"
```

### Build the Vector Index

```bash
python -m src.build_index
```

Creates:

```text
data/index/faiss_index.bin
data/index/chunks.pkl
```

### Run the API

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

API:

```text
http://localhost:8000
```

Swagger:

```text
http://localhost:8000/docs
```

## API Usage

### Health Check

```text
GET /
```

### Ask a Question

```text
POST /ask
```

Request:

```json
{
    "question": "What is the purpose of self-attention?"
}
```

### Try with cURL

```bash
curl -X POST http://15.252.13.92:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is multi-head attention?"}'
```

## Docker

Build the image:

```bash
docker build -t research-paper-rag .
```

Run the container:

```bash
docker run -d \
  --name research-paper-rag \
  -p 8000:8000 \
  --restart unless-stopped \
  research-paper-rag
```

Access the API at:

```text
http://localhost:8000/docs
```

## AWS Deployment

```text
GitHub -> GitHub Actions -> Docker Build -> Amazon ECR -> Amazon EC2 -> FastAPI + Uvicorn
```

| Metric | Value |
|---|---|
| EC2 instance type | m7i-flex.large |
| Docker image size | ~3.3 GB |

## CI/CD

On every push to `main`, GitHub Actions:

1. Checks out the repository
2. Configures AWS credentials and logs in to Amazon ECR
3. Builds and pushes the Docker image
4. Connects to EC2 over SSH and pulls the latest image
5. Stops and removes the old container
6. Starts the new container

**Scope:** CI/CD currently builds and deploys only. Tests are run locally with `pytest` and are not yet part of the pipeline.

## Environment and Security

Credentials are stored as GitHub Actions secrets and are never committed to the repository:

```text
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
EC2_HOST
EC2_USERNAME
EC2_SSH_KEY
```

## Evaluation

Retrieval is checked with predefined questions and expected source pages:

- What is the purpose of self-attention?
- What is the Transformer architecture?
- What is multi-head attention?
- Why does the Transformer use positional encoding?
- How does the Transformer differ from recurrent neural networks?

**Result:** Correct source page in the top 3 for **4/5 questions**.

**Missed:** "Why does the Transformer use positional encoding?"

## Testing

```bash
pytest
```

Tests cover:

- PDF ingestion
- Chunking
- Embeddings
- Vector store
- Retrieval
- RAG pipeline

## Limitations

- Built around a single research paper; no document upload yet
- `flan-t5-base` is small, so answers can be short and may miss nuance on multi-part questions
- CPU-only inference on EC2 limits response speed
- Large Docker image due to ML dependencies and model files
- No authentication, monitoring, or structured logging
- Small evaluation set; no formal Recall@K or MRR evaluation yet

## Future Improvements

- Multi-document support and document upload
- Streamlit user interface
- Recall@K, Precision@K, and MRR for retrieval
- Correctness and faithfulness evaluation for generation
- Show source text alongside page numbers
- Run tests in CI before deployment
- Docker image and model optimization
- Latency monitoring and improved error handling/logging
- Larger instruction-tuned LLM

## Repository

https://github.com/Vineetyadav07-42/research-paper-rag

## Author

**Vineet Yadav** | [GitHub](https://github.com/Vineetyadav07-42)