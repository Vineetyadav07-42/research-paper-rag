from pathlib import Path

from src.embeddings import create_embeddings
from src.vector_store import load_vector_store
from src.storage import load_chunks
from src.retrieval import retrieve_chunks
from src.reranker import rerank_chunks
from src.llm import generate_answer


BASE_DIR = Path(__file__).resolve().parent.parent

INDEX_DIR = BASE_DIR / "data" / "index"

INDEX_PATH = INDEX_DIR / "faiss_index.bin"
CHUNKS_PATH = INDEX_DIR / "chunks.pkl"


# Load the index and chunks once when the application starts
index = load_vector_store(INDEX_PATH)
chunks = load_chunks(CHUNKS_PATH)


def answer_question(query):

    # 1. Convert question into an embedding
    query_embedding = create_embeddings([query])

    # 2. Retrieve candidate chunks from FAISS
    candidates = retrieve_chunks(
        query_embedding,
        index,
        chunks,
        top_k=10
    )

    # 3. Rerank candidates
    results = rerank_chunks(
        query,
        candidates,
        top_k=3
    )

    # 4. Generate answer using the retrieved context
    answer = generate_answer(
        query,
        results
    )

    # 5. Return answer + sources
    sources = []

    for result in results:

        sources.append({
            "document": result["document"],
            "page": result["page"],
            "similarity": result["similarity"],
            "rerank_score": result["rerank_score"]
        })

    return {
        "answer": answer,
        "sources": sources
    }