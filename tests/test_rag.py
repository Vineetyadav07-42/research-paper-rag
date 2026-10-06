from pathlib import Path

from src.embeddings import create_embeddings
from src.vector_store import load_vector_store
from src.storage import load_chunks
from src.retrieval import retrieve_chunks
from src.llm import generate_answer
from src.reranker import rerank_chunks


BASE_DIR = Path(__file__).resolve().parent.parent

INDEX_DIR = BASE_DIR / "data" / "index"

INDEX_PATH = INDEX_DIR / "faiss_index.bin"
CHUNKS_PATH = INDEX_DIR / "chunks.pkl"


# Load saved index
index = load_vector_store(INDEX_PATH)

# Load chunks
chunks = load_chunks(CHUNKS_PATH)

# User question
query = "What is the purpose of self-attention?"

# Embed question
query_embedding = create_embeddings([query])

# Retrieve relevant chunks
candidates = retrieve_chunks(
    query_embedding,
    index,
    chunks,
    top_k=10
)

results = rerank_chunks(
    query,
    candidates,
    top_k=3
)


# Generate answer
answer = generate_answer(
    query,
    results
)

print("\nQUESTION:")
print(query)

print("\nANSWER:")
print(answer)

print("\nSOURCES:\n")

for i, result in enumerate(results, start=1):

    print(f"--- Source {i} ---")

    print(f"Document: {result['document']}")

    print(f"Page: {result['page']}")

    print(f"FAISS similarity: {result['similarity']:.4f}")
    
    print(f"Rerank score: {result['rerank_score']:.4f}")

    print(f"\n{result['text']}")

    print()