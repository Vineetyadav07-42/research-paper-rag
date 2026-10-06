from pathlib import Path

from src.embeddings import create_embeddings
from src.vector_store import load_vector_store
from src.storage import load_chunks
from src.retrieval import retrieve_chunks
from src.reranker import rerank_chunks


BASE_DIR = Path(__file__).resolve().parent.parent

INDEX_DIR = BASE_DIR / "data" / "index"

INDEX_PATH = INDEX_DIR / "faiss_index.bin"
CHUNKS_PATH = INDEX_DIR / "chunks.pkl"


index = load_vector_store(INDEX_PATH)
chunks = load_chunks(CHUNKS_PATH)


test_questions = [
    {
        "question": "What is the purpose of self-attention?",
        "expected_pages": [2],
    },
    {
        "question": "What is the Transformer architecture?",
        "expected_pages": [1, 2, 3],
    },
    {
        "question": "What is multi-head attention?",
        "expected_pages": [5],
    },
    {
        "question": "Why does the Transformer use positional encoding?",
        "expected_pages": [4],
    },
    {
        "question": "How does the Transformer differ from recurrent neural networks?",
        "expected_pages": [1, 2],
    },
]


for test in test_questions:

    question = test["question"]
    expected_pages = test["expected_pages"]

    print("\n" + "=" * 70)
    print(f"QUESTION: {question}")
    print(f"EXPECTED PAGES: {expected_pages}")

    # 1. Convert question to embedding
    query_embedding = create_embeddings([question])

    # 2. Retrieve candidates from FAISS
    candidates = retrieve_chunks(
        query_embedding,
        index,
        chunks,
        top_k=10
    )

    # 3. Rerank candidates
    results = rerank_chunks(
        question,
        candidates,
        top_k=3
    )

    # 4. Check whether expected page was retrieved
    retrieved_pages = [
        result["page"]
        for result in results
    ]

    hit = any(
        page in expected_pages
        for page in retrieved_pages
    )

    print(f"RETRIEVED PAGES: {retrieved_pages}")
    print(f"RESULT: {'PASS' if hit else 'FAIL'}")

    print("\nTOP RESULTS:")

    for i, result in enumerate(results, start=1):

        print(
            f"\n--- Result {i} ---"
        )

        print(
            f"Page: {result['page']}"
        )

        print(
            f"FAISS similarity: "
            f"{result['similarity']:.4f}"
        )

        print(
            f"Rerank score: "
            f"{result['rerank_score']:.4f}"
        )

        print(
            result["text"][:500]
        )