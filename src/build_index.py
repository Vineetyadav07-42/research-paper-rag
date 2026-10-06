from pathlib import Path

from src.ingestion import extract_pages_from_pdf
from src.chunking import chunk_pages
from src.embeddings import create_embeddings
from src.vector_store import create_vector_store, save_vector_store
from src.storage import save_chunks


BASE_DIR = Path(__file__).resolve().parent.parent

PDF_PATH = (
    BASE_DIR
    / "data"
    / "papers"
    / "attention_is_all_you_need.pdf"
)

INDEX_DIR = BASE_DIR / "data" / "index"

INDEX_DIR.mkdir(parents=True, exist_ok=True)


# 1. Extract pages
print("Extracting pages...")

pages = extract_pages_from_pdf(PDF_PATH)

print(f"Extracted {len(pages)} pages.")


# 2. Create chunks
print("Creating chunks...")

chunks = chunk_pages(pages)

print(f"Created {len(chunks)} chunks.")


# 3. Extract text from chunks
chunk_texts = [
    chunk["text"]
    for chunk in chunks
]


# 4. Create embeddings
print("Creating embeddings...")

embeddings = create_embeddings(chunk_texts)


# 5. Create FAISS index
print("Creating FAISS index...")

index = create_vector_store(embeddings)


# 6. Save FAISS index
index_path = INDEX_DIR / "faiss_index.bin"

save_vector_store(index, index_path)


# 7. Save chunks + metadata
chunks_path = INDEX_DIR / "chunks.pkl"

save_chunks(chunks, chunks_path)


print("\nIndexing completed!")

print(f"FAISS index: {index_path}")
print(f"Chunks: {chunks_path}")