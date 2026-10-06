from pathlib import Path

from src.ingestion import extract_text_from_pdf
from src.chunking import chunk_text
from src.embeddings import create_embeddings
from src.vector_store import create_vector_store


BASE_DIR = Path(__file__).resolve().parent.parent

pdf_path = (
    BASE_DIR
    / "data"
    / "papers"
    / "attention_is_all_you_need.pdf"
)

# 1. Extract text
text = extract_text_from_pdf(pdf_path)

# 2. Create chunks
chunks = chunk_text(text)

# 3. Create embeddings
embeddings = create_embeddings(chunks)

# 4. Create FAISS index
index = create_vector_store(embeddings)

print("Number of chunks:", len(chunks))
print("Embedding dimension:", embeddings.shape[1])
print("FAISS vectors:", index.ntotal)