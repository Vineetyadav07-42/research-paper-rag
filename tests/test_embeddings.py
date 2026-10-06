from pathlib import Path

from src.ingestion import extract_text_from_pdf
from src.chunking import chunk_text
from src.embeddings import create_embeddings


BASE_DIR = Path(__file__).resolve().parent.parent

pdf_path = BASE_DIR / "data" / "papers" / "attention_is_all_you_need.pdf"

text = extract_text_from_pdf(pdf_path)

chunks = chunk_text(text)

embeddings = create_embeddings(chunks)

print("Number of chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)