from pathlib import Path

from src.ingestion import extract_text_from_pdf
from src.chunking import chunk_text


BASE_DIR = Path(__file__).resolve().parent.parent

pdf_path = BASE_DIR / "data" / "papers" / "attention_is_all_you_need.pdf"

text = extract_text_from_pdf(pdf_path)

chunks = chunk_text(text)

print("Total characters:", len(text))
print("Total chunks:", len(chunks))

print("\nFirst chunk:\n")
print(chunks[0])