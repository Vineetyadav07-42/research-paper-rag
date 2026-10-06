from pathlib import Path
from src.ingestion import extract_text_from_pdf

BASE_DIR=Path(__file__).resolve().parent.parent

pdf_path = BASE_DIR/'data'/'papers'/'Attention_is_all_you_need.pdf'

text = extract_text_from_pdf(pdf_path)

print(text[:500])

