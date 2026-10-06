import nltk
from nltk.tokenize import sent_tokenize


def chunk_pages(
    pages,
    chunk_size=500,
    overlap_sentences=1
):
    chunks = []

    for page in pages:

        text = page["text"].strip()

        if not text:
            continue

        sentences = sent_tokenize(text)

        current_sentences = []
        current_length = 0

        for sentence in sentences:

            sentence_length = len(sentence)

            # If adding this sentence exceeds the chunk size,
            # save the current chunk first.
            if (
                current_sentences
                and current_length + sentence_length > chunk_size
            ):

                chunk_text = " ".join(
                    current_sentences
                )

                chunks.append({
                    "text": chunk_text,
                    "page": page["page"],
                    "document": page["document"]
                })

                # Keep the last sentence as overlap
                current_sentences = (
                    current_sentences[
                        -overlap_sentences:
                    ]
                )

                current_length = sum(
                    len(sentence)
                    for sentence in current_sentences
                )

            current_sentences.append(sentence)

            current_length += sentence_length

        # Add remaining sentences
        if current_sentences:

            chunk_text = " ".join(
                current_sentences
            )

            chunks.append({
                "text": chunk_text,
                "page": page["page"],
                "document": page["document"]
            })

    return chunks