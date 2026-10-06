import numpy as np
import faiss


def retrieve_chunks(
    query_embedding,
    index,
    chunks,
    top_k=3
):

    query_embedding = np.asarray(
        query_embedding
    ).astype("float32")

    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    # Normalize query vector
    faiss.normalize_L2(query_embedding)

    similarities, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for similarity, index_position in zip(
        similarities[0],
        indices[0]
    ):

        chunk = chunks[index_position]

        results.append({
            "text": chunk["text"],
            "page": chunk["page"],
            "document": chunk["document"],
            "similarity": float(similarity)
        })

    return results