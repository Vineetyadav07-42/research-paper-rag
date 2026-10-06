from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

reranker = CrossEncoder(MODEL_NAME)


def rerank_chunks(query, candidates, top_k=3):

    pairs = [
        (query, candidate["text"])
        for candidate in candidates
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: float(x[1]),
        reverse=True
    )

    results = []

    for chunk, score in ranked[:top_k]:

        result = dict(chunk)

        result["rerank_score"] = float(score)

        results.append(result)

    return results