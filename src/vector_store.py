import faiss
import numpy as np


def create_vector_store(embeddings):

    embeddings = np.asarray(
        embeddings
    ).astype("float32")

    # Normalize vectors
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    # Inner product on normalized vectors
    # is equivalent to cosine similarity
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


def save_vector_store(index, path):
    faiss.write_index(index, str(path))


def load_vector_store(path):
    return faiss.read_index(str(path))