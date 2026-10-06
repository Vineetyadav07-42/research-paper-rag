import pickle


def save_chunks(chunks, path):
    with open(path, "wb") as file:
        pickle.dump(chunks, file)


def load_chunks(path):
    with open(path, "rb") as file:
        return pickle.load(file)