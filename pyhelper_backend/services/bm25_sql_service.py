import os
import pickle
from nltk.tokenize import word_tokenize

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BM25_PER_PYTHON_PATH = os.path.normpath(os.path.join(BASE_DIR, "../../rag_implementation/bm25_per_python_version.pkl"))


class Bm25SqlService:
    def __init__(self):
        self.bm25_per_python = self.load_bm25_per_python()

    def search(self, query, python_version, top_k=5):
        data = self.bm25_per_python.get(python_version)
        if not data:
            return []

        bm25 = data["bm25"]
        documents = data["documents"]
        ids = data["ids"]

        tokenized_query = word_tokenize(query.lower())
        scores = bm25.get_scores(tokenized_query)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        return [documents[i] for i in top_indices]

    def load_bm25_per_python(self):
        with open(BM25_PER_PYTHON_PATH, "rb") as f:
            return pickle.load(f)


bm25_sql_service = Bm25SqlService()


def get_bm25_sql_service() -> Bm25SqlService:
    return bm25_sql_service
