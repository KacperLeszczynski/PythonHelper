from sentence_transformers import CrossEncoder


class RerankerService:
    def __init__(self):
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def rerank_results(self, query, retrieved_docs):
        if not retrieved_docs:
            return []

        pairs = [(query, doc) for doc in retrieved_docs]
        scores = self.reranker.predict(pairs)

        ranked_results = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in ranked_results]


reranker_service = RerankerService()


def get_reranker_service() -> RerankerService:
    return reranker_service
