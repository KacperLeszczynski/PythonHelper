from services.bm25_sql_service import Bm25SqlService, get_bm25_sql_service
from services.chat_types_services.chat_strategy import ChatStrategy
from services.chroma_service import ChromaService, get_chroma_service
from services.openai_service import OpenAIService, get_openai_service
from services.reranker_service import RerankerService, get_reranker_service
from services.summary_buffer_memory import SummaryBufferMemory, get_summary_buffer_memory


class VectorBM25RerankStrategy(ChatStrategy):
    def __init__(self, openai_service: OpenAIService, memory: SummaryBufferMemory,
                 chroma_service: ChromaService, bm25_sql_service: Bm25SqlService, reranker_service: RerankerService):
        self.openai_service = openai_service
        self.tokenizer = self.TOKENIZER
        self.memory = memory
        self.chroma_service = chroma_service
        self.bm25_sql_service = bm25_sql_service
        self.reranker_service = reranker_service

    def retrieve_documents(self, query, python_version, top_k=7):
        results = self.chroma_service.query(query, python_version, top_k)
        embedding_results = results["documents"][0] if "documents" in results and results["documents"] else []
        bm25_results = self.bm25_sql_service.search(query, python_version, 5)
        combined_results = list(set(embedding_results + bm25_results))

        return combined_results

    def generate_response(self, query, retrieved_docs, memory, python_version, chat_model):
        reranked_docs = self.reranker_service.rerank_results(query, retrieved_docs)
        context = "\n\n".join(reranked_docs)
        prompt = self.memory.get_prompt(query, context, python_version)

        response = self.openai_service.generate(
            messages=self.get_messages(prompt),
            model=chat_model,
            temperature=self.CHAT_TEMPERATURE
        )

        self.memory.add_interaction(query, assistant_output=response)

        return response


vector_bm25_rerank_strategy = VectorBM25RerankStrategy(
    openai_service=get_openai_service(),
    chroma_service=get_chroma_service(),
    memory=get_summary_buffer_memory(),
    bm25_sql_service=get_bm25_sql_service(),
    reranker_service=get_reranker_service()
)


def get_vector_bm25_rerank_strategy() -> VectorBM25RerankStrategy:
    return vector_bm25_rerank_strategy