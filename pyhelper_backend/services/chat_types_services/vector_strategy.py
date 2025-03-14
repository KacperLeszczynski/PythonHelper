import tiktoken

from services.chat_types_services.chat_strategy import ChatStrategy
from services.chroma_service import ChromaService, get_chroma_service
from services.openai_service import OpenAIService, get_openai_service
from services.summary_buffer_memory import SummaryBufferMemory, get_summary_buffer_memory


class VectorStrategy(ChatStrategy):
    def __init__(self, openai_service: OpenAIService, tokenizer, memory: SummaryBufferMemory,
                 chroma_service: ChromaService):
        self.openai_service = openai_service
        self.tokenizer = tokenizer
        self.memory = memory
        self.chroma_service = chroma_service

    def retrieve_documents(self, query, python_version, top_k=7):
        results = self.chroma_service.query(query, python_version, top_k)

        return results["documents"][0] if "documents" in results and results["documents"] else []

    def generate_response(self, query, retrieved_docs, memory, python_version):
        context = "\n\n".join(retrieved_docs)
        prompt = self.memory.get_prompt(query, context, python_version)

        response = self.openai_service.generate(
            messages=self.get_messages(prompt),
            model=self.CHAT_MODEL,
            temperature=self.CHAT_TEMPERATURE
        )

        self.memory.add_interaction(query, assistant_output=response)

        return response

tokenizer = tiktoken.encoding_for_model("text-embedding-ada-002")

vector_strategy = VectorStrategy(
    openai_service=get_openai_service(),
    chroma_service=get_chroma_service(),
    memory=get_summary_buffer_memory(),
    tokenizer=tokenizer
)

def get_vector_strategy() -> VectorStrategy:
    return vector_strategy