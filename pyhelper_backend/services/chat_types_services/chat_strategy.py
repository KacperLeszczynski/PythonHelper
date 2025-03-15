from abc import ABC, abstractmethod

import tiktoken

from services.summary_buffer_memory import SummaryBufferMemory


class ChatStrategy(ABC):
    CHAT_MODEL = "gpt-4o-mini"
    CHAT_TEMPERATURE = 0.3
    TOKENIZER = tiktoken.encoding_for_model("text-embedding-ada-002")

    @abstractmethod
    def retrieve_documents(self, query, python_version, top_k=7):
        pass

    @abstractmethod
    def generate_response(self, query: str, retrieved_docs: list, memory: SummaryBufferMemory, python_version: str):
        pass

    def get_messages(self, prompt: str) -> list:
        return [{"role": "system",
                 "content": "You are python expert and you provide answer only based on given context."},
                {"role": "user", "content": prompt}]

