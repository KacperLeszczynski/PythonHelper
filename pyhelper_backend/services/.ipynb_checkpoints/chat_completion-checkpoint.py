from models.chat_model_strategy_enum import ChatModelStrategyEnum, StrategyFactory
from models.chat_model_type_enum import ChatModelTypeEnum
from services.summary_buffer_memory import SummaryBufferMemory


class ChatCompletionService:
    def __init__(self, strategy: ChatModelStrategyEnum, summary: SummaryBufferMemory, chat_model: ChatModelTypeEnum):
        self.summary = summary
        self.strategy = StrategyFactory.get_strategy(strategy)
        self.chat_model = chat_model

    def format_query(self, user_query: str):
        if "python" not in user_query.lower():
            return f"{user_query} in Python"
        return user_query

    def generate_answer(self, query, python_version, num_docs):
        query = self.format_query(query)
        retrieved_docs = self.strategy.retrieve_documents(query, python_version, num_docs)
        response = self.strategy.generate_response(query, retrieved_docs, self.summary, python_version, self.chat_model)
        return response