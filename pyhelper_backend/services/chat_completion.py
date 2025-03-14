from models.chat_model_strategy_enum import ChatModelStrategyEnum, StrategyFactory
from services.summary_buffer_memory import SummaryBufferMemory


class ChatCompletionService:
    def __init__(self, strategy: ChatModelStrategyEnum, summary: SummaryBufferMemory):
        self.summary = summary
        self.strategy = StrategyFactory.get_strategy(strategy)

    def format_query(self, user_query: str):
        if "python" not in user_query.lower():
            return f"{user_query} in Python"
        return user_query

    def generate_answer(self, query, python_version, num_docs):
        query = self.format_query(query)
        retrieved_docs = self.strategy.retrieve_documents(query, python_version, num_docs)
        response = self.strategy.generate_response(query, retrieved_docs, self.summary, python_version)
        return response