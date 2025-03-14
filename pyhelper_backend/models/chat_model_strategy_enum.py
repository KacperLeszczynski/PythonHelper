from enum import Enum

from services.chat_types_services.chat_strategy import ChatStrategy
from services.chat_types_services.vector_strategy import VectorStrategy, get_vector_strategy


class ChatModelStrategyEnum(Enum):
    VECTOR = "vector"
    VECTOR_BM25 = "vector_bm25"
    VECTOR_RERANK = "vector_rerank"
    VECTOR_BM25_RERANK = "vector_bm25_rerank"


class StrategyFactory:
    @staticmethod
    def get_strategy(strategy_enum: ChatModelStrategyEnum) -> ChatStrategy:
        match strategy_enum:
            case ChatModelStrategyEnum.VECTOR:
                return get_vector_strategy()
            case _:
                raise ValueError(f"Unsupported strategy: {strategy_enum}")
