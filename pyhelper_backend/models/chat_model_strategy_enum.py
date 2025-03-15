from enum import Enum

from services.chat_types_services.chat_strategy import ChatStrategy
from services.chat_types_services.vector_bm25_rerank_strategy import get_vector_bm25_rerank_strategy
from services.chat_types_services.vector_bm25_strategy import get_vector_bm25_strategy
from services.chat_types_services.vector_rerank_strategy import get_vector_rerank_strategy
from services.chat_types_services.vector_strategy import get_vector_strategy


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
            case ChatModelStrategyEnum.VECTOR_RERANK:
                return get_vector_rerank_strategy()
            case ChatModelStrategyEnum.VECTOR_BM25_RERANK:
                return get_vector_bm25_rerank_strategy()
            case ChatModelStrategyEnum.VECTOR_BM25:
                return get_vector_bm25_strategy()
            case _:
                raise ValueError(f"Unsupported strategy: {strategy_enum}")
