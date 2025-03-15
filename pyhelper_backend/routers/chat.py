from fastapi import APIRouter

from models.chat_model_strategy_enum import ChatModelStrategyEnum
from models.chat_model_type_enum import ChatModelTypeEnum
from services.chat_completion import ChatCompletionService
from services.summary_buffer_memory import get_summary_buffer_memory

router = APIRouter(prefix="/chat", tags=["chat"])
@router.get("/")
async def get_answer(user_query: str,
                     python_version: str = "3.10",
                     num_docs: int = 5,
                     strategy: ChatModelStrategyEnum = ChatModelStrategyEnum.VECTOR,
                     chat_model: ChatModelTypeEnum = ChatModelTypeEnum.GPT_4O_MINI):
    strategy_enum = ChatModelStrategyEnum(strategy)
    chat_service = ChatCompletionService(strategy_enum, get_summary_buffer_memory(), chat_model)

    response = chat_service.generate_answer(user_query, python_version, num_docs)
    return {"response": response}


