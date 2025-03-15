import openai
import os
from dotenv import load_dotenv
from openai.types.chat import ChatCompletion
import tiktoken

from models.chat_model_type_enum import ChatModelTypeEnum

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
tokenizer = tiktoken.encoding_for_model("text-embedding-ada-002")


class OpenAIService:
    def generate(self, messages, model: ChatModelTypeEnum, temperature: float) -> str | None:
        response = openai.chat.completions.create(
            model=model.value,
            temperature=temperature,
            messages=messages
        )

        return response.choices[0].message.content

    def get_openai_embedding(self, text):
        response = openai.embeddings.create(
            input=[text],
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding


openai_service = OpenAIService()


def get_openai_service() -> OpenAIService:
    return openai_service
