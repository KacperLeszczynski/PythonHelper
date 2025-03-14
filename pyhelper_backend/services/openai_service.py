import openai
import os
from dotenv import load_dotenv
from openai.types.chat import ChatCompletion
import tiktoken


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
tokenizer = tiktoken.encoding_for_model("text-embedding-ada-002")


class OpenAIService:
    def generate(self, messages, model: str, temperature: float) -> ChatCompletion:
        response = openai.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=messages
        )

        return response

    def get_openai_embedding(self, text):
        response = openai.embeddings.create(
            input=[text],
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding


openai_service = OpenAIService()


def get_openai_service() -> OpenAIService:
    return openai_service
