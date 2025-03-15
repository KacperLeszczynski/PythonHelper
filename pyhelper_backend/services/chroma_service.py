import chromadb
from dotenv import load_dotenv
import os

from services.openai_service import OpenAIService, get_openai_service

load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_PATH = os.path.normpath(os.path.join(BASE_DIR, "../../data_collecting/chroma_db"))


class ChromaService:
    def __init__(self, openai_service: OpenAIService, collection_name: str = "python_data"):
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.openai_service = openai_service

    def query(self, query_text: str, python_version, top_k: int = 5):
        query_embedding = self.openai_service.get_openai_embedding(query_text)

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"version": python_version}
        )


chroma_service = ChromaService(openai_service=get_openai_service())


def get_chroma_service() -> ChromaService:
    return chroma_service
