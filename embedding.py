"""
自定义 Ollama Embedding —— 批量 + Session 复用
"""

from typing import List
from langchain_core.embeddings import Embeddings
from config import CONFIG
from session import get_session


class DirectOllamaEmbeddings(Embeddings):
    def __init__(self, model="bge-m3", base_url="http://127.0.0.1:11434", batch_size=50):
        self.model = model
        self.base_url = base_url
        self.batch_size = batch_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        total = len(texts)
        session = get_session()
        for i in range(0, total, self.batch_size):
            batch = texts[i:i + self.batch_size]
            print(f"  Embedding: {min(i + self.batch_size, total)}/{total}")
            response = session.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": batch},
                timeout=300
            )
            if response.status_code != 200:
                raise Exception(f"Embedding 失败: {response.text}")
            embeddings.extend(response.json()["embeddings"])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        response = get_session().post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": [text]},
            timeout=120
        )
        if response.status_code != 200:
            raise Exception(f"Embedding 失败: {response.text}")
        return response.json()["embeddings"][0]
