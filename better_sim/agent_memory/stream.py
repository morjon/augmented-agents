from local_models.llama import llama
from datetime import datetime, timedelta
from pydantic import Field
from typing import List

from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.embeddings import LlamaCppEmbeddings
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS
from langchain.schema import Document

from functools import lru_cache
from faiss import IndexFlatL2

import numpy as np


class MemoryStream(TimeWeightedVectorStoreRetriever):
    def __init__(self, llm: llama):
        embeddings = LlamaCppEmbeddings(model_path=llm.get_model_path)
        super().__init__(
            vectorstore=FAISS(
                embedding_function=embeddings.embed_query,
                index=IndexFlatL2(1536),
                docstore=InMemoryDocstore({}),
                index_to_docstore_id={},
            )
        )
        self.last_refreshed: datetime = Field(default_factory=datetime.now)
        self.llm_embeddings = embeddings

    def _get_recency_score(self, doc_time: datetime) -> float:
        time_diff = (self.last_refreshed - doc_time).total_seconds()
        max_time_diff = (self.last_refreshed - timedelta(days=30)).total_seconds()
        recency_score = 1 - ((time_diff - max_time_diff) / max_time_diff)
        return max(recency_score, 0)

    def _get_relevance_score(self, doc: Document, context: str) -> float:
        prompt_embedding = self.llm_embeddings.embed_query(context)
        doc_embedding = self.llm_embeddings.embed_documents([doc.content])
        relevance_score = np.dot(prompt_embedding, doc_embedding) / (
            np.linalg.norm(prompt_embedding) * np.linalg.norm(doc_embedding)
        )
        return relevance_score

    @lru_cache(maxsize=None)  # maybe
    def retrieve_memories(
        self,
        context: str,
        last_k: int = 50,
        alpha_recency: float = 1.0,
        alpha_relevance: float = 1.0,
        alpha_importance: float = 1.0,
    ) -> List[str]:
        docs = self.memory_stream[-last_k:]
        memories = []
        for doc in docs:
            recency_score = self._get_recency_score(doc.meta["timestamp"])
            relevance_score = self._get_relevance_score(doc, context)
            importance_score = doc.meta["importance"]
            retrieval_score = (
                alpha_recency * recency_score
                + alpha_relevance * relevance_score
                + alpha_importance * importance_score
            )
            memories.append({"text": doc.content, "score": retrieval_score})
        memories_sorted = sorted(memories, key=lambda x: x["score"], reverse=True)
        return [m["text"] for m in memories_sorted[:last_k]]
