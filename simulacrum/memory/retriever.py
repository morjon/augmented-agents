from collections import defaultdict
from functools import lru_cache
from typing import List

from datetime import datetime, timedelta

# from faiss import IndexFlatL2
from pydantic import Field
from sklearn.metrics.pairwise import cosine_similarity

# from langchain.docstore import InMemoryDocstore
# from langchain.embeddings import LlamaCppEmbeddings
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.schema import Document

# from langchain.vectorstores import FAISS

from models.llama import llama


class MemoryRetriever(TimeWeightedVectorStoreRetriever):
    last_refreshed: datetime = Field(default_factory=datetime.now)
    llm: llama

    def __init__(self, llm: llama):
        # embeddings = LlamaCppEmbeddings(
        #     model_path=
        # "/home/ubuntu/repos/augmented-agents/llama.cpp/models/13B/ggml-model-q4_0.bin"
        # )
        # super().__init__(
        #     vectorstore=FAISS(
        #         embedding_function=embeddings.embed_query,
        #         index=IndexFlatL2(1536),
        #         docstore=InMemoryDocstore({}),
        #         index_to_docstore_id={},
        #     )
        # )
        self.default_salience = 1.0
        self.k = 5  # max memory fragments retrieved on call

    def to_dict(self):
        return {
            "llm": self.llm,
            "last_refreshed": self.last_refreshed,
            "default_salience": self.default_salience,
            "k": self.k,
        }

    def _get_recency_score(self, doc_time: datetime) -> float:
        time_diff = (self.last_refreshed - doc_time).total_seconds()
        max_time_diff = (self.last_refreshed - timedelta(days=30)).total_seconds()
        recency_score = 1 - ((time_diff - max_time_diff) / max_time_diff)
        return max(recency_score, 0)

    @lru_cache(maxsize=None)  # maybe?
    def retrieve_and_filter_memories(
        self,
        context: str,
        query: str,
        last_k: int = 50,
        alpha_recency: float = 1.0,
        alpha_relevance: float = 1.0,
        alpha_importance: float = 1.0,
    ) -> List[Document]:
        current_time = datetime.datetime.now()
        docs_and_scores = defaultdict(lambda: (None, self.default_salience))
        for doc in self.memory_stream[-self.k :]:
            if doc.content and query in doc.content.lower():
                docs_and_scores[doc.metadata["buffer_idx"]] = (
                    doc,
                    self.default_salience,
                )
        memories = self.memory_stream[-last_k:]
        prompt_embedding = self.llm_embeddings.embed_query(context)
        for i, doc in enumerate(memories):
            if doc.content and doc.content not in (
                d.content for d, _ in docs_and_scores.values()
            ):
                doc_embedding = self.llm_embeddings.embed_documents([doc.content])
                recency_score = self._get_recency_score(doc.meta["timestamp"])
                relevance_score = cosine_similarity(
                    prompt_embedding.reshape(1, -1), doc_embedding.reshape(1, -1)
                )[0][0]
                importance_score = doc.meta["importance"]
                retrieval_score = (
                    alpha_recency * recency_score
                    + alpha_relevance * relevance_score
                    + alpha_importance * importance_score
                )
                docs_and_scores[len(self.memory_stream) + i] = (doc, retrieval_score)
        rescored_docs = [
            (doc, self._get_combined_score(doc, score, current_time))
            for doc, (doc_obj, score) in docs_and_scores.items()
            if doc_obj is not None
        ]
        rescored_docs.sort(key=lambda x: x[1], reverse=True)
        result = []
        for doc, _ in rescored_docs[: self.k]:
            buffered_doc = (
                self.memory_stream[doc.metadata["buffer_idx"]]
                if doc.metadata["buffer_idx"] is not None
                else doc
            )
            buffered_doc.metadata["last_accessed_at"] = datetime.fromtimestamp(
                current_time
            )
            result.append(buffered_doc)
        return result
