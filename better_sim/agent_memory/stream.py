from local_models.llama import llama
from typing import Optional
from pydantic import Field

from langchain import LLMChain
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.schema import BaseMemory

# consider abstracting some agent.py stuff (test chain usage first..)
class MemoryStream(BaseMemory):
    llm: llama = Field(init=False)
    retriever: TimeWeightedVectorStoreRetriever
    threshold: Optional[float] = None

    importance_sum: LLMChain = Field(init=False)
    importance_weight: float = 0.10