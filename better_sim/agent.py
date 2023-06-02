from local_models.llama import llama
from agent_memory.chains import Reflection, Importance
from pydantic import BaseModel, List, Field, Optional
from datetime import datetime

from langchain import LLMChain
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from lanchain.schema import Document

import networkx as nx
import re


class Agent(BaseModel):
    # core
    name: str
    description: str
    traits: List[str] = Field(default_factory=list)
    plan: List[str] = []

    # sandbox
    location: str = Field(default="")
    world_graph: nx.Graph = Field(init=False, default=nx.Graph)

    # state
    status: str = "idle"
    short_term_memory: str = ""
    last_refresh: datetime = Field(default_factory=None)

    # memory tooling
    llm: llama = Field(init=False)
    long_term_memory: TimeWeightedVectorStoreRetriever = Field(init=False)
    short_term_memory: LLMChain = Field(init=False)
    compressed_memory: LLMChain = Field(init=False)

    # chains
    generate_reflections: LLMChain = Field(init=False)
    generate_importance: LLMChain = Field(init=False)

    importance_sum: float = 0.0
    importance_weight: float = 0.10
    reflection_threshold: Optional[float] = None

    verbose: bool = False
    max_token_limit: int = 1200

    class Config:
        # avoid validation errors
        arbitrary_types_allowed = True

    def __init__(self, *args, **kwargs):
        chain = dict(
            llm=kwargs["llm"],
            verbose=kwargs.get("verbose", True),
            callback_manager=kwargs.get("callback_manager", None),
        )
        super().__init__(
            *args,
            **kwargs,
            generate_reflections=Reflection.from_llm(**chain),
            generate_importance=Importance.from_llm(**chain)
        )

    def _predict_importance(self, experience_content: str) -> float:
        score = self.generate_importance.run(
            experience_content=experience_content
        ).strip()
        match = re.search(r"^\D*(\d+)", score)
        if not match:
            return 0.0
        return (float(match.group(1)) / 10) * self.importance_weight

    def add_memory(
        self, experience_content: str, now: Optional[datetime] = None
    ) -> List[str]:
        score = self._predict_importance(experience_content)
        self.importance_sum += score
        record = Document(content=experience_content, metadata={"importance": score})
        result = self.long_term_memory.add_documents([record], current_time=now)
        return result
