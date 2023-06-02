from local_models.llama import llama
from agent_memory.chains import Reflection, Importance, Compress
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

    # memory storage
    llm: llama = Field(init=False)
    long_term_memory: TimeWeightedVectorStoreRetriever = Field(init=False)
    short_term_memory: LLMChain = Field(init=False)

    # memory tooling
    generate_reflections: LLMChain = Field(init=False)
    generate_importance: LLMChain = Field(init=False)
    generate_compression: LLMChain = Field(init=False)

    importance_threshold: float = 0.0
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
            generate_compression=Compress.from_llm(**chain),
        )

    def _predict_importance(self, experience: str) -> float:
        score = self.generate_importance.run(experience=experience).strip()
        match = re.search(r"^\D*(\d+)", score)
        if not match:
            return 0.0
        return (float(match.group(1)) / 10) * self.importance_weight

    def _add_memory(self, experience: str, now: Optional[datetime] = None) -> List[str]:
        score = self._predict_importance(experience)
        self.importance_threshold += score
        record = Document(content=experience, metadata={"importance": score})
        result = self.long_term_memory.add_documents([record], current_time=now)
        return result

    def add_memory(self, experience: str) -> List[str]:
        result = self._add_memory(experience)
        if self.time_to_reflect():
            self.pause_and_reflect()
        return result

    def time_to_reflect(self) -> bool:
        return (
            self.reflection_threshold is not None
            and self.importance_threshold > self.reflection_threshold
        )

    def pause_and_reflect(self):
        if self.status == "reflecting":
            return []

        prev_status = self.status
        self.status = "reflecting"
        insights = self._pause_and_reflect()
        self.importance_threshold = 0.0
        self.status = prev_status
        return insights

    def _pause_and_reflect(self) -> List[str]:
        reflections = self._compress_memories()
        for reflection in reflection:
            self.add_memory(reflection)
        return reflections

    def _compress_memories(self, last_k: int = 50) -> List[str]:
        observations = self.long_term_memory.memory_stream[-last_k:]
        if not any(observations):
            return []
        return self.generate_compression.run(
            # something
        )
        
