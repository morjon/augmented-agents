from local_models.llama import llama
from agent_memory.chains import Reflect, Importance, Compress
from agent_memory.stream import MemoryStream
from pydantic import BaseModel, List, Field, Optional
from datetime import datetime
from textwrap import dedent

from langchain import LLMChain
from lanchain.schema import Document

import re


class Agent(BaseModel):
    name: str
    description: str
    traits: List[str] = Field(default_factory=list)
    plan: List[str] = []

    llm: llama = Field(init=False)
    long_term_memory: MemoryStream = Field(init=False)
    short_term_memory: LLMChain = Field(init=False)

    generate_reflections: LLMChain = Field(init=False)
    generate_importance: LLMChain = Field(init=False)
    generate_compression: LLMChain = Field(init=False)

    status: str = "idle"
    short_term_memory_state: str = ""

    importance_sum: float = 0.5
    reflection_threshold: Optional[float] = None

    verbose: bool = False
    max_token_limit: int = 1200

    class Config:
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
            generate_reflections=Reflect.from_llm(**chain),
            generate_importance=Importance.from_llm(**chain),
            generate_compression=Compress.from_llm(**chain),
        )

    def get_agent_info(self):
        return dedent(
            f"""\
            Name: {self.name}
            Description: {self.description}
            Traits: {", ".join(self.traits)}
            """
        )

    def add_memory(self, event: str) -> List[str]:
        result = self._add_memory(event)
        if self.time_to_reflect():
            self.pause_and_reflect()
        return result

    def fetch_memories(self, observation: str) -> List[str]:
        return self.long_term_memory.get_relevant_documents(observation)

    def time_to_reflect(self) -> bool:
        return (
            self.reflection_threshold is not None
            and self.importance_sum > self.reflection_threshold
        )

    def pause_and_reflect(self):
        if self.status == "reflecting":
            return []

        prev_status = self.status
        self.status = "reflecting"
        insights = self._pause_and_reflect()
        self.importance_sum = 0.0
        self.status = prev_status
        return insights

    def _add_memory(self, event: str, now: Optional[datetime] = None) -> List[str]:
        score = self._predict_importance(event)
        self.importance_sum += score
        record = Document(content=event, metadata={"importance": score})
        result = self.long_term_memory.add_documents([record], current_time=now)
        return result

    def _pause_and_reflect(self) -> List[str]:
        reflections = self._compress_memories()
        for reflection in reflections:
            self.add_memory(reflection)
        return reflections

    def _compress_memories(self, last_k: int = 50) -> List[str]:
        observations = self.long_term_memory.memory_stream[-last_k:]
        if not any(observations):
            return []
        return self.generate_compression.run(
            context=self.get_agent_info(),
            memories="\n".join(o.page_content for o in observations),
        )

    def _predict_importance(self, event: str) -> float:
        score = self.generate_importance.run(event=event).strip()
        match = re.search(r"^\D*(\d+)", score)
        if not match:
            return 0.0
        return (float(match.group(1)) / 10) * self.importance_weight
