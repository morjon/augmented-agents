from pydantic import BaseModel, List, Field, Optional
from textwrap import dedent

from langchain import LLMChain
from lanchain.schema import Document

from agent_memory.chains import (
    EntityAction,
    EntityObserved,
    MemoryCompress,
    MemoryImportance,
    MemoryReflect,
)
from agent_memory.retriever import MemoryRetriever
from local_models.llama import llama

import re


class Agent(BaseModel):
    name: str
    description: str
    plan: List[str] = []
    traits: List[str] = Field(default_factory=list)
    status: str = "idle"

    llm: llama = Field(init=False)
    long_term_memory: MemoryRetriever = Field(init=False)

    generate_reflections: LLMChain = Field(init=False)
    generate_importance: LLMChain = Field(init=False)
    generate_compression: LLMChain = Field(init=False)
    generate_entity_observation: LLMChain = Field(init=False)
    generate_entity_action: LLMChain = Field(init=False)

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
            generate_reflections=MemoryReflect.from_llm(**chain),
            generate_importance=MemoryImportance.from_llm(**chain),
            generate_compression=MemoryCompress.from_llm(**chain),
            generate_entity_observed=EntityObserved.from_llm(**chain),
            generate_entity_action=EntityAction.from_llm(**chain),
        )

    def get_agent_info(self):
        return dedent(
            f"""\
            Name: {self.name}
            Description: {self.description}
            Traits: {", ".join(self.traits)}
            """
        )

    # memory stream related.. may abstract

    def add_memory(self, fragment: str) -> List[str]:
        result = self._add_memory(fragment)
        if self.time_to_reflect():
            self.pause_and_reflect()
        return result

    def fetch_memories(self, observation: str) -> List[str]:
        return self.long_term_memory.retrieve_and_filter_memories(observation)

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
        reflections = self._pause_and_reflect()
        self.importance_sum = 0.0
        self.status = prev_status
        return reflections

    def _add_memory(self, fragment: str) -> List[str]:
        score = self._predict_importance(fragment)
        self.importance_sum += score
        memory_record = Document(content=fragment, metadata={"importance": score})
        result = self.long_term_memory.add_documents([memory_record])
        return result

    def _pause_and_reflect(self) -> List[str]:
        compressed_reflections = self._compress_memories()
        for reflection in compressed_reflections:
            self.add_memory(reflection)
        return compressed_reflections

    def _compress_memories(self, last_k: int = 50) -> List[str]:
        observations = self.long_term_memory.memory_stream[-last_k:]
        if not any(observations):
            return []
        return self.generate_compression.run(
            context=self.get_agent_info(),
            memories="\n".join(o.page_content for o in observations),
        )

    def _predict_importance(self, fragment: str) -> float:
        score = self.generate_importance.run(fragment=fragment).strip()
        match = re.search(r"^\D*(\d+)", score)
        if not match:
            return 0.0
        return (float(match.group(1)) / 10) * self.importance_weight

    # agent/entity related

    def _get_entity_from_observed(self, observation: str = "") -> str:
        return self.generate_entity_observed.run(observation=observation).strip()

    def _get_entity_action(self, observation: str, entity_name: str) -> str:
        return self.generate_entity_action.run(
            observation=observation, entity=entity_name
        ).strip()
