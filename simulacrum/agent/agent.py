from pydantic import BaseModel, Field
from typing import List, Optional
from textwrap import dedent

from langchain import LLMChain
from langchain.schema import Document

from memory.chains import (
    AgentSummary,
    EntityAction,
    EntityObserved,
    MemoryCompress,
    MemoryImportance,
    MemoryReflect,
)
from models.local_llamas import vicuna
from memory.stream import AgentMemory

import re


class Agent(BaseModel):
    name: str
    description: str
    traits: List[str] = Field(default_factory=list)

    llm: vicuna = Field(init=False)
    long_term_memory = AgentMemory()
    short_term_memory: str = ""
    status: str = "idle"

    generate_reflections: LLMChain = Field(init=False)
    generate_importance: LLMChain = Field(init=False)
    generate_compression: LLMChain = Field(init=False)
    generate_entity_observed: LLMChain = Field(init=False)
    generate_entity_action: LLMChain = Field(init=False)
    generate_agent_summary: LLMChain = Field(init=False)

    importance_sum: float = 0.5
    importance_weight: float = 0.20
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
        print(f"Chain dict: {chain}")

        super().__init__(
            *args,
            **kwargs,
            generate_reflections=MemoryReflect.from_llm(**chain),
            generate_importance=MemoryImportance.from_llm(**chain),
            generate_compression=MemoryCompress.from_llm(**chain),
            generate_entity_observed=EntityObserved.from_llm(**chain),
            generate_entity_action=EntityAction.from_llm(**chain),
            generate_agent_summary=AgentSummary.from_llm(**chain),
        )

    def get_agent_info(self):
        return dedent(
            f"""\

            Name: {self.name}
            Description: {self.description}
            Traits: {", ".join(self.traits)}
            """
        )

    def add_memory(self, memory_fragment: str) -> List[str]:
        result = self._add_memory(memory_fragment)
        if self.time_to_reflect():
            self.pause_and_reflect()
        return result

    def _add_memory(self, memory_fragment: str) -> List[str]:
        content = memory_fragment
        score = self._predict_importance(content)
        print(f"Memory score: {score}\n")
        nodes = self.long_term_memory.add_documents(
            [Document(page_content=content, metadata={"importance": score})]
        )
        print(content if len(content) < 280 else content[:280] + "...")
        self.importance_sum += score

        return nodes

    def _predict_importance(self, memory_fragment: str) -> float:
        score = "".join(
            self.generate_importance.run(memory_fragment=memory_fragment)
        ).strip()
        print(f"Predict Importance: {score}\n")
        match = re.search(r"^\D*(\d+)", score)
        if not match:
            return 0.0
        return (float(match.group(1)) / 10) * self.importance_weight

    def get_context(self, force_refresh: bool = False) -> str:
        context = self.short_term_memory
        if not context or force_refresh or self.time_to_reflect():
            context = self._compute_agent_memory()

        return f"{self.get_agent_info()}Summary:\n{context}\n"

    def _compute_agent_memory(self):
        memories = self.fetch_memories("Importance.")
        if not any(memories):
            return "[Empty]"

        self.short_term_memory = "\n".join(
            f"{n}. {mem}"
            for (n, mem) in enumerate(
                self.generate_compression.run(
                    context=self.get_agent_info(),
                    memories="\n".join(f"{m.page_content}" for m in memories),
                )
            )
        )

    def _pause_and_reflect(self) -> List[str]:
        compressed_reflections = self._compress_memories()
        for reflection in compressed_reflections:
            self.add_memory(reflection)
        return compressed_reflections

    def _compress_memories(self, last_k: int = 50) -> List[str]:
        observations = self.long_term_memory[-last_k:]
        if not any(observations):
            return []
        return self.generate_compression.run(
            context=self.get_agent_info(),
            memories="\n".join(o.page_content for o in observations),
        )

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
        reflections = self._pause_and_reflect()
        self.importance_sum = 0.0
        self.status = prev_status
        return reflections

    def _get_entity_from_observed(self, observation: str = "") -> str:
        return self.generate_entity_observed.run(observation=observation).strip()

    def _get_entity_action(self, observation: str, entity_name: str) -> str:
        return self.generate_entity_action.run(
            observation=observation, entity=entity_name
        ).strip()
        return self.short_term_memory
