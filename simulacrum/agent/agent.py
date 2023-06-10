from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
from textwrap import dedent
from datetime import datetime

from langchain import LLMChain
from langchain.schema import Document

from memory.chains import (
    AgentSummary,
    # EntityAction,
    # EntityObserved,
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

    generate_agent_summary: LLMChain = Field(init=False)
    generate_reflections: LLMChain = Field(init=False)
    generate_importance: LLMChain = Field(init=False)
    generate_compression: LLMChain = Field(init=False)
    # generate_entity_observed: LLMChain = Field(init=False)
    # generate_entity_action: LLMChain = Field(init=False)

    importance_weight: float = 0.20

    importance_sum: float = 0.5
    reflection_threshold: Optional[float] = float("inf")

    time_without_reflection: int = 0
    countdown_to_reflection: int = 5

    verbose: bool = False
    max_token_limit: int = 1200
    last_refresh: datetime = Field(default_factory=datetime.now)

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
            generate_agent_summary=AgentSummary.from_llm(**chain),
            generate_reflections=MemoryReflect.from_llm(**chain),
            generate_importance=MemoryImportance.from_llm(**chain),
            generate_compression=MemoryCompress.from_llm(**chain),
            # generate_entity_observed=EntityObserved.from_llm(**chain),
            # generate_entity_action=EntityAction.from_llm(**chain),
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

    def _add_memory(
        self, memory_fragment: str, now: Optional[datetime] = None
    ) -> List[str]:
        score = self._predict_importance(memory_fragment)
        print(f"Memory score: {score}")
        document = Document(
            page_content=memory_fragment, metadata={"importance": score}
        )
        result = self.long_term_memory.add_documents([document], current_time=now)
        self.importance_sum += score
        self.time_without_reflection += 1
        return result

    def _predict_importance(self, memory_fragment: str) -> float:
        score = "".join(
            self.generate_importance.run(
                context=self.get_agent_info(), memory_fragment=memory_fragment
            )
        ).strip()
        print("\n")
        print(f"Predict Importance: {score}")
        match = re.search(r"^\D*(\d+)", score)
        print(f"Match: {match}\n")
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
        print(f"Computed short term: {self.short_term_memory}")

    def fetch_memories(self, observation: str) -> List[str]:
        return self.long_term_memory.get_relevant_documents(observation)

    def time_to_reflect(self) -> bool:
        return (
            self.time_without_reflection >= self.countdown_to_reflection - 1
            or self.importance_sum > self.reflection_threshold
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

    def _pause_and_reflect(self, now: Optional[datetime] = None) -> List[str]:
        print(f"{self.name} is reflecting...")
        reflections = self._get_salient_questions()
        # compressed_reflections = self._compress_memories()
        for reflection in reflections:
            self.add_memory(reflection)
        return reflections

    def _get_salient_questions(self, last_k: int = 25) -> List[str]:
        return self.generate_reflections.run(
            recent_memories=self.short_term_memory
        ).strip()

    def _compress_memories(self, last_k: Optional[int] = None) -> Tuple[str, str, str]:
        if last_k is None:
            last_k = self.countdown_to_reflection
        observations = self.long_term_memory[-last_k:]
        print(f"Observations in Compression: {observations}")
        if not any(observations):
            return []
        return self.generate_compression.run(
            context=self.get_agent_info(),
            memories="\n".join(o.page_content for o in observations),
        )

    # def _get_entity_from_observed(self, observation: str = "") -> str:
    #     return self.generate_entity_observed.run(observation=observation).strip()

    # def _get_entity_action(self, observation: str, entity_name: str) -> str:
    #     return self.generate_entity_action.run(
    #         observation=observation, entity=entity_name
    #     ).strip()
    #     return self.short_term_memory
