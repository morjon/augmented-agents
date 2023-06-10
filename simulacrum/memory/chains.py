from typing import Any, Dict, List
from textwrap import dedent

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# from langchain.chat_models.base import BaseChatModel

from models.local_llamas import vicuna

import re
import json


class MemoryParser(LLMChain):
    output_key = "items"

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, List[str]]:
        print(f"_call inputs: {inputs}\n")
        text = super()._call(inputs)[self.output_key].strip()

        items = [
            re.sub(r"^\s*\d+\.\s*", "", line).strip()
            for line in re.split(r"\n", text.strip())
        ]
        return {"items": items}

    def run(self, **kwargs) -> Dict[str, List[str]]:
        return self._call(inputs=kwargs)[self.output_key]


class MemoryJSONParser(LLMChain):
    key = "json"

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        data = json.loads(super()._call(inputs)["json"].strip())
        return {"json": data}


class MemoryCompress(MemoryParser):
    @classmethod
    def from_llm(cls, llm: vicuna, verbose: bool = True, **kwargs) -> LLMChain:
        return cls(
            **kwargs,
            llm=llm,
            verbose=verbose,
            prompt=PromptTemplate.from_template(
                dedent(
                    """\
                    ### SYSTEM:
                    {context}

                    ### INSTRUCTIONS:
                    Given the memories below, accurately compress your memories for
                    long-term storage (maintaining important content) with one
                    compression per line.

                    {memories}

                    ### RESPONSE:
                    """
                )
            ),
        )


class MemoryImportance(MemoryParser):
    @classmethod
    def from_llm(cls, llm: vicuna, verbose: bool = True, **kwargs) -> LLMChain:
        return cls(
            **kwargs,
            llm=llm,
            verbose=verbose,
            prompt=PromptTemplate.from_template(
                dedent(
                    """\
                    ### SYSTEM:
                    {context}

                    On the scale of 1 to 10, where 1 is purely mundane (e.g., brushing 
                    teeth, making bed) and 10 is extremely poignant (e.g., a break up, 
                    college acceptance), rate the poignancy of the following memory:

                    {memory_fragment}

                    ### INSTRUCTION: 
                    Return a number 1-10 and a single sentence explanation. 

                    ### RESPONSE:
                    """
                )
            ),
        )


class MemoryReflect(MemoryParser):
    @classmethod
    def from_llm(cls, llm: vicuna, verbose: bool = True, **kwargs) -> LLMChain:
        return cls(
            **kwargs,
            llm=llm,
            verbose=verbose,
            prompt=PromptTemplate.from_template(
                dedent(
                    """\
                    ### SYSTEM:
                    {recent_memories}

                    ### INSTRUCTIONS:

                    Given your recent memories above, what are 3 most salient high-level
                    questions we can answer about the subjects in the statements?

                    ### RESPONSE:
                    """
                )
            ),
        )


class AgentSummary(MemoryParser):
    @classmethod
    def from_llm(cls, llm: vicuna, verbose: bool = True, **kwargs) -> LLMChain:
        return cls(
            **kwargs,
            llm=llm,
            verbose=verbose,
            prompt=PromptTemplate.from_template(
                dedent(
                    """\
                    Summarize {name}'s core characteristics based on the following:

                    {memories}                     

                    Be concise and avoid embellishment.

                    Summary: <fill in> \
                    """
                )
            ),
        )


# class EntityObserved(MemoryParser):
#     @classmethod
#     def from_llm(cls, llm: vicuna, verbose: bool = True, **kwargs) -> LLMChain:
#         return cls(
#             **kwargs,
#             llm=llm,
#             verbose=verbose,
#             prompt=PromptTemplate.from_template(
#                 dedent(
#                     """\
#                     What is the observed entity in the following: {observation}

#                     Entity: <fill in> \
#                     """
#                 )
#             ),
#         )


# class EntityAction(MemoryParser):
#     @classmethod
#     def from_llm(cls, llm: vicuna, verbose: bool = True, **kwargs) -> LLMChain:
#         return cls(
#             **kwargs,
#             llm=llm,
#             verbose=verbose,
#             prompt=PromptTemplate.from_template(
#                 dedent(
#                     """\
#                     What action is the {entity} taking in the following: {observation}

#                     The {entity} is: <fill in> \
#                     """
#                 )
#             ),
#         )
