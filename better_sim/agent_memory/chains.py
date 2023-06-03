from local_models.llama import llama
from typing import Any, Dict, List
from textwrap import dedent

from langchain import LLMChain, PromptTemplate

import re
import json


class MemoryParser(LLMChain):
    key = "items"

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, List[str]]:
        text = super._call(inputs)[self.key].strip()

        items = [
            re.sub(r"^\s*\d+\.\s*", "", line).strip()
            for line in re.split(r"\n", text.strip())
        ]
        return {"items": items}

    def run(self, **kwargs) -> Dict[str, List[str]]:
        return self._call(inputs=kwargs)[self.key]


class MemoryJSONParser(LLMChain):
    key = "json"

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        data = json.loads(super()._call(inputs)["json"].strip())
        return {"json": data}


class Compress(MemoryParser):
    @classmethod
    def from_llm(cls, llm: llama, verbose: bool = True, **kwargs) -> LLMChain:
        return cls(
            **kwargs,
            llm=llm,
            verbose=verbose,
            prompt=PromptTemplate.from_tempalte(
                dedent(
                    """\
                    {context}

                    These are your recent memories: 
                    {memories}

                    Given the information above, accurately compress your memories for
                    long-term storage, with 1 compression per line.
                    """
                )
            )
        )


class Reflect(MemoryParser):
    @classmethod
    def from_llm(cls, llm: llama, verbose: bool = True, **kwargs) -> LLMChain:
        return cls(
            **kwargs,
            llm=llm,
            verbose=verbose,
            prompt=PromptTemplate.from_template(
                dedent(
                    """\
                    {statements}

                    Given only the information above, what are 3 most salient high-level
                    questions we can answer about the subjects in the statements?
                    """
                )
            )
        )


class Importance(MemoryParser):
    @classmethod
    def from_llm(cls, llm: llama, verbose: bool = True, **kwargs) -> LLMChain:
        return cls(
            **kwargs,
            llm=llm,
            verbose=verbose,
            prompt=PromptTemplate.from_template(
                dedent(
                    """\
                    On the scale of 1 to 10, where 1 is purely mundane (e.g., brushing 
                    teeth, making bed) and 10 is extremely poignant (e.g., a break up, 
                    college acceptance), rate the likely poignancy of the following 
                    piece of memory. 

                    Memory: {memory}

                    Rating: <fill in> \
                    """
                )
            )
        )
