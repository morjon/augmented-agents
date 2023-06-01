from typing import Any, Dict, List

# from local_models.llama import llama

from langchain import LLMChain

import re


class parsed_list(LLMChain):
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
