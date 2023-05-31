from typing import Any, List, Optional
from langchain.llms.base import LlamaCpp


class local_llama(LlamaCpp):
    def __init__(self, model_path: str, **kwargs: Any):
        super().__init__(model_path=model_path, **kwargs)

    def generate_agent_text(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None,
    ) -> str:
        return self._call(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )
