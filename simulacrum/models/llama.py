from pydantic import Field
from typing import Any, Optional

from langchain.llms import LlamaCpp


class llama(LlamaCpp):
    model_path: Optional[str] = "../llama.cpp/models/13B/ggml-model-q4_0.bin"
    temperature: Optional[float] = 0.8
    max_tokens: Optional[int] = 150
    n_ctx: int = Field(2000, alias="n_ctx")

    n_gpu_layers: Optional[int] = Field(32, alias="n_gpu_layers")

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    @property
    def _llm_type(self) -> str:
        """Return type of local llm."""
        return "local llama"

    def get_model_path(self) -> Optional[str]:
        return self.model_path
