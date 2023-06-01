from typing import Any, Optional
from pydantic import Field
from langchain.llms import LlamaCpp


class llama(LlamaCpp):
    model_path: Optional[str] = "../../llama.cpp/models/13B/ggml-model-q4_0.bin"
    temperature: Optional[float] = 0.85
    max_tokens: Optional[int] = 150
    n_ctx: int = Field(2048, alias="n_ctx")

    # consider the following closely...
    n_gpu_layers: Optional[int] = Field(32, alias="n_gpu_layers")
    # n_threads: Optional[int] = Field(4, alias="n_threads")
    # n_batch: Optional[int] = Field(4, alias="n_batch")

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    @property
    def _llm_type(self) -> str:
        """Return type of local llm."""
        return "local llama"

    def generate(
        self,
        prompt: str,
    ) -> str:
        return self._call(
            prompt,
        )


if __name__ == "__main__":
    llm = llama()
    prompt = "Why don't you tell me about your day?"
    response = llm.generate(prompt)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print(llm._get_parameters)
