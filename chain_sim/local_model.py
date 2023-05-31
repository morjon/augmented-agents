from typing import Any, List, Optional
from langchain.llms import LlamaCpp


class local_llama(LlamaCpp):
    def __init__(self, **kwargs: Any):
        model_path = "../llama.cpp/models/13B/ggml-model-q4_0.bin"
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


if __name__ == "__main__":
    llama = local_llama()
    prompt = "Hello, how are you today?"
    response = llama.generate_agent_text(prompt)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
