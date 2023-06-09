from langchain.embeddings import LlamaCppEmbeddings

from agent.agent import Agent
from models.local_llamas import vicuna


llm = vicuna()
embeddings = LlamaCppEmbeddings(
    model_path="../llama.cpp/models/wizard-vicuna-13B.ggmlv3.q4_0.bin"
)

agent = Agent(
    name="Bob",
    description="A helpful young man who's good at Math proofs.",
    llm=llm,
)

observations = [
    "Bob remembers his dog, Bruno, from when he was a kid",
    "Bob sees the new home his wife bought for them",
]
for observation in observations:
    agent.add_memory(observation)

context = agent.get_context()
