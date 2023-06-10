from agent.agent import Agent
from memory.stream import AgentMemory
from models.local_llamas import vicuna
from utils.callbacks import ConsoleManager


llm = vicuna()
memory = AgentMemory()
manager = ConsoleManager([])

agent = Agent(
    name="Bob",
    description="A helpful young man who's good at Math proofs.",
    llm=llm,
    verbose=True,
    callback_manager=manager,
)

observations = [
    "Bob remembers his dog, Bruno, from when he was a kid",
    "Bob sees the new home his wife bought for them",
]
for observation in observations:
    agent.add_memory(observation)

context = agent.get_context()
