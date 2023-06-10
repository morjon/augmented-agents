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
    traits=["comedian"],
    llm=llm,
    verbose=True,
    callback_manager=manager,
)

observations = [
    "Bob sees Katie, who is his rival in Chess.",
    "Bob's cat is missing.",
]
for observation in observations:
    agent.add_memory(observation)

context = agent.get_context()
