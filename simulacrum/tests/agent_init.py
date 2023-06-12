from agent.agent import Agent
from memory.stream import AgentMemory
from langchain.llms import OpenAI
from models.local_llamas import vicuna
from utils.callbacks import ConsoleManager


llm = OpenAI(model_name="text-davinci-003")  #vicuna()
memory = AgentMemory()
manager = ConsoleManager([])

bob = Agent(
    name="Bob",
    description="A troubled youth who hates the world.",
    traits=["comedian", "sarcastic"],
    llm=llm,
    verbose=True,
    callback_manager=manager,
)

# alice = Agent(
#     name="Alice",
#     description="A tech-savvy millennial who loves social media.",
#     traits=["organized", "outgoing"],
#     llm=llm,
#     verbose=True,
#     callback_manager=manager,
# )

# charlie = Agent(
#     name="Charlie",
#     description="A retired army veteran with a heart of gold.",
#     traits=["loyal", "brave"],
#     llm=llm,
#     verbose=True,
#     callback_manager=manager,
# )


observations = [
    "Bob sees Katie, who is his rival in Checkers",
    # "Bob's cat is missing",
    # "Bob just got a new phone",
    # "Bob is excited to go to a party tonight",
    # "Bob got told to stop being a phony",
]
for observation in observations:
    bob.add_memory(observation)

context = bob.get_context()

# observations = [
#     [
#         "Bob sees Katie, who is his rival in Checkers.",
#         "Bob's cat is missing.",
#     ],
#     [
#         "Alice just got a new phone.",
#         "Alice is excited to go to a party tonight.",
#     ],
#     [
#         "Charlie visits the local VA hospital.",
#         "Charlie's granddaughter just got accepted into college.",
#     ],
# ]

# for i, agent in enumerate([bob, alice, charlie]):
#     for observation in observations[i]:
#         agent.add_memory(observation)

# contexts = []
# for agent in [bob, alice, charlie]:
#     contexts.append(agent.get_context())
