from better_sim import Agent
from better_sim.local_models import llama

agent = Agent(
    name="Bob",
    description="A helpful assistant",
    llm=llama(),
    verbose=False,
    max_token_limit=1000,
    reflection_threshold=0.6,
)

observations = [
    "Bob remembers his dog, Bruno, from when he was a kid",
    "Bob feels tired from driving so far",
    "Bob sees the new home",
    "The new neighbors have a cat",
    "The road is noisy at night",
    "Bob is hungry",
    "Bob tries to get some rest.",
]
for observation in observations:
    agent.memory.add_memory(observation)

summary = agent.get_summary(num_memories=3)
print(summary)

summary = agent.get_summary(num_memories=10)
print(summary)
