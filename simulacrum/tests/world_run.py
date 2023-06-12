from agent.agent import Agent
from world.locations import Locations
from models.local_llamas import vicuna
from utils.callbacks import ConsoleManager

import json
import networkx as nx

step_sim = 0
repeat_sim = 5

output_sim = ""

with open("/home/ubuntu/repos/augmented-agents/simulacrum/world/small_sandbox.json", "r") as f:
    world_data = json.load(f)

agent_data = world_data["agent_data"]
location_data = world_data["location_data"]

world = nx.Graph()
final_loc = None
for loc in location_data.keys():
        world.add_node(loc)
        world.add_edge(loc, loc) # add self edge
        if final_loc is not None:
                world.add_edge(loc, final_loc)
        final_loc = loc

world.add_edge(list(location_data.keys())[0], final_loc)


locations = Locations()
for name, description in location_data.items():
        locations.add_location(name, description)

agents = []
llm = vicuna()
callback_manager = ConsoleManager([])

for name, description in agent_data.items() :
        starting_location = locations.get_random_location()
        agents.append(
                Agent(
                        name=name, 
                        description=description, 
                        llm=llm, 
                        verbose=True, 
                        callback_manager=callback_manager,
                        world=world,
                        starting_location=starting_location,
                )
        ) 

# Run the simulation.
for i in range(repeat_sim):
        log_output = ""
        print(f"==================== Simulation {i} ====================\n")
        log_output += f"==================== Simulation {i} ====================\n"

        for agent in agents:
            agent.plan(step_sim)
            agent.get_context()

