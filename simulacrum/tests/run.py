from utils.system import Simulation

sim = Simulation()
sim.setup()

# sim.create_world(
#     world_file_path="/home/ubuntu/repos/augmented-agents/simulacrum/world/small_sandbox.json"
# )
# sim.test_planning()

bob = sim.create_agent(
    name="bob",
    description="bob is a smart, strong, and brave agent",
    traits=["smart", "strong", "brave"],
    location="0",
)

bob.get_plan()
print(bob.plan)
