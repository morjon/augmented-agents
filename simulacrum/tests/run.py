from utils.system import Simulation


sim = Simulation()
sim.setup()

sim.create_world(
    world_file_path="/home/ubuntu/repos/augmented-agents/simulacrum/world/small_sandbox.json"
)
sim.test_planning()
