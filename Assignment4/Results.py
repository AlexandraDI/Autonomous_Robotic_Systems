"""
Script to show the results: it shows the best individual on different maps

Methods:
    * run_robot: show a demo of a robot navigating a map.

Constants:
    * FILENAME: file to visualize
"""
from time import time, sleep
from typing import Optional
from GE import GE
from Robot import Robot
import DustyEnvironments
from RobotController import Controller
from Simulation import Simulation
from Visualization import Visualization
from Plotting import Plotting

# FILENAME = "2022-03-06-182240_200.pkl"
FILENAME = "2022-03-07-001338_200.pkl"
# FILENAME = "2022-03-07-000208_1.pkl"


def run_robot(
    map_: "Environment",
    controller: Controller,
    position: "np.array",
    radius: float,
    dt: Optional[float] = 0.3,
    length: Optional[float] = 120,
    speedup: Optional[float] = 20,
) -> None:
    """
    Demo of the robot

    Args:
        map_: environment used for testing
        controller: controller of the robot
        position: starting position as (x, y, angle)
        radius: radius of the robot
        dt: time step of the simulation in seconds
        length: length of the simulation in seconds
        speedup: speed up the visualization by this factor
    """
    robot = Robot(
        position=position,
        radius=radius,
        maximum_value_sensor=200,
        acceleration=1,
        sensors=Robot.make_sensors(12),
    )

    simulation = Simulation(map_, [robot], dt)

    visualization = Visualization(simulation, minimum_update=dt, controller=controller)
    visualization.start(False)
    last = time()
    for i in range(int(length / dt)):
        visualization.loop(dt)
        sleep((dt - time() + last) / speedup)
        last = time()
        if not visualization.running:
            break
    visualization.close()


if __name__ == "__main__":
    ge, fitness, diversity, best = GE.load(FILENAME)
    fitness = [i for i in fitness if i is not None]
    diversity = [i for i in diversity if i is not None]
    best = [i for i in best if i is not None]

    print("Running the robot controller in different maps")

    print("MAP 1: square (used for training)")
    controller = Controller(best[-1])
    run_robot(
        DustyEnvironments.dusty_square,
        controller,
        DustyEnvironments.dusty_square_positions[0].copy(),
        DustyEnvironments.dusty_square_radius,
    )
    input("Press ENTER to continue")

    print("MAP 2: double square (used for training)")
    controller = Controller(best[-1])
    run_robot(
        DustyEnvironments.dusty_double_square,
        controller,
        DustyEnvironments.dusty_double_square_positions[0].copy(),
        DustyEnvironments.dusty_double_square_radius,
    )
    input("Press ENTER to continue")

    print("MAP 3: room with acute angles")  # Collision problem
    controller = Controller(best[-1])
    run_robot(
        DustyEnvironments.dusty_angles,
        controller,
        DustyEnvironments.dusty_angles_positions[1].copy(),
        DustyEnvironments.dusty_angles_radius,
    )
    input("Press ENTER to continue")

    print("MAP 4: complex room")
    controller = Controller(best[-1])
    run_robot(
        DustyEnvironments.dusty_room,
        controller,
        DustyEnvironments.dusty_room_positions[0].copy(),
        DustyEnvironments.dusty_room_radius,
    )
    input("Press ENTER to continue")
    ###########################################################################
    print("Fitness and diversity during the training")

    plot = Plotting()

    plot.show(diversity, fitness, best, "Robot simulation")
