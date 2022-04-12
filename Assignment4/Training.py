"""
Script to run the GE to train the robots.

Methods:
    * make_name: generate a name when saving the dats
    * robot_fitness: compute the fitness of the robots

Constants:
    * MAPS: list of environmnents used for the training
    * SIZE: list of radius of the robot for each environmnet
    * POSITION: starting position of the robots for each environment

"""
from typing import List, Optional
from DustyEnvironments import (
    dusty_square,
    dusty_square_radius,
    dusty_square_positions,
    dusty_double_square,
    dusty_double_square_radius,
    dusty_double_square_positions,
    dusty_room,
    dusty_room_radius,
    dusty_room_positions,
    dusty_angles,
    dusty_angles_radius,
    dusty_angles_positions,
    shift,
)
from RobotController import Controller
from Robot import Robot
from Simulation import Simulation
import numpy as np
from datetime import datetime

from GE import (
    GESettings,
    GE,
    mutation,
    tournament_selection,
    distance,
    rank_selection,
    one_point_crossover,
)

MAPS = [dusty_square, dusty_double_square]


SIZES = [
    dusty_square_radius,
    dusty_double_square_radius,
    dusty_angles_radius,
]

POSITIONS = [
    dusty_square_positions[:1],
    dusty_double_square_positions[:1],
    dusty_angles_positions[:1],
]


def make_name(start_time: datetime, generation: int) -> str:
    """
    Args:
        datetime: when the GE started
        generation: current generation number

    Returns:
        Name of the file to save

    """
    return f"{str(start_time).replace(' ', '-').split('.')[0]}_{generation}.pkl"


def robot_crossover(
    population: List[np.array],
    fitness: List[float],
    rank: List[int],
    rng,
    input: Optional[int] = 15,
    hidden: Optional[int] = 5,
    output: Optional[int] = 2,
    *args: List,
) -> List[np.array]:
    """
    Crossover. Split the genome on a point and exchange the genome between two
    individuals

    Args:
        population: population of the GE
        fitness: fitness of each individual
        rank: rank of each individual
        rng: random number generator
        input: size of the input layer (including bias)
        hidden: size of the hidden layer (including bias)
        output: size of the output layer
        *args: other args

    Returns:
        New population
    """
    if len(population) == 0:
        return []

    pairs = rng.choice(rank, (len(rank) // 2, 2), False)
    new_population = [None for i in range(len(pairs) * 2)]

    arch = [input, hidden, output]
    w1 = input * (hidden - 1)
    for i, (i1, i2) in enumerate(pairs):
        layer = rng.randint(0, 3)
        neuron = rng.randint(0, arch[layer])

        new_1 = population[i1].copy()
        new_2 = population[i2].copy()
        if layer == 0:
            # Weights of the first layer
            # Select the neuron
            new_1[neuron:w1:input] = population[i2][neuron:w1:input]
            new_2[neuron:w1:input] = population[i1][neuron:w1:input]
        elif layer == 1:
            # Select the weights from the hidden layer to the output layer
            new_1[w1 + neuron :: hidden] = population[i2][w1 + neuron :: hidden]
            new_2[w1 + neuron :: hidden] = population[i1][w1 + neuron :: hidden]
            if neuron != 0:  # not bias
                # Select also the weights from the input layer
                new_1[(neuron - 1) * input : neuron * input] = population[i2][
                    (neuron - 1) * input : neuron * input
                ]
                new_2[(neuron - 1) * input : neuron * input] = population[i1][
                    (neuron - 1) * input : neuron * input
                ]
        else:
            # Select the weights from the hidden layer to the output layers
            new_1[w1 + neuron * hidden : w1 + (neuron + 1) * hidden] = population[i2][
                w1 + neuron * hidden : w1 + (neuron + 1) * hidden
            ]
            new_2[w1 + neuron * hidden : w1 + (neuron + 1) * hidden] = population[i1][
                w1 + neuron * hidden : w1 + (neuron + 1) * hidden
            ]

        new_population[i * 2] = new_1
        new_population[i * 2 + 1] = new_2
    return new_population


def _fitness(
    controllers: List[Controller],
    map_: "Environment",
    dt: float,
    length: float,
    size: float,
    position: np.array,
    max_sensor: float,
    max_speed: float,
):
    """
    Run a simgle experiment to compute the fitness of the robots

    Args:
        controllers: list of robot controllers to test
        map_: environment used for the test
        dt: time steps for the simulation
        length: length of the simulation
        size: radius of the robots
        position: starting position of the robots
        max_sensor: maximum distance of the sensors
        max_speed: maximum output of the robot controller

    Returns:
        Fitness of the robots

    """
    steps = int(length / dt)
    robots = [
        Robot(
            position=position.copy(),
            radius=size,
            maximum_value_sensor=max_sensor,
            acceleration=1,  # Not used
            sensors=Robot.make_sensors(12),
            trace=True,
            trace_size=steps,
        )
        for i in controllers
    ]

    # Simulation
    simulation = Simulation(map_, robots, dt)
    for i in range(steps):
        # Update simulation
        simulation.update(dt)

        # Controller acts
        for robot, controller in zip(robots, controllers):
            robot.v_left, robot.v_right = controller(robot.sensor_values)
    # Compute surface/dust covered
    surface = np.array(
        [_calculate_the_dust(robot.trace, robot.radius) for robot in robots]
    )
    collisions = np.array([r.collisions for r in robots])

    # Fitness based on surface covered and collisions
    # TODO use a better formula
    print("Collisions:", collisions)
    return surface - collisions


def _calculate_the_dust(
    trace: List[np.array], radius: float, map_size: int = 600
) -> int:
    """
    Calculate the amount of dust that was cleaned

    Args:
        trace: list of positions of the robot
        radius: radius of the robot
        map_size: size of the square bounding the room
    Returns:
        Surface covered / dust collected
    """

    dust = np.zeros((map_size + 1, map_size + 1), dtype=bool)
    trace = [shift(i) for i in trace if i is not None]

    rad_sqr = radius ** 2
    for i, t in enumerate(trace):
        x_center, y_center = np.rint(t)
        x_center = int(x_center)
        y_center = int(y_center)
        # if (np.ceil(t - radius) < ).any() or (np.floor(t + radius) > map_size).any():
        if (t < 0).any() or (t > map_size).any():
            print("The robot crossed the outer walls:", i, t)
            return 0
        else:
            # Walk on the diamater aligned with one axis
            # Compute the "column" of filled pixels on the point
            # Fill the column

            for x in range(0, radius + 1):
                column = int(np.rint(np.sqrt(rad_sqr - x ** 2)))

                dust[
                    np.clip(x + x_center, 0, map_size),
                    np.clip(y_center - column, 0, map_size) : np.clip(
                        y_center + column, 0, map_size
                    ),
                ] = True
                dust[
                    np.clip(-x + x_center, 0, map_size),
                    np.clip(y_center - column, 0, map_size) : np.clip(
                        y_center + column, 0, map_size
                    ),
                ] = True

    return np.sum(dust)


def robot_fitness(
    population: List[np.array],
    rng,
    dt: Optional[float] = 0.3,
    length: Optional[float] = 300,
    maps: Optional[List["Environment"]] = MAPS,
    sizes: Optional[List[float]] = SIZES,
    positions: Optional[List[List[np.array]]] = POSITIONS,
    max_sensor: Optional[float] = 200,
    max_speed: Optional[float] = 20,
) -> np.array:
    """
    Compute the average fitness of the entire population on multiple maps.

    Args:
        population: population to test
        rng: random number generator
        dt: time step of the simulation
        length: length of the simulation
        maps: environment to test
        sizes: radius of the robot for each environment
        positions: list of starting positions (x, y) for each environemnt
        max_sensor: maximum value of the sensors
        max_speed: maximum output of the controller
    Returns:
        Fitness value of each robot.

    """
    controllers = Controller.from_population(population, sensors=12, clip=max_speed)
    experiments = 0
    avg_fitness = np.ones(len(population))
    for i, map_ in enumerate(maps):
        print("Map:", i + 1)
        for position in positions[i]:
            experiments += 1
            avg_fitness += _fitness(
                controllers, map_, dt, length, SIZES[i], position, max_sensor, max_speed
            )
    if experiments == 0:
        return avg_fitness
    return avg_fitness / experiments


# TODO move this part in another file
if __name__ == "__main__":

    DT = 0.3
    LENGTH = 120
    MAX_SPEED = 150

    setting = GESettings(
        population_size=30,
        mutation_rate=0.05,
        crossover=robot_crossover,
        mutation=mutation,
        selection=tournament_selection,
        individual_generator=lambda rng: rng.rand(Controller.weight_size()) - 0.5,
        fitness_function=lambda pop, rng, *args: robot_fitness(
            pop, rng, dt=DT, length=LENGTH, max_speed=MAX_SPEED
        ),
        distance_function=distance,
        individual_fitness=False,
        keep_best=1,
    )
    ge = GE(setting)

    generations = 200
    fitness = [None for i in range(generations)]
    diversity = [None for i in range(generations)]
    best = [None for i in range(generations)]

    start_time = datetime.now()
    print("Start at:", start_time)
    for i in range(1, generations + 1):
        ge.iterate()
        fitness[i - 1] = ge.fitness
        diversity[i - 1] = ge.get_diversity()
        best[i - 1] = ge.get_best()
        print("-" * 80)
        print("Generation:", i)
        print("Time:", datetime.now(), "- Elapsed:", datetime.now() - start_time)
        print(
            "Estimated end at:",
            start_time + (datetime.now() - start_time) / i * generations,
        )
        print("Max. fitness:", np.max(fitness[i - 1]))
        print("Avg. fitness:", np.average(fitness[i - 1]))
        print("Fitness:", fitness[i - 1])
        print("Diversity:", diversity[i - 1])
        print("-" * 80)
        if i % 50 == 0:
            print("Saving...")
            ge.save(
                make_name(str(start_time).replace(":", ""), i),
                [fitness, diversity, best],
            )
    end_time = datetime.now()
    print("End at:", end_time)
    print("Elapsed:", end_time - start_time)
    print("Avg. time per generation:", (end_time - start_time) / generations)
