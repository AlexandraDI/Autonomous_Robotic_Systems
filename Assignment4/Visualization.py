"""
Visualization for the simulation

Implemented by Alexandra Gianzina
"""
import pygame
import numpy as np

from Actions import Action, ActionTypes
from Robot import Robot
from Environment import Environment
from Simulation import Simulation
from RobotController import Controller
from typing import Optional
from time import time
import DustyEnvironments

WHITE = pygame.Color(255, 255, 255)
BLACK = pygame.Color(0, 0, 0)
RED = pygame.Color(255, 0, 0)
BLUE = pygame.Color(0, 0, 255)
GREEN = pygame.Color(218, 221, 152)
DARK_GREY = pygame.Color(52, 52, 52)


class Visualization:
    """
    Args:
        simulation: simulation to run and display
        minimum_update: mimimum time between two updates of the simulation
    """

    def __init__(
        self,
        simulation: Simulation,
        minimum_update: float = 0.1,
        controller: Optional[Controller] = None,
    ) -> None:
        self.running = False
        self.simulation = simulation
        self._minimum_dt = minimum_update
        self._screen = None
        self._font = None
        self.controller = controller

    def start(self, blocking: bool = False) -> None:
        """
        Start the visualization and show the screen.

        Args:
            blocking: if true block the execution until the pygame window is closed.
                If false, the method Visualization.loop has to be run until
                Visualization.running is false

        Returns:
            None.

        """
        pygame.init()
        self.running = True

        self._screen = pygame.display.set_mode((700, 700))
        self._screen.fill(WHITE)
        pygame.display.set_caption("Robot Simulation")

        self._font = pygame.font.SysFont("arial", 15)

        # If blocking call loop here
        # Otherwise the event loop is called externally
        dt = self._minimum_dt
        if blocking:
            while self.running:
                loop_start = time()
                self.loop(dt)
                dt = time() - loop_start
            self.close()

    def close(self) -> None:
        """
        Handle the pygame.QUIT event

        Returns:
            None.
        """
        pygame.quit()

    def visualize_environment(self) -> None:
        """
        Visualize the environment
        """
        for wall in self.simulation.environment.wall_points:
            pygame.draw.aaline(self._screen, BLACK, wall[0], wall[1])

    def visualize_robot(self):
        """
        Visualize the robot and the sensor values.

        Returns:
            None.
        """

        for robot in self.simulation.robots:
            # Robot shape
            pygame.draw.circle(self._screen, RED, robot.position, robot.radius, width=2)
            # Robot direction
            pygame.draw.line(
                self._screen,
                RED,
                robot.position,
                robot.position
                + (
                    np.cos(robot.angle) * robot.radius,
                    np.sin(robot.angle) * robot.radius,
                ),
                width=2,
            )
            # for angle, value in zip(robot.sensors, robot.sensor_values):
            #     text = self._font.render(str(int(value)), True, (0, 0, 0))
            #     s_dir = np.array(
            #         [np.cos(robot.angle + angle), np.sin(robot.angle + angle)]
            #     )
            #     self._screen.blit(
            #         text,
            #         robot.position
            #         + s_dir * robot.radius * 3 * min(-0.5, np.sin(0.5 * (angle)))
            #     )

            n_sensors = len(robot.sensors)
            for i, x in enumerate(robot.sensor_values):
                text = self._font.render(str(int(x)), True, (0, 0, 0))
                self._screen.blit(
                    text,
                    robot.position
                    + (
                        np.cos(robot.angle + (np.pi / n_sensors * 2) * i) * robot.radius
                        - robot.radius / n_sensors * 2,
                        np.sin(robot.angle + (np.pi / n_sensors * 2) * i) * robot.radius
                        - robot.radius / n_sensors * 2,
                    ),
                )
            self._screen.blit(
                self._font.render(str(int(robot.v_left)), True, (0, 0, 0)),
                (robot.position[0], robot.position[1] - robot.radius / 2),
            )
            self._screen.blit(
                self._font.render(str(int(robot.v_right)), True, (0, 0, 0)),
                (robot.position[0], robot.position[1]),
            )

    def handle_events(self, dt: Optional[float] = None):
        """
        Handle the pygame events and pass the inputs to the simulation.

        Args:
            dt: time since last update
        Returns:
            None.

        """
        if self.controller is None:
            dt = dt if dt is not None else self._minimum_dt
            action = None
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_w:
                        action = Action(0, ActionTypes.L_POS)
                    elif event.key == pygame.K_s:
                        action = Action(0, ActionTypes.L_NEG)
                    elif event.key == pygame.K_t:
                        action = Action(0, ActionTypes.POS)
                    elif event.key == pygame.K_g:
                        action = Action(0, ActionTypes.NEG)
                    elif event.key == pygame.K_o:
                        action = Action(0, ActionTypes.R_POS)
                    elif event.key == pygame.K_l:
                        action = Action(0, ActionTypes.R_NEG)
                    elif event.key == pygame.K_x:
                        action = Action(0, ActionTypes.ZERO)
            if action is not None:
                self.simulation.apply_action(action, dt)
        else:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            robot = self.simulation.robots[0]
            robot.v_left, robot.v_right = self.controller(robot.sensor_values)
        self._screen.fill(WHITE)

    def __enter__(self):
        """
        Context manager (with-statement). Start the visualization

        Returns:
            None.

        """
        self.start(blocking=False)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager (with-statement). End the visualization

        """
        self.close()

    def loop(self, delta_time: Optional[float] = None):
        """
        Run an iteration of the simulation and update the visualization.

        Args:
            delta_time: time elapsed since last update.

        Returns:
            None.

        """
        # Event loop, single iteration
        dt = delta_time if delta_time is not None else self._minimum_dt
        self.visualize_environment()
        self.visualize_robot()
        pygame.display.update()
        self.handle_events(dt)

        self.simulation.update(dt)


if __name__ == "__main__":
    print("Starting the visualization")

    robot = Robot(
        position=np.array([250, 150, np.pi * 2]),
        radius=30,
        maximum_value_sensor=10000,
        acceleration=1000,
        sensors=Robot.make_sensors(12),
    )
    np.random.seed(37)
    # add a random number of inner walls

    number_of_inner_walls = 7
    inner_walls = [
        (
            tuple(np.random.randint(50, 650, size=2)),
            tuple(np.random.randint(50, 650, size=2)),
        )
        for i in range(number_of_inner_walls)
    ]
    outer_walls = [
        ((50, 50), (50, 650)),
        ((50, 650), (650, 650)),
        ((650, 650), (650, 50)),
        ((650, 50), (50, 50)),
    ]

    # environment = Environment(outer_walls + inner_walls)
    environment = DustyEnvironments.dusty_room
    simulation = Simulation(environment, [robot], 0.1)
    # with Visualization(simulation) as visualization:
    #     dt = 0.1
    #     while visualization.running:
    #         loop_start = time()
    #         visualization.loop(dt)
    #         dt = time() - loop_start

    dt = 0.3  # 300 ms
    length = 120  # 2 minutes
    controller = Controller(np.random.rand(Controller.weight_size(),) - 0.5)
    visualization = Visualization(simulation, minimum_update=dt, controller=controller)
    visualization.start(False)
    for i in range(int(length / dt)):
        visualization.loop(dt)
        print(controller._output)
    visualization.close()
