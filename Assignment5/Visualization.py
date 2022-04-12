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
from KalmanFilter import Kalman, R_LARGE, R_SMALL, Q_LARGE, Q_SMALL
from typing import Optional, List, Tuple
from time import time, sleep
from collections import deque

WHITE = pygame.Color(255, 255, 255)
BLACK = pygame.Color(0, 0, 0)
RED = pygame.Color(255, 0, 0)
ORANGE = pygame.Color(255, 165, 0)
BLUE = pygame.Color(0, 0, 150)
LIGHT_BLUE = pygame.Color(20, 20, 150)
GREEN = pygame.Color(218, 221, 152)
BRIGHT_GREEN = pygame.Color(0, 255, 0)
DARK_GREEN = pygame.Color(0, 128, 128)
DARK_GREY = pygame.Color(52, 52, 52)
LIGHT_GREY = pygame.Color(150, 150, 150)


class Visualization:
    """
    Args:
        simulation: simulation to run and display
        minimum_update: mimimum time between two updates of the simulation
        controllers: list of controllers for the robors
        filters: filters to estimate the robot pose over time
        history_size: size of the history of poses used to draw the trajectory
    """

    def __init__(
        self,
        simulation: Simulation,
        minimum_update: float = 0.1,
        controllers: Optional[List["Controller"]] = None,
        filters: Optional[List[Kalman]] = None,
        history_size: int = int(1e6),
        ellipsis_scale: Tuple[float, float] = (1, 20),
    ) -> None:
        self.running = False
        self.simulation = simulation
        self._minimum_dt = minimum_update
        self._screen = None
        self._font = None
        self.controllers = controllers
        self.filters = filters
        self._trajectory = [deque(maxlen=history_size) for i in self.simulation.robots]
        self._est_pose_history = [
            deque(maxlen=history_size) for i in self.simulation.robots
        ]
        self._est_covariance_history = [
            deque(maxlen=history_size) for i in self.simulation.robots
        ]

        self._scales = ellipsis_scale

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

        for landmark in self.simulation.environment.landmarks:
            pygame.draw.circle(self._screen, DARK_GREEN, landmark, 4)

    def visualize_robot(self) -> None:
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

            # Robot sensors
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

            # Robot and sensed landmarks
            for beacon in robot.beacons:
                pygame.draw.aaline(
                    self._screen,
                    DARK_GREEN,
                    self.simulation.environment.landmarks[int(beacon[2])],
                    robot.position,
                )

    def visualize_trajectory(self) -> None:
        """
        Visualize the robot's trajectory
        """
        for i in range(len(self.simulation.robots)):
            for point in self._trajectory[i]:
                pygame.draw.circle(self._screen, BLUE, point, 1)

            for n, (pose, cov) in enumerate(
                zip(self._est_pose_history[i], self._est_covariance_history[i])
            ):
                pygame.draw.circle(self._screen, LIGHT_GREY, pose[:2], 1)

                # There could be issues when the deque is filled
                if (n % 400) == 0 and n > 0:
                    std = np.sqrt(np.abs(cov))
                    # Position
                    std_pos = (std[:2, :2] * self._scales[0]).diagonal()
                    ellipse_pos = pygame.Surface(tuple(std_pos), pygame.SRCALPHA)

                    pygame.draw.ellipse(ellipse_pos, DARK_GREEN, (0, 0, *std_pos), 1)

                    # ellipse_pos = pygame.transform.rotate(ellipse_pos, rot)

                    # Angle
                    std_angle = (
                        np.array([np.cos(std[2, 2]), np.sin(std[2, 2])])
                        * self._scales[1]
                    )
                    ellipse_angle = pygame.Surface(
                        tuple(np.abs(std_angle)), pygame.SRCALPHA
                    )

                    pygame.draw.ellipse(ellipse_angle, BLUE, (0, 0, *std_angle), 1)
                    # ellipse_angle = pygame.transform.rotate(
                    #     ellipse_angle, rot)
                    # )

                    self._screen.blits(
                        [
                            (ellipse_pos, ellipse_pos.get_rect(center=pose[:2]),),
                            (ellipse_angle, ellipse_angle.get_rect(center=pose[:2])),
                        ]
                    )

    def handle_events(self, dt: Optional[float] = None) -> None:
        """
        Handle the pygame events and pass the inputs to the simulation.

        Args:
            dt: time since last update
        Returns:
            None.

        """
        if self.controllers is None:
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
            for robot in enumerate(self.simulation.robots):
                robot.v_left, robot.v_right = self.controllers[i](robot.sensor_values)
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

        # Draw
        self.visualize_environment()
        self.visualize_trajectory()
        self.visualize_robot()
        pygame.display.update()

        # Events
        self.handle_events(dt)

        # Update trajectories
        for i, robot in enumerate(self.simulation.robots):
            # TODO Kalman should not update at every iteration
            est_pose, est_covariance = self.filters[i].estimate(
                robot.compute_motion(dt), robot.estimated_position, dt,
            )
            self._est_pose_history[i].append(est_pose.copy())
            self._est_covariance_history[i].append(est_covariance.copy())

            if (
                len(self._trajectory[i]) == 0
                or (self._trajectory[-1] != robot.position).any()
            ):
                self._trajectory[i].append(robot.position)

        # Move the robot
        self.simulation.update(dt)


if __name__ == "__main__":
    np.random.seed(37)

    outer_walls = [
        ((50, 50), (50, 650)),
        ((50, 650), (650, 650)),
        ((650, 650), (650, 50)),
        ((650, 50), (50, 50)),
    ]
    inner_walls = [
        ((350, 650), (350, 150)),
        ((50, 550), (250, 550)),
        ((150, 450), (350, 450)),
        ((50, 350), (250, 350)),
        ((150, 150), (350, 150)),
    ]
    landmarks = [
        (350, 650),
        (250, 550),
        (150, 450),
        (350, 450),
        (50, 350),
        (250, 350),
        (150, 250),
        (150, 150),
        (350, 150),
    ]
    environment = Environment(outer_walls + inner_walls, landmarks)

    robot = Robot(
        position=np.array([100, 600, np.pi * 2]),
        radius=20,
        maximum_value_sensor=150,
        acceleration=1000,
        sensors=Robot.make_sensors(0),
        map_=environment,
        rng=np.random,
        noise=(np.zeros((3,)), np.eye(3) * 2),
    )

    # filter_ = Kalman(robot.pose, R_SMALL, Q_SMALL)
    # filter_ = Kalman(robot.pose, R_LARGE, Q_SMALL)
    # filter_ = Kalman(robot.pose, R_SMALL, Q_LARGE)
    filter_ = Kalman(robot.pose, R_LARGE, Q_LARGE)

    # environment = Environment(outer_walls + inner_walls)
    simulation = Simulation(environment, [robot], 0.1)
    with Visualization(simulation, filters=[filter_]) as visualization:
        dt = 0.1
        constant = 0.01
        i = 0
        while visualization.running:
            loop_start = time()
            visualization.loop(dt)
            dt = time() - loop_start
            dt = max(dt, constant)
            i += 1
