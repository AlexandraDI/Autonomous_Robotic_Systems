"""
This module contains the Environment of the Robot.

Implemented by Alexandra Gianzina
"""
from typing import List, Optional
import numpy as np


class Robot:
    """
    Robot class.

    Args:
        position: position of the robot as [x, y, rotation]
        radius: radius of the circular robot
        maximum_value_sensor: maximum distance that can be sensed by the robot
        acceleration: acceleration of the robot, used to change its velocity
        sensors: list of sensors. The sensors are define with the counter-clockwise
            angle (in radiants) w.r.t. the foward direction of the robot.
            E.g., [0, pi/2, pi, 3pi/2] represent 4 sensors: one on the front,
            one on the left, one on the right, and one on the back
        trace: keep history of positions
        trace_size: maximum size of the trace history
    """

    def __init__(
        self,
        position: np.array,
        radius: int,
        maximum_value_sensor: float,
        acceleration: float,
        sensors: List[float],
        trace: Optional[bool] = False,
        trace_size: Optional[int] = 0,
    ) -> None:
        self._position = position
        self.radius = radius
        self.v_left = 0
        self.v_right = 0
        self.maximum_value_sensor = maximum_value_sensor
        self.acceleration = acceleration
        self.sensors = sensors.copy()
        self.sensor_values = [self.maximum_value_sensor for i in self.sensors]
        self.collisions = 0

        self._keep_trace = trace
        if trace:
            self._trace = [None for i in range(trace_size + 1)]
            self._trace[0] = self.position.copy()
            self._iteration = 1

    def _calculate_omega(self) -> float:
        """
        Returns:
            Calculates and returns omega for the ICC formula

        """
        if self.v_right == self.v_left:
            return 0
        else:
            return (self.v_left - self.v_right) / (2 * self.radius)

    def _calculate_r_capital(self) -> float:
        """
        Returns:
            Calculates and returns R for the ICC formula

        """
        if self.v_right == self.v_left:
            # in case they have similar values
            # we should not divide by zero
            # so, we are adding a very small constant
            constant = 0.0001
            return (
                self.radius
                * (self.v_left + self.v_right)
                / (self.v_left - self.v_right + constant)
            )
        else:
            return (
                self.radius
                * (self.v_left + self.v_right)
                / (self.v_left - self.v_right)
            )

    def _calculate_icc(self, r_capital) -> np.array:
        """
        Returns:
            Calculates and returns ICC
        """
        return np.array(
            [
                self._position[0] - r_capital * np.sin(self._position[2]),
                self._position[1] + r_capital * np.cos(self._position[2]),
            ]
        )

    def _calculate_new_position(
        self, omega: float, delta_t: float, icc: np.array
    ) -> np.array:
        """
        Args:
            omega: angle
            delta_t: time step
            icc: icc component

        Returns:
            New position of the robot

        """
        if self.v_right == self.v_left:
            delta_r_x = self.v_right * delta_t * np.cos(self._position[2])
            delta_r_y = self.v_right * delta_t * np.sin(self._position[2])
            return self._position + [delta_r_x, delta_r_y, 0]

        else:
            return np.array(
                np.array(
                    [
                        [np.cos(omega * delta_t), -np.sin(omega * delta_t), 0],
                        [np.sin(omega * delta_t), np.cos(omega * delta_t), 0],
                        [0, 0, 1],
                    ]
                )
                @ np.array(
                    [
                        [self._position[0] - icc[0]],
                        [self._position[1] - icc[1]],
                        [self._position[2]],
                    ]
                )
                + np.array([[icc[0]], [icc[1]], [omega * delta_t]])
            ).ravel()

    def compute_movement(self, delta_t: float) -> np.array:
        """
        Compute the kinematic movement of the robot, assuming no collisions.

        Args:
            dt: time elapsed

        Returns:
            New position as [x, y, angle]
        """
        omega = self._calculate_omega()
        r_capital = self._calculate_r_capital()
        icc = self._calculate_icc(r_capital)
        return self._calculate_new_position(omega, delta_t, icc)

    def set_position(self, position: np.array, angle: Optional[float] = None) -> None:
        """
        Set the new position and angle of the robot

        Args:
            position: position as [x, y, angle] or [x, y]
            angle: angle. If None, then the position must include the angle

        Returns:
            None
        """
        self._position[:2] = position.copy()
        if angle is None:
            self._position[2] = position[2]
        else:
            self._position[2] = angle
        if self._keep_trace:
            if self._iteration < len(self.trace):
                self._trace[self._iteration] = self.position.copy()
                self._iteration += 1

    def update_sensors(self, values: List[float]) -> None:
        """
        Args:
            values: new values of the sensors

        Returns:
            None
        """
        self.sensor_values = values.copy()

    @property
    def position(self) -> np.array:
        """
        Returns:
            Position of the robot in the x-y plane

        """
        return self._position[:2]

    @property
    def angle(self) -> float:
        """
        Returns:
            Angle of the robot w.r.t. the horizontal axis. The angle is
            counterclock-wise and expressed in radiants.
        """
        return self._position[2]

    @staticmethod
    def make_sensors(number: int) -> List[float]:
        """
        Generate a list of sensors uniformely distributed.

        Args:
            number: number of sensors

        Returns:
            List of sensor angles

        """
        return list(np.linspace(0, np.pi * 2, number + 1)[:-1])

    @property
    def trace(self) -> List[np.array]:
        """
        Returns:
            Position of the robot at each time step of the simulation

        """
        return self._trace


if __name__ == "__main__":
    robot = Robot(np.array([10, 10, 90]), 20, 200, 10, [0])
    robot.v_left = 10
    robot.v_right = 10
    robot.compute_movement(0.1)
