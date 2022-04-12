"""
This module contains the Environment of the Robot.

Implemented by Alexandra Gianzina
"""
from typing import List, Optional, Tuple
import numpy as np
import Geometry


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
        map_: map of the environment to estimate the position given the landmarks
        rng: random number generator
        noise: Gaussian noise of the position sensors. Given as tuple (mu, Sigma)
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
        map_: Optional["Environment"] = None,
        rng=None,
        noise: Tuple[np.array, np.array] = (np.zeros((3,)), np.eye(3)),
    ) -> None:
        # Base robot
        self._position = position
        self._radius = radius
        self.v_left = 0
        self.v_right = 0
        self.maximum_value_sensor = maximum_value_sensor
        self.acceleration = acceleration

        # Distance sensors
        self.sensors = sensors.copy()
        self.sensor_values = [self.maximum_value_sensor for i in self.sensors]

        # GE robot
        self.collisions = 0
        self._keep_trace = trace
        if trace:
            self._trace = [None for i in range(trace_size + 1)]
            self._trace[0] = self.position.copy()
            self._iteration = 1

        # KF robot
        self.beacons = []
        self.estimated_position = position.copy()
        self.map = map_
        self._rng = rng
        self._noise_mean = noise[0].copy()
        self._noise_cov = noise[1].copy()

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

    def compute_movement(self, dt: float) -> np.array:
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
        return self._calculate_new_position(omega, dt, icc)

    def compute_motion(self, dt: float) -> np.array:
        """
        Compute the velocity and angular velocty of the robot. Ignore collisions

        Args:
            dt: time elapsed

        Returns:
            Array containing the magnitude of the velocity and the angular
            velocity [v, omega].

        """
        omega = (self.compute_movement(dt)[2] - self.angle) / dt
        return np.array([self.v_left + self.v_right, omega])

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

    def update_sensors(
        self, values: List[float], beacons: Optional[List[np.array]] = []
    ) -> None:
        """
        Args:
            values: new values of the sensors
            beacons: distance, angle snd id from the beacons

        Returns:
            None
        """
        self.sensor_values = values.copy()

        # Keep closest beacons
        self.beacons = sorted(beacons, key=lambda x: x[0])[:2]
        self.estimated_position = self._triangulate_position()

    def _triangulate_position(self) -> Optional[np.array]:
        """
        Returns:
            Position estimate from the beacons

        """
        # TODO clean this method
        if len(self.beacons) == 0:
            return None

        point = None
        angle = None

        if len(self.beacons) == 2:
            points = Geometry.sphere_sphere_intersection(
                self.map.landmarks[int(self.beacons[0][2])],
                self.beacons[0][0],
                self.map.landmarks[int(self.beacons[1][2])],
                self.beacons[1][0],
            )
            if len(points) == 1:
                angle = Geometry.ray_point_angle(
                    points[0],
                    self.map.landmarks[int(self.beacons[0][2])],
                    self.beacons[0][1],
                )
                point = points[0]
            else:
                # Check first point
                point = points[0]
                angle = Geometry.ray_point_angle(
                    point,
                    self.map.landmarks[int(self.beacons[0][2])],
                    self.beacons[0][1],
                )
                angle_other = Geometry.ray_point_angle(
                    point,
                    self.map.landmarks[int(self.beacons[1][2])],
                    self.beacons[1][1],
                )

                if not np.allclose(angle - angle_other, 0):
                    # Wrong point -> compute the angle with the other point
                    point = points[1]
                    angle = Geometry.ray_point_angle(
                        point,
                        self.map.landmarks[int(self.beacons[0][2])],
                        self.beacons[0][1],
                    )

        if point is not None and angle is not None:
            return np.array([*point, angle]) + self._rng.multivariate_normal(
                self._noise_mean, self._noise_cov
            )
        else:
            return None

    @property
    def position(self) -> np.array:
        """
        Returns:
            Position of the robot in the x-y plane

        """
        return self._position[:2].copy()

    @property
    def angle(self) -> float:
        """
        Returns:
            Angle of the robot w.r.t. the horizontal axis. The angle is
            counterclock-wise and expressed in radiants.
        """
        return self._position[2].copy()

    @property
    def pose(self) -> float:
        """
        Returns:
            Position and angle of the robot
        """
        return self._position.copy()

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

    @property
    def detected_beacons(self) -> List[int]:
        """
        Returns:
            Index of the beacons currently detected
        """
        return [i[2] for i in self.beacons]

    @property
    def radius(self) -> float:
        """
        Returns:
            Radius of the robot
        """
        return self._radius


if __name__ == "__main__":
    robot = Robot(np.array([10, 10, 90]), 20, 200, 10, [0])
    robot.v_left = 10
    robot.v_right = 10
    robot.compute_movement(0.1)
