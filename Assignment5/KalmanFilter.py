"""
Implemented by Gianluca Vico
"""
from typing import Optional, Tuple
import numpy as np

# Motion noise
R_LARGE = np.eye(3) * [10, 10, 10]
R_SMALL = np.eye(3) * [1, 1, 1]

# Sensor noise
Q_LARGE = np.eye(3) * [10, 10, 10]
Q_SMALL = np.eye(3) * [1, 1, 1]


class Kalman:
    def __init__(
        self,
        initial_state: np.array,
        motion_cov_noise: np.array,
        sensor_cov_noise: np.array,
        eps: Optional[float] = 1e-4,
    ) -> None:
        """
        Implementation of the calman filter for the circular robot.

        Both noise matrices are 3x3 and contain noise for x, y and theta
        coordinates of the robot.

        The A mtrix in the Kalman filter is the identity matrix.
        B is estimated given the robot.
        C is also the identity matrix.

        Args:
            initial_state: initial pose of the robot as [x, y, theta]
            motion_cov_noise: noise in the motion estimate
            sensor_cov_noise: noise in the sensor estimate
            esp: small number to estimate the initial covariance
        """
        self._R = motion_cov_noise.copy()
        self._Q = sensor_cov_noise.copy()
        self._state = initial_state.copy()
        self._cov = np.eye(3) * eps

    def estimate(
        self, motion: np.array, sensor_position: np.array, dt: float
    ) -> Tuple[np.array, np.array]:
        """
        Estimate the new position of the robot

        Args:
            motion: array with the velocity and the angular velocity
            sensor_position: position estimated by the sensors. If None, use
                the filter without the sensor estimate
            dt: time step

        Returns:
            Estimate of the pose (3x1) and its covariance (3x3)
        """
        # Prediction
        B = (
            np.array([[np.cos(self._state[2]), 0], [np.sin(self._state[2]), 0], [0, 1]])
            * dt
        )
        state_estimate = self._state + B @ motion
        cov_estimate = self._cov + self._R

        # Correction
        if sensor_position is None:
            self._state = state_estimate
            self._cov = cov_estimate
        else:
            K = cov_estimate @ np.linalg.inv(cov_estimate + self._Q)

            self._state = state_estimate + K @ (sensor_position - state_estimate)
            self._cov = (np.eye(3) - K) @ cov_estimate
        return self._state.copy(), self._cov.copy()

    @property
    def state(self) -> np.array:
        """
        Returns:
            Current estimated pose of the robot
        """
        return self._state.copy()

    @property
    def covariance(self) -> np.array:
        """
        Returns:
            Current covariance of the estimation of the pose of the robot
        """
        return self._cov.copy()
