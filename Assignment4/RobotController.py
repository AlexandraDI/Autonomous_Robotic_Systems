"""
ANN to control the robot.

The controller can set the velocity directly without using the actions as the
human controller does

Classes:
    Controller: ANN to control the robot

Functions:
    relu: f(x) = max(0, x)
    linear: f(x) = x
    sigmoid: f(x) = 1 / (1 + exp(-x))
"""
from typing import Optional, Callable, Union, List
import numpy as np


def relu(x: np.array) -> np.array:
    """
    Args:
        x: input vector

    Returns:
        max(0, x)
    """
    return np.clip(x, a_min=0, a_max=np.inf)


def linear(x: np.array) -> np.array:
    """
    Args:
        x: input vector

    Returns:
        x
    """
    return x


def sigmoid(x: np.array) -> np.array:
    """
    Args:
        x: input vector

    Returns:
        1 / (1 + exp(-x))

    """
    return 1 / (1 + np.exp(-x))


class Controller:
    """
    ANN to control the robot

    Args:
        weights: weights of the network as a single vector
        sensors: number of sensors of the robot
        hidden: size of the hidden layer
        output: size of the output layer
        clip: limit the output to [-clip, +clip]
        func: activation function for the hidden layer
    """

    def __init__(
        self,
        weights: np.array,
        sensors: Optional[int] = 12,
        hidden: Optional[int] = 4,
        output: Optional[int] = 2,
        clip: Optional[float] = 200,
        func: Callable[[np.array], np.array] = relu,
    ) -> None:
        assert len(weights) == self.weight_size(sensors, hidden, output)
        # Select first elements as w1
        self._w1 = weights[: (sensors + 1 + output) * hidden].reshape(hidden, -1)
        # Select last elements as w2
        self._w2 = weights[-(hidden + 1) * output :].reshape(output, -1)

        self._output = np.zeros((2,))

        self._clip = clip
        self._function = func

    def __call__(self, sensors: Union[List[float], np.array]) -> np.array:
        """
        Compute the new velocity of the robot

        Args:
            sensors: value of the sensors on the robot

        Returns:
            velocity of the left and right wheel

        """
        self._output = self._w1 @ np.hstack([1, sensors, self._output])
        self._output = self._function(self._output)
        self._output = self._w2 @ np.hstack([1, self._output])
        self._output = np.clip(self._output, -self._clip, self._clip)
        return self._output

    @classmethod
    def from_population(
        cls,
        population: List[np.array],
        sensors: Optional[int] = 12,
        hidden: Optional[int] = 4,
        output: Optional[int] = 2,
        clip: Optional[float] = 10,
        func: Callable[[np.array], np.array] = relu,
    ) -> List["Controller"]:
        """
        Generate a list of controller from the GE population

        Args:
            population: list of weights
            sensors: number of sensors of the robot
            hidden: size of the hidden layer
            output: size of the output layer
            clip: limit the output to [-clip, +clip]
            func: activation function for the hidden layer
        Returns:
            List of controllers

        """
        return [cls(i, sensors, hidden, output, clip, func) for i in population]

    @staticmethod
    def weight_size(
        sensors: Optional[int] = 12,
        hidden: Optional[int] = 4,
        output: Optional[int] = 2,
    ) -> int:
        """


        Args:
           sensors: number of sensors of the robot
           hidden: size of the hidden layer
            output: size of the output layer
        Returns:
            Length of the weight vector

        """
        return (1 + sensors + output) * hidden + (hidden + 1) * output
