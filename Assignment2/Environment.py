"""
This module contains the Environment of the Robot.

Implemented by Alexandra Gianzina
"""
from typing import List, Tuple, Union
import numpy as np
import pygame


class Environment:
    """
    Environment class.

    Args:
        points: List of walls. Each wall is a tuple containing the its start point
        and end point as np.array
    """

    def __init__(
        self,
        points: Union[
            List[Tuple[np.array, np.array]],
            List[Tuple[Tuple[float, float], Tuple[float, float]]],
        ],
    ) -> None:
        self.wall_points = points
        # Make sure each wall is (nparray([x1, y1]), np.array([x2, y2]))
        self.wall_points = [
            (np.array(start), np.array(end)) for start, end in self.wall_points
        ]

    def __repr__(self) -> str:
        """
        Returns:
            String representation
        """
        s = [
            f"Wall: ({i[0][0]}, {i[0][1]}) - ({i[1][0]}, {i[1][1]})"
            for i in self.wall_points
        ]
        s = "\n".join(s)
        return "Environment:\n" + s




if __name__ == "__main__":
    pygame.init()
    number_of_inner_walls = 20
    np.random.seed(37)

    a = [
        (np.random.randint(700, size=2), np.random.randint(700, size=2))
        for i in range(number_of_inner_walls)
    ]
    env = Environment(a)

