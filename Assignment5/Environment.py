"""
This module contains the Environment of the Robot.

Implemented by Alexandra Gianzina
"""
from typing import List, Tuple, Union, Optional
import numpy as np


class Environment:
    """
    Environment class.

    Args:
        points: List of walls. Each wall is a tuple containing the its start point
        and end point as np.array
        lanmarks: list of positions of the landmarks
    """

    def __init__(
        self,
        points: Union[
            List[Tuple[np.array, np.array]],
            List[Tuple[Tuple[float, float], Tuple[float, float]]],
        ],
        landmarks: Optional[Union[List[np.array], List[Tuple[float, float]]]] = [],
    ) -> None:
        # Make sure each wall is (nparray([x1, y1]), np.array([x2, y2]))
        self.wall_points = [(np.array(start), np.array(end)) for start, end in points]

        self.landmarks = [np.array(i) for i in landmarks]

    def __repr__(self) -> str:
        """
        Returns:
            String representation
        """
        walls = [
            f"Wall: ({i[0][0]}, {i[0][1]}) - ({i[1][0]}, {i[1][1]})"
            for i in self.wall_points
        ]
        landmarks = [f"Landmark: ({i[0]}, {i[1]})" for i in self.landmarks]
        walls = "\n".join(walls)
        landmarks = "\n".join(landmarks)
        return "Environment:\n" + walls + "\n" + landmarks
