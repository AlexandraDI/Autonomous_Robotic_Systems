"""
Collection of environments to train and test the robots.
For each environment, we have the environment itself, a list of starting
positions for the robot and the size of the robot.

The coordiante system used is the same used in the pygame visualization.
The frame is 700x700. The surface used for the environments is a square from
(50,50) to (650, 650)

Utility functions
    - shift
    - normalize
    - shift_norm

Envoronments:
    - dusty_square: simple square with no obstacles
        - dusty_square_positions
        - dusty_square_radius
    - dusty_double_square: dusty_square with a big square in the middle.
        The robot explores the space outside the inner square.
        - dusty_double_square_positions
        - dusty_double_square_radius
    - dusty_room: two rooms with diagonal wall and obstacles
        - dusty_room_positions
        - dusty_room_radius
    - dusty_angles: single room with diagonal walls
        - dusty_angles_positions
        - dusty_angles_radius

"""

import numpy as np
from Environment import Environment


def shift(x: np.array) -> np.array:
    """
    Args:
        x: position in the pygame frame coordinates

    Returns:
        Room coordinates

    """
    return x - [50, 50]


def normalize(x: np.array) -> np.array:
    """

    Args:
        x: position in the room coordinates

    Returns:
        Coordinates in range [0, 1]
    """
    return x / 600


def shift_norm(x: np.array) -> np.array:
    """
    Args:
        x: position in the frame coordinates

    Returns:
        Position in the room coordinates in range [0, 1]
    """
    return normalize(shift(x))


##############################################################################

_outer_walls = [
    ((50, 50), (50, 650)),
    ((50, 650), (650, 650)),
    ((650, 650), (650, 50)),
    ((650, 50), (50, 50)),
]

dusty_square = Environment(_outer_walls)

dusty_square_positions = [
    np.array([350, 350, np.pi * 2]),
    np.array([350, 350, np.pi]),
    np.array([150, 150, np.pi * 2]),
    np.array([500, 500, np.pi]),
]

dusty_square_radius = 30

##############################################################################

_inner_square = [
    ((250, 450), (450, 450)),
    ((250, 450), (250, 250)),
    ((250, 250), (450, 250)),
    ((450, 250), (450, 450)),
]

dusty_double_square = Environment(_outer_walls + _inner_square)

dusty_double_square_positions = [
    np.array([150, 150, np.pi * 2]),
    np.array([500, 500, np.pi]),
    np.array([150, 350, np.pi * 2]),
]

dusty_double_square_radius = 30

##############################################################################

_room_walls = [
    ((50, 50), (350, 50)),
    ((350, 50), (350, 200)),
    ((350, 200), (400, 200)),
    ((400, 200), (400, 50)),
    ((400, 50), (650, 50)),
    ((650, 50), (650, 100)),
    ((650, 100), (600, 250)),
    ((600, 250), (600, 400)),
    ((600, 400), (650, 400)),
    ((650, 400), (650, 650)),
    ((650, 650), (400, 650)),
    ((400, 650), (400, 400)),
    ((400, 400), (350, 400)),
    ((350, 400), (350, 500)),
    ((350, 500), (250, 650)),
    ((250, 650), (50, 250)),
    ((50, 250), (50, 50)),
    ((200, 200), (250, 250)),
    ((250, 250), (200, 300)),
    ((200, 300), (150, 250)),
    ((150, 250), (200, 200)),
]

dusty_room = Environment(_room_walls)

dusty_room_positions = [
    np.array([550, 550, np.pi]),
    np.array([250, 450, np.pi * 2]),
    np.array([200, 100, np.pi * 2]),
]

dusty_room_radius = 30

##############################################################################

_angle_walls = [
    ((200, 50), (450, 50)),
    ((450, 50), (450, 250)),
    ((450, 250), (650, 50)),
    ((650, 50), (650, 350)),
    ((650, 350), (400, 650)),
    ((400, 650), (250, 400)),
    ((250, 400), (50, 400)),
    ((50, 400), (200, 50)),
]

dusty_angles = Environment(_angle_walls)

dusty_angles_positions = [
    np.array([600, 200, np.pi]),
    np.array([400, 400, np.pi]),
    np.array([350, 150, np.pi * 2]),
]

dusty_angles_radius = 30
