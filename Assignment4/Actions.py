"""
Module to handle that actions to control the robots.

Classes:
    * ActionTypes: enumerator for the different types of actions
    * Action: represent an atomic action

Constants:
    * ACTION_MAP (dict): handler for the actions. Call the correct function for
        the given action

Implemented by Gianluca Vico
"""
from typing import Optional
from enum import Enum, auto
from Robot import Robot


class ActionTypes(Enum):
    """
    NULL: no action
    L_POS: accelerate the left wheel
    L_NEG: decelerate the left wheel
    R_POS: accelerate the right wheel
    R_NEG: decelerate the right wheel
    POS: accelerate both wheels
    NEG: decelerate both wheels
    ZERO: stop both wheel
    """

    NULL = auto()  # Null action
    L_POS = auto()  # Left wheel +
    L_NEG = auto()  # Left wheel -
    R_POS = auto()  # Right wheel +
    R_NEG = auto()  # Right wheel -
    POS = auto()  # Both wheels +
    NEG = auto()  # Both wheels -
    ZERO = auto()  # Both to 0


class Action:
    """
    Atomic action.

    Args:
        robot: index of the robot affected
        action: type of the action
    """

    def __init__(
        self,
        robot: Optional[int] = 0,
        action: Optional[ActionTypes] = ActionTypes.NULL,
    ) -> None:
        self.robot = robot
        self.action = action

    def apply(self, robot: Robot, dt: float) -> None:
        """
        Apply the action on a concrete robot for dt time.

        This method is intented to be called by the simulation.

        Args:
            robot: robot affected.
            dt: time

        Returns:
            None.

        """
        self.action_map[self.action](robot, dt)

    def __repr__(self) -> str:
        """
        Returns:
            String representation
        """
        return f"Robot {self.robot} - {self.action}"


def _l_pos(robot, dt):
    robot.v_left += dt * robot.acceleration


def _l_neg(robot, dt):
    robot.v_left -= dt * robot.acceleration


def _r_pos(robot, dt):
    robot.v_right += dt * robot.acceleration


def _r_neg(robot, dt):
    robot.v_right -= dt * robot.acceleration


def _pos(robot, dt):
    _l_pos(robot, dt)
    _r_pos(robot, dt)


def _neg(robot, dt):
    _l_neg(robot, dt)
    _r_neg(robot, dt)


def _zero(robot, dt):
    robot.v_right = 0
    robot.v_left = 0


ACTION_MAP = {
    ActionTypes.NULL: lambda *args: None,
    ActionTypes.L_POS: _l_pos,
    ActionTypes.L_NEG: _l_neg,
    ActionTypes.R_POS: _r_pos,
    ActionTypes.R_NEG: _r_neg,
    ActionTypes.POS: _pos,
    ActionTypes.NEG: _neg,
    ActionTypes.ZERO: _zero,
}
