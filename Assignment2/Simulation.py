"""
Implemented by Gianluca Vico
"""
import numpy as np
from typing import List, Optional, Tuple
from Environment import Environment
from Robot import Robot
from Actions import Action, ACTION_MAP
import Geometry


class Simulation:
    """
    Simulation for the robot environment.

    Check for collision and update the robots state.

    Methods:
        * update: update the physics.
        * apply_action: control the robots


    Args:
        environment: position of the walls
        robots: list of robots in the environment
        dt: time step for the simulation
    """

    def __init__(
        self, environment: Environment, robots: List[Robot], dt: float
    ) -> None:
        self.environment = environment
        self.robots = robots
        self.dt = dt

    def update(self, dt: Optional[float] = None) -> None:
        """
        Compute next time step of the simulation.
        We use Euler method for simplicity.

        Args:
            dt: time step for this update. By default the simulation time step is used.

        Returns:
            None.

        """
        self._update_robots(self.dt if dt is None else dt)
        self._update_sensors()

    def apply_action(self, action: Action, dt: Optional[float] = None) -> None:
        """
        Change the state of a robot. The action effect will be visible the next
        time that simulation.update is calles

        Args:
            action: acttion to be applied on a robot
            dt: how long the action is applied. If none, the defeault time step is used

        Returns:
            None

        """
        robot = self.robots[action.robot]
        ACTION_MAP[action.action](robot, self.dt if dt is None else dt)

    def _update_robots(self, dt) -> None:
        """
        Update the position of the robots

        Returns:
            None.

        """
        for r in self.robots:
            position = r.position
            new_position = r.compute_movement(dt)
            angle = new_position[2]
            new_position = new_position[:2]
            colliding = True
            movement = new_position - position
            old_wall = None
            # while np.linalg.norm(movement) > 0:
            while colliding:
                point, dist, wall = self._get_collision(
                    position, new_position, r.radius
                )

                # Check if we are colliding or if we are colliding with the
                # exact same wall
                if point is None or (point is not None and wall is old_wall):
                    movement = 0
                    colliding = False
                else:
                    old_wall = wall
                    # Move until the collision point
                    position = self._get_position_before_collision(
                        position, new_position, wall, r.radius
                    )
                    # Movement that we still have to do
                    movement = new_position - position

                    # Compute movement with sliding
                    new_position = self._compute_sliding_movement(
                        position, movement, wall
                    )

                    # TEST only one collision
                    movement = 0
                r.set_position(new_position, angle)

    def _get_collision(
        self, start_position: np.array, end_position: np.array, radius: float
    ) -> Tuple[np.array, float, Tuple[np.array, np.array]]:
        """
        Find the collision between a moving robot and the walls

        Args:
            start_position: initial position of the robot as [x, y]
            end_position: ending position of the robot as [x, y] if no
                collision occoured
            radius: radius of the robot

        Returns:
            collision_point: point of collision on the capsule representing the
                movement of the robot
            collision_dist: distance between the initial position and the
                collision point
            colliding_wall: wall that collided with the robot
            (that wall popped out of nowhere)

        """
        collision_point = None
        colliding_wall = None
        collision_dist = np.inf
        movement = end_position - start_position

        # Collide with a capsule if the movememt is big enough
        # Otherwise a circle is enough
        if np.linalg.norm(movement) > 2 * radius:
            intersection = Geometry.capsule_segment_intersection
        else:
            intersection = lambda x, *args: Geometry.sphere_segment_intersection(*args)
        # Check all collision
        for wall in self.environment.wall_points:
            collisions = intersection(start_position, end_position, radius, wall)
            if len(collisions) != 0:
                # Find closest collision
                for c in collisions:
                    dist = Geometry.dist(start_position, c)
                    if collision_point is None or dist < collision_dist:
                        collision_point = c
                        colliding_wall = wall
                        collision_dist = dist
        return collision_point, collision_dist, colliding_wall

    def _update_sensors(self) -> None:
        """
        Set the new values for the sensors on the robots

        Returns:
            None.

        """
        for r in self.robots:
            values = [r.maximum_value_sensor for i in r.sensors]
            point = None
            for i, s in enumerate(r.sensors):
                direction = np.array([np.cos(s + r.angle), np.sin(s + r.angle)])
                for wall in self.environment.wall_points:
                    sensor_position = r.position + direction * r.radius

                    point = Geometry.ray_segment_intersection(
                        r.position, direction, wall
                    )
                    if point is not None:
                        values[i] = min(
                            values[i], Geometry.dist(sensor_position, point)
                        )
            r.update_sensors(values)

    def _compute_sliding_movement(
        self, position: np.array, movement: np.array, wall: Tuple[np.array, np.array]
    ) -> np.array:
        """
        When there is a collision the robot slides on the wall

        Args:
            position: last position before the collision
            movement: movement that the robot still have to perfomr
            wall: colliding wall

        Returns:
            New position taking into account the sliding

        """
        wall_direction = Geometry.normalize(wall[1] - wall[0])
        sliding = np.dot(wall_direction, movement)
        return position + wall_direction * sliding

    def _get_position_before_collision(
        self,
        position: np.array,
        new_position: np.array,
        wall: Tuple[np.array, np.array],
        radius: float,
    ) -> np.array:
        """
        Compute the last valid position of the robot right before the collision

        Args:
            position: initial position of the robot
            new_position: final position if the collision didn't occoured'
            wall: colliding wall
            radius: radius of the robot

        Returns:
            np.array: position before the collision

        """

        direction = Geometry.normalize(new_position - position)
        wall_direction = Geometry.normalize(wall[1] - wall[0])
        # Angle between the movement and the wall
        angle = np.arccos(np.dot(direction, wall_direction))

        projection_point = Geometry.ray_line_intersection(position, direction, wall)

        if angle == 0:  # Already sliding on the wall
            return new_position
        # Distance in the direction of the movement in the moment of the inpact
        distance_to_wall = radius / np.sin(angle)

        if projection_point is None:
            # print("The walls are moving!!")
            return new_position
        else:
            return projection_point - direction * distance_to_wall


if __name__ == "__main__":
    robot = Robot(
        position=np.array([50, 50, np.pi * 2]),
        radius=1,
        maximum_value_sensor=10000,
        acceleration=1,
        sensors=[0, np.pi / 2, np.pi, np.pi * 1.5],
    )
    env = Environment(
        [
            ((0, 0), (0, 100)),
            ((0, 100), (100, 100)),
            ((100, 100), (100, 0)),
            ((100, 0), (0, 0)),
        ]
    )
    sim = Simulation(env, [robot], 0.1)
    sim.update()
    print(robot.sensor_values)
