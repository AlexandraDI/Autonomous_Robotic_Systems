# Assignment 2 - Robot simulation

## Files
* Action.py: this module is used to pass the inputs to the simulation in order
to control the robot
* Environment.py: it represents the walls in the simulation
* Geometry.py: functions to detect collisions
* Simulation.py: simulate the robot movemement and handle the collisions
* Visualization.py: display the robot, the environment and the sensor values

## Requirements
This project need the following packages:

* pygame
* numpy

## How to run
`python3 Visualization.py`

## Description
The simulation contains a robot and a set of walls.
The walls are simply lines, so they do not have thickness. However, the robot
can collide with them.

When a collision occours, the robot slides on the wall, unless it is perpendicular
to it.

On the robot, the values of the sensors are displayed. Each sensor is on the
perimeter of the robot and it is defined by the angle from the foward direction
of the robot.
Inside the robot, the velocities of the left wheel and the right wheel are also
displayed. The left one is on the top, the right one is on the bottom.


## Notes
Pygame behaves slightly differently on Windows and Linux. In some cases, keeping
a key pressed is interpreted as a continuous input (and so the velocity of the
robot keeps changing). In other cases, pygame reads it as a single input (so the
velocity increases only once).
