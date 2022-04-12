#  Assignment 4
## Files

* Actions.py: represents the user actions on the robot
* DustyEnvironment.py: collection of maps to train the robots
* Environment.py: represent a map that robots can navigate
* GE.py: evolutionary algorithm
* Geometry.py: collection of method to compute the collisions
* Plotting.py: plot the result of the GE optimization
* Results.py: show a demo of the trained robot controller.
* Robot.py: model a circular robot with two independent wheels
* RobotController.py: simple ANN to control a robot given the sensors' values
* Simulation.py: simulate the environment and update the position of the robots
    in a realistic way
* Training.py: methods to run the GE for the robots
* Visualization.py: visualize the robot navigating a maze

## How to run

To run the GE: `python3 Training.py`
To run the results: `python3 Results.py`

## Dependencies

Dill: `conda -c anaconda dill`

