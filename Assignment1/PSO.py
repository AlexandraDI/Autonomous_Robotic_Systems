"""
This module contain the Particle Swarm Optimization (PSO) algorithm.

It contains the following classes:
    - PSOSettings: dataclass containing the parameters for the PSO algorithm.
    - PSO: main class containing the PSO algorithms

Implemented by Gianluca Vico

"""
from dataclasses import dataclass
from typing import Tuple, Callable, Iterable
import numpy as np


@dataclass
class PSOSettings:
    """
    Dataclass with the settings for the PSO

    Args:
        dimenstions: the number of dimension of the optimization problem
        particles: the number of particles used in the optimization
        neighbours: the size of the neighbourhood. Each particle will use the
            :neighbours: closest particles to look for the best neighbour.
        a: "a" parameter used to compute the new velocity of each particle
        b: "b" parameter used to compute the new velocity of each particle
        c: "c" parameter used to compute the new velocity of each particle
        a_decay: "a" decrease over time by this factor. A decay of 0 means that "a" remains constant
        eps: when the value of the particles changes by less the eps the algorithm stops
        v_max: maximum velocity of the particles
        size: the area of space to investigate. The particles start from this area but the could move outsie.
            This is a tuple (min_value, max_value) for each dimension.
        max_iteration: the algorithm stops after a fixed number of iteration even if it did not converge
        seed: seed for the rng

    Note:
        The function to minimize does not need to be known in advance, but the
        dimenstions of the input does.
    """

    dimensions: int = 2
    particles: int = 20
    neighbours: int = 3
    a: float = 0.9
    b: float = 2
    c: float = 2
    a_decay: float = 0.001  # from 0.9 to 0.45 in 1000 iterations
    eps: float = 1e-4
    v_max: float = 0.02
    size: Tuple[float] = (-10, 10)
    max_iterations: float = np.inf
    seed: int = 42

    def __repr__(self):
        return (
            "PSO Settings:\n"
            f"- Dimensions: {self.dimensions}\n"
            f"- Particles: {self.particles}\n"
            f"- Neighbours: {self.neighbours}\n"
            f"- a: {self.a}\n"
            f"- b: {self.b}\n"
            f"- c: {self.c}\n"
            f"- a_decay: {self.a_decay}\n"
            f"- Epsilon: {self.eps}\n"
            f"- Maximum velocity: {self.v_max}\n"
            f"- Size: {self.size}\n"
            f"- Maximum iterations: {self.max_iterations}\n"
            f"- Seed: {self.seed}\n"
        )


class PSO:
    """
    PSO algorithm for minimizing functions.

    Args:
        settings: PSO_Settings for the algorithm
    """

    def __init__(self, settings: PSOSettings) -> None:
        # Random number generator. Make the results reproducible
        self._rng: np.random.mtrand.RandomState = np.random.mtrand.RandomState(
            settings.seed
        )

        # Settings for the algorithm that will be used later
        self._max_iterations: float = settings.max_iterations
        self._v_max: float = settings.v_max
        self._eps: float = settings.eps

        self._a: float = settings.a
        self._b: float = settings.b
        self._c: float = settings.c
        self._a_decay: float = settings.a_decay

        self._neighbours: int = settings.neighbours

        # Particle matrix. Each row is the position of a particle
        self._particles: np.array = (
            self._rng.rand(settings.particles, settings.dimensions)
            * (settings.size[1] - settings.size[0])
            + settings.size[0]
        )

        # Velocity matrix. Velocity in each direction for each particles
        self._velocity: np.array = self._rng.rand(*self._particles.shape) - 0.5

        # Position of the best value for each particle
        self._local_best: np.array = self._particles.copy()

        # Best value for each particle. Start by -inf
        self._local_best_values: np.array = np.full(
            (1, self._particles.shape[0]), np.inf
        )

        # Position of the best neighbour for each particle
        self._global_best: np.array = self._particles.copy()

        # Current value of the particles
        self._values: np.array = np.full((1, self._particles.shape[0]), np.inf)

        # Iteration performed so far
        self._iteration: int = 0

        # How muh the particles' value changed in the last iteration
        self._avg_change: float = 0

    def optimize(
        self, function: Callable[[np.array], np.array]
    ) -> Tuple[np.array, np.array]:
        """
        Minimize the given function

        Args:
            function (Callable[[np.array], np.array]): function to minimize.
                It has to take as input the matrix of particles and return a
                single float for each particle.

        Returns:
            optimal point as row vector.
            optimal value as single float in a numpy array.
        """
        # Iterate until it finds the optimal value
        for _, _ in self.optimize_generator(function):
            pass

        # Find the index of the minimum value
        return self.get_optimal_value(self._particles, self._values)

    def optimize_generator(
        self, function: Callable[[np.array], np.array]
    ) -> Tuple[np.array, np.array]:
        """
        Optimization as generator. This can be used in a loop to investigate
        the evolution of the particle swarm.

        Args:
            function (Callable[[np.array], np.array]): function to minimize.
                It has to take as input the matrix of particles and return a
                single float for each particle.

        Returns:
            Generator that yields the particles' matrix and the value vector

        """
        particles, values = self._init_optimize(function)
        yield self._particles, self._values

        # Iterated untile max iterations
        while self._iteration < self._max_iterations:
            # Move the particles and find the new values
            particles, new_values = self.optimize_step(function)

            change = np.abs(new_values - values)
            # Check if we are still improving
            if (change < self._eps).all():
                break
            self._avg_change = np.sum(change) / self._particles.shape[0]
            # Update the values and the particles
            values = new_values
            self._particles = particles
            yield self._particles, self._values

    def optimize_step(
        self, function: Callable[[np.array], np.array]
    ) -> Iterable[Tuple[np.array, np.array]]:
        """
        Single iteration of the PSO algorithms

         Args:
             function (Callable[[np.array], np.array]): function to minimize.
                 It has to take as input the matrix of particles and return a
                 single float for each particle.

        Returns:
            particle matrix.
            vector of particle's values.
        """
        if self._iteration == 0:
            return self._init_optimize(function)
        return self._optimize_step(function)

    def _optimize_step(
        self, function: Callable[[np.array], np.array]
    ) -> Tuple[np.array, np.array]:
        """
        Single iteration of the PSO algorithms.

        For each particle, the velocity and the new position are computed:

        v(t+1) = a * v(t) + b * Rand * (lbest - s(t)) + c * Rand * (gbest - s(t))
        s(t+1) = s(t) + v(t+1)

        Then, the local best and the gloabal (in the neighbourhood) are computed.

        Args:
            function (Callable[[np.array], float]): function to minimize.
                It has to take as input the matrix of particles and return a single float.

        Returns:
            particle matrix.
            vector of particle's values.
        """
        # Update the velocity, vectorized
        # v(t+1) = a * v(t) + b*R*(lbest - s(t)) + c*R'*(gbest - s(t))
        self._velocity = (
            self.a * self._velocity
            + self._b
            * np.random.rand(*self._velocity.shape)
            * (self._local_best - self._particles)
            + self._c
            * np.random.rand(*self._velocity.shape)
            * (self._global_best - self._particles)
        )

        # Limit to max velocity
        # self._velocity = np.clip(self._velocity, a_max=self._v_max, a_min=-self._v_max)
        v = np.sqrt(np.sum(self._velocity ** 2, axis=1))
        mask = (v < -self._v_max) + (v > self._v_max)
        self._velocity[mask] = self._velocity[mask] * (self._v_max / v[mask]).reshape(
            -1, 1
        )

        # Update position, vectorized
        # # s(t+1) = s(t) + v(t) * dT -> dT: 1 iteration
        self._particles += self._velocity

        # Compute the new values
        self._values = function(self._particles)

        # Update the local best -> does the value decrease?
        mask = self._values < self._local_best_values
        self._local_best_values[mask] = self._values[mask]
        self._local_best[mask] = self._particles[mask]

        # Update the global best
        self._best_neighbour()

        # Update a
        # self._a = self._a / (1 + self._a_decay * self.iteration)

        self._iteration += 1

        return self._particles, self._values

    def _init_optimize(
        self, function: Callable[[np.array], np.array]
    ) -> Tuple[np.array, np.array]:
        """
        First step of the optimization

        Args:
            function (Callable[[np.array], float]): function to minimize.
                It has to take as input the matrix of particles and return a single float.

        Returns:
            particle matrix.
            vector of particle's values.
        """
        # Compute the function
        self._values = function(self._particles)

        # Best values found so far
        self._local_best_values = self._values.copy()

        # Update best neighbours
        self._best_neighbour()

        self._iteration += 1
        return self._particles, self._values

    def _best_neighbour(self) -> None:
        """
        Update the best neighbours.
        For each particle find the position and the value of the neighbour with
        the lower function value.

        Returns:
            None
        """
        for i, particle in enumerate(self._particles):
            # Square distance between this particle and the others
            # Include itself in the neighbourhood
            distances = [
                np.sum((particle - other) ** 2)
                for j, other in enumerate(self._particles)
            ]

            # Sort the index of the particles by distance
            sorted_indexes = np.argsort(distances)

            # Limit the neighbourhood size
            neighbours = sorted_indexes[: self._neighbours]

            # Index of the neighbour with the lowest value
            best_neighbour = np.argmin(self._local_best_values[neighbours])

            # Assign the position of the best neighbour in the global best matrix
            self._global_best[i] = self._local_best[neighbours][best_neighbour]

    @staticmethod
    def get_optimal_value(
        particles: np.array, values: np.array
    ) -> Tuple[np.array, np.array]:
        """
        Find the position and the value of the minimum.

        Args:
            particles (np.array): matrix containing the particles' position.
            values (np.array): vector containing the function value for each particles.

        Returns:
            Position of the particle the minimum value.
            Minimum value.

        """
        # Index of the minimum
        best = np.argmin(values)
        return particles[best], values[best]

    @property
    def iteration(self) -> int:
        """
        Returns:
            int: Iterations completed so far.
        """
        return self._iteration

    @property
    def a(self) -> float:
        """
        Returns:
            float: Updated value for the "a" parameter of the PSO algorithm.
        """
        return self._a / (1 + self._a_decay * self._iteration)

    @property
    def b(self) -> float:
        """
        Returns:
            float: "b" parameter of the PSO algorithm.
        """
        return self._b

    @property
    def c(self) -> float:
        """
        Returns:
            float: "c" parameter of the PSO algorithm.
        """
        return self._c

    @property
    def a_decay(self) -> float:
        """
        a = a / (1 + decay * iteration)

        Returns:
            float: how fast "a" decay over time
        """
        return self._a_decay

    @property
    def v_max(self) -> float:
        """
        Returns:
            float: maximum velocity of the particles on each axis.
        """
        return self._v_max

    @property
    def max_iterations(self) -> float:
        """
        Returns:
            int: Maximum number of iterations of the algorithm.
        """
        return self._max_iterations

    @property
    def epsilon(self) -> float:
        """
        Returns:
            float: When the gloabal fitness increase is less than epsilon the
            algorithm stops.

        """
        return self._eps

    @property
    def neighbours(self) -> float:
        """
        Returns:
            float: Size of the neighbourhood considered when searching for the
            global optimum.

        """
        return self._neighbours

    @property
    def particles(self) -> int:
        """
        Returns:
            int: Number of particles.
        """
        return self._particles.shape[0]

    @property
    def dimensions(self) -> int:
        """
        Returns:
            int: Number of dimenstions of the function to optimize.
        """
        return self._particles.shape[1]


if __name__ == "__main__":
    print("Simple test for PSO")
    print("Minimizing f(x) = sum_i x_i**2")
    print("Dimensions: 5")
    f = lambda x: np.sum(x ** 2, axis=1)
    s = PSOSettings(dimensions=5, a_decay=0)
    print(s)
    pso = PSO(s)
    p, v = pso.optimize(f)
    print("Minimum:", p)
    print("Minimum value:", v)
    print("Iterations:", pso.iteration)
