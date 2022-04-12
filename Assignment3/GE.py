"""
Genetic algorithm

Classes:
    * GESettings: parameters for the genetic algorithm
    * GE: genetic algorithm

Methods:
    * distance
    * nbest_selection
    * rank_selection
    * proportional_selection
    * tournament_selection
    * one_point_crossover
    * avg_crossover
    * mutation

impemented by Gianluca Vico
"""
import numpy as np
import pickle
from dataclasses import dataclass
import itertools
from typing import Callable, List, Optional, Tuple


@dataclass
class GESettings:
    """
    Settings for GE

    Args:
        population_size: size of the population
        mutation_rate: probability of a mutation
        crossover: method to perform the crossover
        mutation: method to perform the mutation
        selection: method to select the next population
        individual_generator: method to generate the individuals
        fitness_function: method to compute the fitness
        distance_function: method to compute the diversity between 2 individuals
        keep_best: number of best individuals to keep for the nest generation
        seed: random seed
    """

    population_size: int = 1000
    mutation_rate: float = 0.01
    crossover: Callable = None
    mutation: Callable = None
    selection: Callable = None
    individual_generator: Callable = None
    fitness_function: Callable = None
    distance_function: Callable = None  # For computing the diversity
    keep_best: int = 5
    seed: int = 42
    size: Tuple[float] = (-10, 10)


class GE:
    """
    Genetic algorithm

    Args:
        settings: settings for the algorithm
    """

    def __init__(self, settings: GESettings) -> None:
        self._settings = settings
        self.reset()

    def save(self, path: str) -> None:
        """
        Save the current state of the GE

        Args:
            path: destination
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "GE":
        """
        Load a GE from a file

        Args:
            path: destination

        Returns:
            GE from file
        """
        with open(path, "rb") as f:
            ge = pickle.load((f))
        return ge

    def reset(self) -> None:
        """
        Reinitialize the algorithm
        """
        self._rng = np.random.RandomState(self._settings.seed)
        self._population = [
            self._settings.individual_generator(self._rng)
            for i in range(self._settings.population_size)
        ]

        self._fitness = [1 for i in range(self._settings.population_size)]
        self._sorting = list(range(self._settings.population_size))
        self._generation = 0

    def iterate(self) -> None:
        """
        Single iteration of the algorithm
        """
        # Selection & cross over
        best = [self._population[i] for i in self._sorting[: self._settings.keep_best]]
        self._population = self._settings.selection(
            self._population, self._fitness, self._sorting, self._rng
        )
        self._population = self._settings.crossover(
            self._population, self._fitness, self._sorting, self._rng
        )
        # Mutation
        for i, individual in enumerate(self._population):
            if self._rng.rand() < self._settings.mutation_rate:
                self._population[i] = self._settings.mutation(individual, self._rng)
        self._population = best + self._population[: -self._settings.keep_best]
        self._update_fitness()
        self._generation += 1

    @property
    def generation(self) -> int:
        """
        Returns:
            Generation counter
        """
        return self._generation

    def _update_fitness(self) -> None:
        """
        Compute the new fitness and rank
        """
        self._fitness = [
            self._settings.fitness_function(i, self._rng) for i in self._population
        ]
        self._sorting = np.flip(np.argsort(self._fitness))  # From largest to smallest

    @property
    def fitness(self) -> List[float]:
        """
        Returns:
            Fitness of each individual (not sorted)
        """
        return self._fitness

    def get_diversity(self) -> float:
        """
        Returns:
            diversity of the population
        """
        div = 0
        for i, j in itertools.combinations(self._population, 2):
            div += self._settings.distance_function(i, j)
        return div

    def get_best(self):
        return self._population[self._sorting[0]]


# Example functions
def nbest_selection(
    population: List[np.array],
    fitness: List[float],
    rank: List[int],
    rng,
    nbest: Optional[int] = 10,
    *args: List
) -> List[np.array]:
    """
    Select the best individuals

    Args:
        population: population of the GE
        fitness: fitness of each individual
        rank: rank of each individual
        rng: random number generator
        nbest: best individuals to select
        *args: other args

    Returns:
        New population
    """
    selection = np.repeat(
        np.array(population)[rank][:nbest], int(np.ceil(len(population) / nbest)), 0
    )
    selection = selection[: len(population)]
    return list(selection)


def rank_selection(
    population: List[np.array], fitness: List[float], rank: List[int], rng, *args: List
) -> List[np.array]:
    """
    Select the best individuals with probability based on the rank

    Args:
        population: population of the GE
        fitness: fitness of each individual
        rank: rank of each individual
        rng: random number generator
        *args: other args

    Returns:
        New population
    """
    # Probability of selection each individual
    probs = len(rank) - np.array(rank)
    probs = probs / np.sum(probs)
    # return rng.choice(population, len(population), p=probs / np.sum(probs))
    return [population[i] for i in rng.choice(rank, len(rank), p=probs)]


def proportional_selection(
    population: List[np.array], fitness: List[float], rank: List[int], rng, *args: List
) -> List[np.array]:
    """
    Select the best individuals with probability based on the fitness

    Args:
        population: population of the GE
        fitness: fitness of each individual
        rank: rank of each individual
        rng: random number generator
        *args: other args

    Returns:
        New population
    """
    probs = np.array(fitness) / np.sum(fitness)
    return [population[i] for i in rng.choice(rank, len(rank), p=probs)]


def tournament_selection(
    population: List[np.array],
    fitness: List[float],
    rank: List[int],
    rng,
    k: Optional[int] = 3,
    *args: List
) -> List[np.array]:
    """
    Select the best individuals with tournaments

    Args:
        population: population of the GE
        fitness: fitness of each individual
        rank: rank of each individual
        rng: random number generator
        k: size of the tournament
        *args: other args

    Returns:
        New population
    """
    probs = np.array(fitness) / np.sum(fitness)
    return [
        population[np.min(rng.choice(rank, k, p=probs))] for i in range(len(population))
    ]


def one_point_crossover(
    population: List[np.array], fitness: List[float], rank: List[int], rng, *args: List
) -> List[np.array]:
    """
    Crossover. Split the genome on a point and exchange the genome between two
    individuals

    Args:
        population: population of the GE
        fitness: fitness of each individual
        rank: rank of each individual
        rng: random number generator
        *args: other args

    Returns:
        New population
    """
    pairs = rng.choice(rank, (len(rank) // 2, 2), False)
    new_population = [None for i in range(len(pairs) * 2)]
    for i, (i1, i2) in enumerate(pairs):
        index = rng.randint(0, len(population[i1]))
        tmp = np.ndarray(len(population[i1]))
        tmp[:index] = population[i1][:index]
        tmp[index:] = population[i2][index:]
        new_population[i * 2] = tmp

        tmp = np.ndarray(len(population[i1]))
        tmp[:index] = population[i2][:index]
        tmp[index:] = population[i1][index:]
        new_population[i * 2 + 1] = tmp
    return new_population


def avg_crossover(
    population: List[np.array], fitness: List[float], rank: List[int], rng, *args: List
) -> List[np.array]:
    """
    Crossover. Take the average between two individuals

    Args:
       population: population of the GE
       fitness: fitness of each individual
       rank: rank of each individual
       rng: random number generator
       *args: other args

   Returns:
       New population

    """
    pairs1 = rng.choice(rank, (len(rank) // 2, 2), False)
    pairs2 = rng.choice(rank, (len(rank) // 2, 2), False)
    new_population = [None for i in range(len(pairs1) + len(pairs2))]
    for i, ((i1, i2), (i3, i4)) in enumerate(zip(pairs1, pairs2)):
        new_population[i * 2] = (population[i1] + population[i2]) / 2
        new_population[i * 2 + 1] = (population[i3] + population[i4]) / 2
    return new_population


def mutation(
    individual: np.array, rng, range_: Optional[int] = 5, *args: List
) -> np.array:
    """
    Change a random gene in the genorm

    Args:
        individual: individual to mutate
        rng: random number generator
        range_: range for mutated gene (-range_, +range_)
        *args: others

    Returns:
        Mutated individual
    """
    tmp = individual.copy()
    i = rng.randint(0, len(tmp))
    tmp[i] = (rng.rand() - 0.5) * range_
    return tmp


def distance(i1: np.array, i2: np.array, *args: List) -> float:
    """
    Distance between 2 individuals

    Args:
        i1: first individual
        i2: second individual
        *args: others

    Returns:
        Distance
        k
    """
    return np.linalg.norm((i1 - i2) ** 2)


if __name__ == "__main__":
    setting = GESettings(
        population_size=10,
        mutation_rate=0.1,
        crossover=avg_crossover,
        mutation=mutation,
        selection=nbest_selection,
        individual_generator=lambda rng: rng.rand(2),
        fitness_function=lambda i, *args: -np.sum(i ** 2),
        distance_function=distance,
    )
    ge = GE(setting)
    for i in range(100):
        ge.iterate()
        print(
            "Avg Fitness:", np.average(ge.fitness), "- Max Fitness:", np.max(ge.fitness)
        )
