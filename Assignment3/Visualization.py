"""
Visualization

implemented by Alexandra Gianzina
"""
from typing import List, Optional, Tuple, Callable
from matplotlib import pyplot as plt
import numpy as np
from GE import (
    GESettings,
    GE,
    avg_crossover,
    mutation,
    nbest_selection,
    distance,
    one_point_crossover,
    tournament_selection,
    rank_selection,
    proportional_selection,
)
from Rosenbrock import Rosenbrock
from Rastrigin import Rastrigin


class Visualization:
    def __init__(self):
        pass

    def show(
        self,
        diversity: List[float],
        fitness: List[List[float]],
        bests: List[np.array],
        title: Optional[str] = "",
        function: Callable[[np.array], np.array] = None,
        range_: Optional[Tuple[int, int]] = (-1, 1),
        subdiv: Optional[int] = 100,
    ) -> None:
        """
        Plot the graphs

        Args:
            diversity: list of diversity values
            fitness: list of list of fitness, a list for each generation
            title: specify what we are plotting
            function: function optimed by the GE. If None, this plot is skipped
            range_: range_ of function to plot
            subdiv: resolution of the function plot
        """
        max_fitness = [np.max(i) for i in fitness]
        avg_fitness = [np.average(i) for i in fitness]
        std_fitness = [np.std(i) for i in fitness]

        generations = [i for i in range(len(fitness))]

        # Fitness
        fig, ax = plt.subplots()
        ax.set_title(f"Fitness - {title}")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness value")
        ax.plot(generations, max_fitness, "-o", markersize=2, label="Max. fitness")
        ax.errorbar(
            generations,
            avg_fitness,
            fmt="-o",
            markersize=2,
            yerr=std_fitness,
            label="Avg. fitness",
        )
        ax.legend()
        ax.grid()
        fig.show()

        # Diversity
        fig, ax_div = plt.subplots()
        ax_fit = ax_div.twinx()
        ax_div.set_title(f"Diversity & fitness - {title}")
        ax_div.set_xlabel("Generation")
        ax_div.set_ylabel("Diversity value")
        ax_fit.set_ylabel("Fitness value")

        l1 = ax_div.plot(
            generations,
            diversity,
            "-o",
            markersize=2,
            label="Diversity",
            color="tab:blue",
        )
        l2 = ax_fit.plot(
            generations,
            max_fitness,
            "-o",
            markersize=2,
            label="Max. fitness",
            color="tab:orange",
        )

        ls = l1 + l2
        labels = [l.get_label() for l in ls]
        ax_div.legend(ls, labels)
        ax_div.grid()
        fig.show()
        plt.show()


if __name__ == "__main__":
    # function = lambda i, *args: np.clip(-np.sum(i ** 2, axis=-1) + 5, a_min=0, a_max=5)
    function = lambda x, *args: -Rosenbrock(0, 2)(x)

    visualization = Visualization()
    # Change parameters
    setting = GESettings(
        population_size=100,
        mutation_rate=0.05,
        crossover=avg_crossover,
        mutation=mutation,
        selection=tournament_selection,
        individual_generator=lambda rng: rng.rand(2) * 2 - 1,
        fitness_function=function,
        distance_function=distance,
    )
    ge = GE(setting)

    generations = 100
    fitness = [None for i in range(generations)]
    diversity = [None for i in range(generations)]
    best = [None for i in range(generations)]
    for i in range(generations):
        ge.iterate()
        # print(
        #     "Avg Fitness:", np.average(ge.fitness), "- Max Fitness:", np.max(ge.fitness)
        # )
        fitness[i] = ge.fitness
        diversity[i] = ge.get_diversity()
        best[i] = ge.get_best()
    print(f"Rosenbrock - best individual (gen. {generations}):", best[-1])
    print(f"Rosenbrock - fitness (gen. {generations}):", np.max(fitness[-1]))
    visualization.show(diversity, fitness, best, "Rosenbrock")

    ############################################################
    n_dim = 3
    function = lambda x, *args: -Rastrigin(n_dim)(x)

    visualization = Visualization()
    # Change parameters
    setting = GESettings(
        population_size=100,
        mutation_rate=0.05,
        crossover=avg_crossover,
        mutation=mutation,
        selection=tournament_selection,
        individual_generator=lambda rng: rng.rand(n_dim) * 10 - 5,
        fitness_function=function,
        distance_function=distance,
    )
    ge = GE(setting)

    generations = 100
    fitness = [None for i in range(generations)]
    diversity = [None for i in range(generations)]
    best = [None for i in range(generations)]
    for i in range(generations):
        ge.iterate()
        fitness[i] = ge.fitness
        diversity[i] = ge.get_diversity()
        best[i] = ge.get_best()

        # print(
        #     "Avg Fitness:",
        #     np.average(ge.fitness),
        #     "- Max Fitness:",
        #     np.max(ge.fitness),
        #     "- Best:",
        #     best[i],
        # )
    print(f"Rastrigin - best individual (gen. {generations}):", best[-1])
    print(f"Rastrigin - fitness (gen. {generations}):", np.max(fitness[-1]))
    visualization.show(diversity, fitness, best, "Rastrigin")
