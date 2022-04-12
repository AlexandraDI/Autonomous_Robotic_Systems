"""
visualization
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
)


class Plotting:
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

        # Best individual
        # TODO choose if we want to keep this
        if function is not None:
            x = np.linspace(*range_, subdiv)
            y = np.linspace(*range_, subdiv)
            X, Y = np.meshgrid(x, y)
            points = np.dstack((X, Y)).reshape(-1, 2)

            # Sample the function
            Z = function(points).reshape(subdiv, subdiv)

            fig, ax = plt.subplots()
            ax.set_title(f"Best individuals - {title}")
            ax.set_ylabel("y")
            ax.set_xlabel("x")
            ax.set_xlim(*range_)
            ax.set_ylim(*range_)
            # ax.imshow(Z)
            ax.scatter(*zip(*bests))
            fig.show()
        plt.show()
