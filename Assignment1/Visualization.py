"""
Module to visualize the PSO.

It contains the following class:
    - Visualize: show a 3D of a PSO experiment

And the followinf function:
    - main: run a demo for the PSO


Implemented by Alexandra Gianzina

"""
import os
from typing import Optional, Callable, Tuple
import matplotlib.cm
import numpy as np
import matplotlib.pyplot as plt
from Assignment1.PSO import PSO, PSOSettings
from Assignment1.Rastrigin import Rastrigin
from Assignment1.Rosenbrock import Rosenbrock


class Visualization:
    """
    Visualize the PSO.

    Args:
        None
    """

    def __init__(self) -> None:
        self._stop: bool = False

    # code from stack overflow: https://stackoverflow.com/questions/12439588/how-to-maximize-a-plt-show-window-using-python
    def plt_set_fullscreen(self) -> None:
        """
        Set the graph fullsreen.
        This affect only the last pyplof Figure that has been created.
        The behaviour depends on the matplotlib backend and on the OS

        Returns:
            None.

        Note:
            The code is taken and adapted from stack overflow:
            https://stackoverflow.com/questions/12439588/how-to-maximize-a-plt-show-window-using-python
        """
        backend = str(plt.get_backend())
        mgr = plt.get_current_fig_manager()
        if backend == "TkAgg":
            if os.name == "nt":
                mgr.window.state("zoomed")
            else:
                mgr.resize(*mgr.window.maxsize())
        elif backend == "wxAgg":
            mgr.frame.Maximize(True)
        elif backend == "Qt4Agg":
            mgr.window.showMaximized()
        elif backend == "Qt5Agg":
            mgr.window.showMaximized()

    def close(
        self, event: Optional["matplotlib.backend_bases.CloseEvent"] = None
    ) -> None:
        """
        Notify to the Visualize object that the graph has been closed

        Args:
            event: mathplotlib close event (not used)

        Returns:
            None
        """
        # Stop animation if user closes the last (running) figure
        if self.fig == event.canvas.figure:
            self._stop = True

    def visualize(
        self,
        function: Callable[[np.array], np.array],
        settings: PSOSettings,
        graph_subdivisions: int,
        angles: Tuple[int, int],
        interval: int,
        print_all_iterations: bool,
        title: Optional[str] = "PSO",
    ) -> None:
        """
        Visualize and run an experiments

        Args:
            function: function to minimize with PSO. The function should be 2D
            settings: settings for the PSO
            graph_subdivisions: samples used to show the functions
            angles: rotate the graph for a better visualization
            interval: update rate of the animationn
            print_all_iterations: wheter to display all the iteration. If false,
                the animation is updated every 10 iterations
            title: title of the graph

        Returns:
            None.
        """
        self._stop = False
        pso = PSO(settings)

        # Sampling points
        x = np.linspace(*settings.size, graph_subdivisions)
        y = np.linspace(*settings.size, graph_subdivisions)
        X, Y = np.meshgrid(x, y)
        points = np.dstack((X, Y)).reshape(-1, 2)

        # Sample the function
        Z = function(points).reshape(graph_subdivisions, graph_subdivisions)

        # Make a new graph
        self.fig = plt.figure(figsize=(8, 7))
        self.fig.canvas.mpl_connect("close_event", self.close)
        # the graph will be displayed fullscreen
        # self.plt_set_fullscreen()
        ax = plt.axes(projection="3d")
        ax.view_init(angles[0], angles[1])

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(title)

        # Plot the functions
        ax.plot_surface(
            X, Y, Z, rstride=1, cstride=1, cmap="viridis", edgecolor="none", alpha=0.4,
        )

        # Colors for the particles
        colors = matplotlib.cm.get_cmap("gist_rainbow")
        colors = colors(np.linspace(0, 1, settings.particles))

        # Get the position of particles and plot
        # p: x, y
        # v: z
        for p, v in pso.optimize_generator(function):

            # for the rotation of the graph
            # ax.view_init(angles[0], angles[1] + (pso.iteration +2))
            if (
                (print_all_iterations)
                or (pso.iteration % 100 == 0)
                or (pso.iteration < 100)
            ):
                ax.scatter3D(p[:, 0], p[:, 1], v, color=colors, s=20, marker=".")
                plt.pause(interval)
            if pso.iteration % 1000 == 0:
                print(f"-Iteration: {pso.iteration}")
                # print(f"-Avg change: {pso._avg_change}")

            if self._stop:
                break

        # Mark the final result
        optimal_p, optimal_v = pso.get_optimal_value(p, v)
        ax.scatter3D(
            optimal_p[0], optimal_p[1], optimal_v+0.1, color="red", s=350, marker="X"
        )

        plt.show()

        print("Minimum:", optimal_v)
        print("Minimum position:", f"x1: {optimal_p[0]}", f"x2: {optimal_p[1]}")
        print("Iterations:", pso.iteration)


def main() -> None:
    """
    Run the demo

    Returns:
        None
    """
    visualize = Visualization()

    # Rosenbrock - experiment 1
    print("##############################")
    print("# Rosenbrock - Experiment 1 #")
    print("We run the function with the following parameters:")
    print("particles = 5, neighbours = 3")
    setting1 = PSOSettings(particles=5, size=(-1, 1))
    a = 0
    b = 2

    function1 = Rosenbrock(a, b)

    visualize.visualize(
        function1,
        setting1,
        graph_subdivisions=30,
        angles=(41, -121),
        interval=0.1,
        print_all_iterations=True,
        title="Experiment 1: Rosenbrock with particles = 5, neighbours = 3",
    )
    input("Press Enter to continue...")

    # Rosenbrock - experiment 2
    # print("##############################")
    # print("# Rosenbrock - Experiment 2 #")
    # print("We run the function with the following parameters:")
    # print("particles = 10, neighbours = 3")
    # setting1 = PSOSettings(particles=10, size=(-1, 1))
    # a = 0
    # b = 2
    #
    # function1 = Rosenbrock(a, b)
    #
    # visualize.visualize(
    #     function1,
    #     setting1,
    #     graph_subdivisions=30,
    #     angles=(41, -121),
    #     interval=0.1,
    #     print_all_iterations=True,
    #     title="Experiment 2: Rosenbrock with particles = 10, neighbours = 3",
    # )
    # input("Press Enter to continue...")

    # Rosenbrock - experiment 3
    # print("##############################")
    # print("# Rosenbrock - Experiment 3 #")
    # print("We run the function with the following parameters:")
    # print("particles = 20, neighbours = 3")
    # setting1 = PSOSettings(particles=20, size=(-1, 1))
    # a = 0
    # b = 2
    #
    # function1 = Rosenbrock(a, b)
    #
    # visualize.visualize(
    #     function1,
    #     setting1,
    #     graph_subdivisions=30,
    #     angles=(41, -121),
    #     interval=0.1,
    #     print_all_iterations=True,
    #     title="Experiment 3: Rosenbrock with particles = 20, neighbours = 3",
    # )
    # input("Press Enter to continue...")

    # Rosenbrock - experiment 4
    # print("##############################")
    # print("# Rosenbrock - Experiment 4 #")
    # print("We run the function with the following parameters:")
    # print("particles = 20, neighbours = 5")
    # setting1 = PSOSettings(particles=20, neighbours=5, size=(-1, 1))
    # a = 0
    # b = 2
    #
    # function1 = Rosenbrock(a, b)
    #
    # visualize.visualize(
    #     function1,
    #     setting1,
    #     graph_subdivisions=30,
    #     angles=(41, -121),
    #     interval=0.1,
    #     print_all_iterations=True,
    #     title="Experiment 4: Rosenbrock with particles = 20, neighbours = 5",
    # )
    # input("Press Enter to continue...")

    # Rosenbrock - experiment 5
    print("##############################")
    print("# Rosenbrock - Experiment 5 #")
    print("We run the function with the following parameters:")
    print("particles = 20, neighbours = 10")
    setting1 = PSOSettings(particles=20, neighbours=10, size=(-1, 1))
    a = 0
    b = 2

    function1 = Rosenbrock(a, b)

    visualize.visualize(
        function1,
        setting1,
        graph_subdivisions=30,
        angles=(41, -121),
        interval=0.1,
        print_all_iterations=True,
        title="Experiment 5: Rosenbrock with particles = 20, neighbours = 10",
    )
    input("Press Enter to continue...")

    # Rastrigin - experiment 1
    print()
    print("############################")
    print("# Rastrigin - Experiment 1 #")
    print("We run the function with the following parameters:")
    print("particles = 5, neighbours = 3")

    setting2 = PSOSettings(particles=5, size=(-5, 5), eps=0.1, a_decay=1e-6)

    dimentions = 2
    function2 = Rastrigin(dimentions)

    visualize.visualize(
        function2,
        setting2,
        graph_subdivisions=60,
        angles=(41, -121),
        interval=0.000001,
        print_all_iterations=False,
        title="Experiment 1: Rastrigin with particles = 5, neighbours = 3",
    )
    input("Press Enter to continue...")

    # Rastrigin - experiment 2
    print()
    print("############################")
    print("# Rastrigin - Experiment 2 #")
    print("We run the function with the following parameters:")
    print("particles = 10, neighbours = 3")

    setting2 = PSOSettings(particles=10, size=(-5, 5), eps=0.1, a_decay=1e-6)

    dimentions = 2
    function2 = Rastrigin(dimentions)

    visualize.visualize(
        function2,
        setting2,
        graph_subdivisions=60,
        angles=(41, -121),
        interval=0.000001,
        print_all_iterations=False,
        title="Experiment 2: Rastrigin with particles = 10, neighbours = 3",
    )
    input("Press Enter to continue...")

    # Rastrigin - experiment 3
    print()
    print("############################")
    print("# Rastrigin - Experiment 3 #")
    print("We run the function with the following parameters:")
    print("particles = 20, neighbours = 3")

    setting2 = PSOSettings(particles=20, size=(-5, 5), eps=0.1, a_decay=1e-6)

    dimentions = 2
    function2 = Rastrigin(dimentions)

    visualize.visualize(
        function2,
        setting2,
        graph_subdivisions=60,
        angles=(41, -121),
        interval=0.000001,
        print_all_iterations=False,
        title="Experiment 3: Rastrigin with particles = 20, neighbours = 3",
    )
    input("Press Enter to continue...")

    # Rastrigin - experiment 4
    print()
    print("############################")
    print("# Rastrigin - Experiment 4 #")
    print("We run the function with the following parameters:")
    print("particles = 20, neighbours = 5")

    setting2 = PSOSettings(
        particles=20, neighbours=5, size=(-5, 5), eps=0.1, a_decay=1e-6
    )

    dimentions = 2
    function2 = Rastrigin(dimentions)

    visualize.visualize(
        function2,
        setting2,
        graph_subdivisions=60,
        angles=(41, -121),
        interval=0.000001,
        print_all_iterations=False,
        title="Experiment 4: Rastrigin with particles = 20, neighbours = 5",
    )
    input("Press Enter to continue...")

    # Rastrigin - experiment 5
    print()
    print("############################")
    print("# Rastrigin - Experiment 5 #")
    print("We run the function with the following parameters:")
    print("particles = 20, neighbours = 10")

    setting2 = PSOSettings(
        particles=20, neighbours=10, size=(-5, 5), eps=0.1, a_decay=1e-6
    )

    dimentions = 2
    function2 = Rastrigin(dimentions)

    visualize.visualize(
        function2,
        setting2,
        graph_subdivisions=60,
        angles=(41, -121),
        interval=0.000001,
        print_all_iterations=False,
        title="Experiment 5: Rastrigin with particles = 20, neighbours = 10",
    )
    input("Press Enter to continue...")


if __name__ == "__main__":
    main()
