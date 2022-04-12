"""
This module contains the Rosenbrock function.

Implemented by Alexandra Gianzina

"""
import numpy as np
import matplotlib.pyplot as plt


class Rosenbrock:
    """
    2-D Rosenbrock function.

    f(x) = (a - x1) ** 2 + b * (x2 - x1**2)**2

    Usage:
        r = Rosenbrock(0, 2)
        r(3) # evaluate at 3

        r = Rosenbrock(0, 2)
        r(np.array([[0,0], [1, 1]])) # evaluate at (0,0) and (1,1)

    Args:
        a: 'a' parameter of the Rosenbrock function
        b: 'b' parameter of the Rosenbrock function
    """

    def __init__(self, a: float, b: float) -> np.array:
        self.a: float = a
        self.b: float = b

    def __call__(self, x: np.array) -> np.array:
        """
        Evaluate the function.

        Args:
            x: input value. Each row is a point.
                m x n: m points in n dimensions.
                When evaluating a single point make sure that the shape of x
                is (1, n)

        Returns:
            Array of size m with the Rosenbrock value for each point

        """
        return (self.a - x[:, 0]) ** 2 + self.b * (x[:, 1] - x[:, 0] ** 2) ** 2


if __name__ == "__main__":
    a = np.array([[1, 2], [3, 4], [5, 6]])
    print(a[:, 0])
    print(a[:, 1])
    rosie = Rosenbrock(0, 2)
    temp = rosie(a)
    print(temp)

    x = np.linspace(-1, 1, 30)
    y = np.linspace(-1, 1, 30)

    X, Y = np.meshgrid(x, y)
    argument = np.dstack((X, Y)).reshape(-1, 2)
    Z = rosie(argument)
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot_surface(
        X, Y, Z.reshape(30, 30), rstride=1, cstride=1, cmap="viridis", edgecolor="none"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.view_init(41, -121)
    plt.show()
