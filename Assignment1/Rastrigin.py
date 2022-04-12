"""
This module contains the Rastrigin function.

Implemented by Alexandra Gianzina

"""
import numpy as np
import matplotlib.pyplot as plt


class Rastrigin:
    """
    Rastring function.

    Usage:
        r = Rastring(n = 1) # 1 dimenstion
        r(3) # evaluate at 3

        r = Rastring(n = 2) # 2 dimenstions
        r(np.array([[0,0], [1, 1]])) # evaluate at (0,0) and (1,1)

    Args:
        n: number of dimensions
    """

    def __init__(self, n: int) -> None:
        if n <= 0:
            raise ValueError("The number of dimensions has to be >= 1")
        self.n: int = n

    def __call__(self, x: np.array) -> np.array:
        """
        Evaluate the function.

        Args:
            x: input value. Each row is a point.
                m x n: m points in n dimensions.
                When evaluating a single point make sure that the shape of x
                is (1, n)

        Returns:
            Array of size m with the Restrigin value for each point

        """
        temp = x ** 2 - 10 * np.cos(2 + np.pi * x)
        return 10 * self.n + np.sum(temp, axis=1)


if __name__ == "__main__":
    a = np.array([[1, 2], [3, 4], [5, 6]])
    print(a[:, 0])
    print(a[:, 1])
    rastie = Rastrigin(2)
    temp = rastie(a)
    print(temp)

    x = np.linspace(-6, 6, 30)
    y = np.linspace(-6, 6, 30)

    X, Y = np.meshgrid(x, y)
    Z = rastie(np.dstack((X, Y)).reshape(-1, 2))
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot_surface(
        X, Y, Z.reshape(30, 30), rstride=1, cstride=1, cmap="viridis", edgecolor="none"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()
