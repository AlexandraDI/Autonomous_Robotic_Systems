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

    def __call__(self, x: np.array, *args) -> np.array:
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
        return 10 * self.n + np.sum(temp, axis=-1)
