import numpy as np


def fitness_function_float(function_number):
    if function_number == 1:
        return lambda chromosome: -(np.abs(chromosome[0]) + np.cos(chromosome[0]))
    elif function_number == 2:
        return lambda chromosome: -(np.abs(chromosome[0]) + np.sin(chromosome[0]))
    elif function_number == 3:
        return lambda chromosome: -(chromosome**2).sum()
    elif function_number == 4:
        return lambda chromosome: -np.sum(
            np.abs(chromosome) - 10 * np.cos(np.sqrt(np.abs(10 * chromosome)))
        )
    elif function_number == 5:
        return lambda chromosome: -(chromosome[0] ** 2 + chromosome[0]) * np.cos(
            chromosome[0]
        )
    elif function_number == 6:
        return lambda chromosome: -(
            chromosome[0] * np.sin(4 * chromosome[0])
            + 1.1 * chromosome[1] * np.sin(2 * chromosome[1])
        )
    elif function_number == 7:  # Bohachevsky, bound in [-100,100], opt at (0,0)
        return lambda chromosome: -(
            0.7
            + chromosome[0] ** 2
            + 2.0 * chromosome[1] ** 2
            - 0.3 * np.cos(3 * np.pi * chromosome[0])
            - 0.4 * np.cos(4 * np.pi * chromosome[1])
        )

    elif function_number == 8:  # Goldstein, bounds in [-2,2], opt at (0,-1)
        a = 1 + (chromosome[0] + chromosome[1] + 1) ** 2 * (
            19
            - 14 * chromosome[0]
            + 3 * chromosome[0] ** 2
            - 14 * chromosome[1]
            + 6 * chromosome[0] * chromosome[1]
            + 3 * chromosome[1] ** 2
        )
        b = 30 + (2 * chromosome[0] - 3 * chromosome[1]) ** 2 * (
            18
            - 32 * chromosome[0]
            + 12 * chromosome[0] ** 2
            + 48 * chromosome[1]
            - 36 * chromosome[0] * chromosome[1]
            + 27 * chromosome[1] ** 2
        )
        return lambda chromosome: -a * b

    elif function_number == 9:  # Hartmann3, bounds in [0,1], opt at (0.11,0.55,0.85)
        a = np.array(
            [[3.0, 10.0, 30.0], [0.1, 10.0, 35.0], [3.0, 10.0, 30.0], [0.1, 10.0, 35.0]]
        )
        c = np.array([1.0, 1.2, 3.0, 3.2])
        p = np.array(
            [
                [0.36890, 0.11700, 0.26730],
                [0.46990, 0.43870, 0.74700],
                [0.10910, 0.87320, 0.55470],
                [0.03815, 0.57430, 0.88280],
            ]
        )
        return lambda chromosome: np.sum(
            c * np.exp(-np.sum(a * (chromosome[:] - p) ** 2, -1)), -1
        )

    elif (
        function_number == 10
    ):  # Hartmann6, bounds in [0,1]. opt at (0.2, 0.15, 0.47, 0.27, 0.31, 0.65)
        a = np.array(
            [
                [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
                [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
                [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
                [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
            ]
        )
        c = np.array([1.0, 1.2, 3.0, 3.2])
        p = np.array(
            [
                [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
                [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
            ]
        )
        return (
            lambda chromosome: (
                np.sum(c * np.exp(-np.sum(a * (chromosome[:] - p) ** 2, -1)), -1) + 2.58
            )
            / 1.94
        )
