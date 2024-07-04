import numpy as np
from quant_met import utils

if __name__ == "__main__":
    lattice_constant = np.sqrt(3)

    all_K_points = (
        4
        * np.pi
        / (3 * lattice_constant)
        * np.array(
            [
                (np.sin(i * np.pi / 6), np.cos(i * np.pi / 6))
                for i in [1, 3, 5, 7, 9, 11]
            ]
        )
    )

    BZ_grid = utils.generate_uniform_grid(
        20, 20, all_K_points[1], all_K_points[5], origin=np.array([0, 0])
    )
