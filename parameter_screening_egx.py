import numpy as np
from quant_met import plotting, utils, hamiltonians
import pandas as pd
import time
from scipy import optimize

lattice_constant = np.sqrt(3)

all_K_points = 4 * np.pi / (3 * lattice_constant) * np.array([
    (np.sin(i * np.pi / 6), np.cos(i * np.pi / 6)) for i in [1, 3, 5, 7, 9, 11]]
)

Gamma = np.array([0, 0])
M = np.pi / lattice_constant * np.array([1, 1 / np.sqrt(3)])
K = 4 * np.pi / (3 * lattice_constant) * np.array([1, 0])

points = [(M, 'M'), (Gamma, r'\Gamma'), (all_K_points[1], 'K')]

band_path, band_path_plot, ticks, labels = plotting.generate_bz_path(points, number_of_points=1000)

BZ_grid = utils.generate_uniform_grid(30, 30, all_K_points[1], all_K_points[5], origin=np.array([0, 0]))

U_range = np.linspace(start=0.1, stop=6, num=10)
beta = None

for V in [0.1, 1, 2]:
    for mu in [0, -0.5, -1, -1.5, -2]:
        gap_size_vs_U = pd.DataFrame(columns=['U', 'delta_0', 'delta_1', 'delta_2'], index=range(len(U_range)))
        gap_size_vs_U['U'] = U_range

        for i, U in enumerate(U_range):
            egx_h = hamiltonians.EGXHamiltonian(t_gr=1, t_x=0.01, a=lattice_constant, V=V, mu=mu, U_gr=U, U_x=U)

            start = time.time()
            solution = optimize.brute(
                func=hamiltonians.free_energy,
                args=(
                    egx_h,
                    BZ_grid,
                    beta,
                ),
                ranges=[(0, 1.5) for _ in range(egx_h.number_of_bands)],
                Ns=50,
                workers=-1,
                finish=optimize.fmin,
                full_output=True,
            )
            print(f'mu = {mu}, U = {U}, solution: {solution}')
            end = time.time()
            print(f'Time taken to solve the gap equation: {end - start:0.2f} seconds')
            gap_size_vs_U.loc[i, ['delta_0', 'delta_1', 'delta_2']] = solution

        gap_size_vs_U.to_csv(f'gap_plots/gap_size_vs_U_V_{V}_mu_{mu}.csv')
