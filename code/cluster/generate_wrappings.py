from VesicleShapesPietrov2 import *
from scipy.optimize import differential_evolution

import numpy as np
plt.rc('text', usetex=False)


Rvesicle = 30.0
Rparticle_list = [3.0, 4.0, 5.0]
phi_list = [15, 20, 25, 30, 35, 40, 45, 60, 90, 120, 150, 165]


omega = 3.0
sigma = 0.0
u0 = 0.0

for Rparticle in Rparticle_list:
    rpa = Rparticle / Rvesicle

    for phi_deg in phi_list:
        phi = deg_to_rad(phi_deg)

        # Boundary conditions
        psistar = np.pi + phi
        ustar = 1/rpa
        xstar = rpa*np.sin(phi)

        # check on xstar value
        if xstar < 0.035:
            print(f"xstar:{xstar}")
            # raise ValueError('xstar too small, can lead to divergences')
            print('xstar too small, can lead to divergences')
            continue

        boundary_conditions = [psistar, ustar, xstar]
        # print(f"boundary_conditions:{boundary_conditions}")
        free_params_extended = [omega, sigma, u0]

        # shoting algorithm and solver
        # result = least_squares(Residuales, free_params_extended, args=(
        #     [boundary_conditions]), method='lm', verbose=1)

        # result = least_squares(Residuales, free_params_extended, args=(
        #     [boundary_conditions]), ftol=1e-15,  method='trf', verbose=1, bounds=[(0, 0, 0), (np.pi, 1e3, 1e3)])

        result = differential_evolution(Residuales_scalar, bounds=[(
            0, np.pi+0.1), (-10, 10), (0, 10)], x0=free_params_extended, args=([boundary_conditions]), polish=True,  disp=True, workers=-1, seed=42)

        best_parameters = result.x
        sol = ShapeCalculator(best_parameters)
        PlotShapes(sol, best_parameters, rpa, phi_deg)

        omega, sigma, u0 = best_parameters

        # print(omega, sigma, u0)

        # if not (rpa, phi_deg) in dict_params.keys():
        save_best_params(best_parameters, rpa, phi_deg)

        # save_coords_file(sol, rpa, phi_deg)
