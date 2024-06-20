from scipy.optimize import differential_evolution
import pickle
import sys
import numpy as np
from VesicleShapesPietrov2 import *  # nopep8


from joblib import effective_n_jobs
max_jobs = effective_n_jobs(-1)
print("effective jobs avail:", )


Rvesicle = 30.0
Rparticle = 4.0
rpa = Rparticle/Rvesicle
phi_deg = 30.0
phi = deg_to_rad(phi_deg)

psistar = np.pi + phi
ustar = 1/rpa
xstar = rpa*np.sin(phi)


N = 100
params = np.zeros((N, 3))
init_params = np.zeros((N, 3))
cost = np.zeros((N))
result_dict = {}


Rvesicle = 30.0
Rparticle = 4.0
rpa = Rparticle/Rvesicle
phi_deg = 30.0
phi = deg_to_rad(phi_deg)

psistar = np.pi + phi
ustar = 1/rpa
xstar = rpa*np.sin(phi)

omega_list = np.linspace(2.5, np.pi, N)
for i, omega in enumerate(omega_list):

    # omega0 = np.array([2.27289])
    sigma = np.array([0.])
    u0 = np.array([0.])

    # Boundary conditions
    psistar = np.pi + phi
    ustar = 1/rpa
    xstar = rpa*np.sin(phi)

    # check on xstar value
    if xstar < 0.035:
        print(f"xstar:{xstar}")
        raise ValueError('xstar too small, can lead to divergences')

    boundary_conditions = [psistar, ustar, xstar]
    # print(f"boundary_conditions:{boundary_conditions}")
    free_params_extended = [omega, sigma[0], u0[0]]
    init_params[i] = free_params_extended
    # print(f"free_params_extended:{free_params_extended}")

    result = differential_evolution(Residuales_scalar, bounds=[(0, np.pi+0.1), (-10, 10), (0, 10)], x0=free_params_extended, args=(
        [boundary_conditions]), polish=True,  disp=True, workers=max_jobs, seed=42)

    params[i] = result.x
    cost[i] = result.fun
    print(params[:i, :])
    np.save('params.npy', params)
    np.save('init_params.npy', init_params)
    np.save('cost.npy', cost)
    result_dict[i] = result

    with open('result_dict', 'wb') as f:
        pickle.dump(result_dict, f)
