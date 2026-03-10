import numpy as np

import VesicleShapesPietrov2 as py
import VesicleShapesPietrov2_numba as nb

def test():
    # Test identical inputs for initial length
    p = (3.0, 0.08)
    s_py = py.InitialArcLength(p)
    s_nb = nb.InitialArcLength(3.0, 0.08)
    print(f"InitialArcLength Py: {s_py}, Numba: {s_nb}")

    # Test initial values
    z_py = py.InitialValues(s_py, (3.0, 0.08, 1.0, 1.0))
    z_nb = nb.InitialValues(s_nb, 3.0, 0.08, 1.0, 1.0)
    print(f"InitialValues Py: {z_py}")
    print(f"InitialValues Numba: {z_nb}")

    # Test integrations
    s = 0.1
    z = [0.1, 0.2, 0.3, 0.4]
    int_py = py.ShapeIntegrator(s, z, 3.0, 1.0)
    int_nb = nb.ShapeIntegrator(s, np.array(z, dtype=np.float64), 3.0, 1.0)
    print(f"Integrator Py: {int_py}")
    print(f"Integrator Numba: {int_nb}")
    
    # Check optimization output
    deg = 35
    Rparticle = 5
    Rvesicle = 30
    initial_guess = [3.0, 0.08, 1.0, Rvesicle/Rparticle]
    
    res_py = py.optimize_for_angle(deg, initial_guess, Rparticle, Rvesicle)
    res_nb = nb.optimize_for_angle(deg, initial_guess, Rparticle, Rvesicle)
    
    print(f"\nPy opt: {res_py[0].x}")
    print(f"Nb opt: {res_nb[0].x}")

if __name__ == '__main__':
    test()
