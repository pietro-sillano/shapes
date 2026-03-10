import numpy as np
import VesicleShapesPietrov2 as py
import VesicleShapesPietrov2_numba as nb

def test():
    params = [3.0, 0.08, 1.0, 6.0]
    boundary_conditions = [3.5, 0.1, 0.5] # psistar, xstar, Astar
    
    res_py = py.Residuales_DoubleShooting(params, boundary_conditions)
    res_nb = nb.Residuales_DoubleShooting(params, boundary_conditions)
    
    print(f"Res Py: {res_py}")
    print(f"Res Nb: {res_nb}")

if __name__ == '__main__':
    test()
