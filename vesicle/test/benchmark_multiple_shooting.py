import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utilities import read_best_params, save_best_params
from VesicleShapesUnified import optimize_for_angle_ms,reconstruct_and_save_plot_ms

def main():
    Rparticle = 3
    Rvesicle = 30

    rpa = Rparticle / Rvesicle
    
    os.makedirs("plot", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    params_file = 'test_params_ms.csv'
    original_params_file = '../params.csv'

    start_deg = 20
    end_deg = 150
    step_deg = 20
    rpa_alt = 0.08333333
    degrees = np.arange(start_deg, end_deg + step_deg, step_deg)
    num_segments = 4


    default_guess = [1.0, 0.01, 1.0, 1.0]

    for deg in degrees:
        if read_best_params(params_file, rpa, deg) != None:
            print("using saved test params")
            current_guess = read_best_params(params_file, rpa, deg)

        elif read_best_params(original_params_file, rpa, deg) != None:
            print("using saved original params")
            current_guess = read_best_params(original_params_file, rpa, deg)

        elif read_best_params(params_file, rpa, deg-step_deg) != None:
            print(f"using saved test params from {deg-step_deg}")
            current_guess = read_best_params(params_file, rpa, deg-step_deg)

        elif read_best_params(original_params_file, rpa, deg-step_deg) != None:
            print(f"using saved original params from {deg-step_deg}")
            current_guess = read_best_params(original_params_file, rpa, deg-step_deg)
            
        else:
            print(f"using default params")
            current_guess = default_guess

        if deg > 80 and deg < 100:
            continue

        result_ms, bc, rpa, n_seg = optimize_for_angle_ms(deg, current_guess, Rparticle, Rvesicle, num_segments)
        
        if result_ms.cost < 1e-4:
            save_best_params(params_file, result_ms.x, rpa, deg, result_ms.cost)
        else:
            print(f"Warning: Poor convergence for Multiple Shooting at deg {deg} cost {result_ms.cost}")

        if not result_ms.success and result_ms.cost > 10:
            print(f"FAILURE for Multiple Shooting at deg {deg}")

        reconstruct_and_save_plot_ms(deg, result_ms.x, bc, rpa, n_seg, filename="ms")

if __name__ == "__main__":
    main()
