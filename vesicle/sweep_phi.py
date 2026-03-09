import numpy as np
import os
from VesicleShapesPietrov2 import  read_best_params, save_best_params

from VesicleShapesPietrov2_multiple_shooting import optimize_for_angle_ms, reconstruct_and_save_plot_ms

def main():
    Rparticle = 5.0
    Rvesicle = 30.0
    rpa = Rparticle / Rvesicle
    # We will write the results to a file
    output_file = "data/sweep_results_rpa_" + f"{rpa:.2f}" + ".dat"
    os.makedirs("data", exist_ok=True)
    
    params_file = 'params.csv'


    start_deg = 15
    end_deg = 1
    step_deg = -1
    rpa_alt = rpa
    degrees = np.arange(start_deg, end_deg + step_deg, step_deg)
    num_segments = 4
    # Initial guess for the very first step (from previous successful run at deg=30)
    # [omega, sigma, u0, u_contact]
    default_guess = [1.0, 0.01, 1.0, 1.0]
    # default_guess = [8.76545115, 0.21125935, 0.21337616, 6.39789426]
    # default_guess = [2.96664071892188,0.11678474718351434,0.9539762087835862,7.012078449148444]
    
    for deg in degrees:
        if read_best_params(params_file, rpa, deg) != None:
            print("using saved params")
            current_guess = read_best_params(params_file, rpa, deg)
        elif read_best_params(params_file, rpa_alt, deg) != None:
            print(f"using saved params from rpa: {rpa_alt} with phi {deg}")
            current_guess = read_best_params(params_file, rpa_alt, deg)
        elif read_best_params(params_file, rpa, deg-step_deg) != None:
            print(f"using saved params from angle: {deg-step_deg}")
            current_guess = read_best_params(params_file, rpa, deg-step_deg)    
            
        else:
            print(f"using default params")
            current_guess = default_guess

        if deg > 80 and deg < 100:
            continue
        # if deg == 90:
        #     continue

        # result_ds, bc, areas, rpa = optimize_for_angle(deg, current_guess, Rparticle, Rvesicle)
        
        # # Generate shape plot for visual inspection
        # reconstruct_and_save_plot(deg, opt_params, bc, rpa,num_segments)
        # print(f"Saved plot plot/wrapped_rpa{rpa:.2f}_phi{deg:.2f}.png")
        
        result_ms, bc, rpa, n_seg = optimize_for_angle_ms(deg, current_guess, Rparticle, Rvesicle,num_segments)

        # Save the optimized parameters
        opt_params = result_ms.x
        cost = result_ms.cost
        save_best_params(params_file,opt_params, rpa, deg, cost)

        reconstruct_and_save_plot_ms(deg, result_ms.x, bc, rpa, n_seg)
        
if __name__ == "__main__":
    main()
