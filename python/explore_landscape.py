import numpy as np
import matplotlib.pyplot as plt
import os
from concurrent.futures import ProcessPoolExecutor
from VesicleShapesPietrov2 import Residuales_DoubleShooting, read_best_params, deg_to_rad,optimize_for_angle

def compute_cost(deg,initial_guess,Rparticle,Rvesicle):
    result_ds, boundary_conditions_ds, (A, A0, Astar), rpa = optimize_for_angle(deg, initial_guess, Rparticle, Rvesicle)
    return result_ds.cost

def eval_grid_point(args):
    omega, sigma, u0, u_contact, deg, Rparticle, Rvesicle = args
    initial_guess = [omega, sigma, u0, u_contact]
    return compute_cost(deg, initial_guess, Rparticle, Rvesicle)

def explore():
    os.makedirs('plot', exist_ok=True)
    Rparticle = 3
    Rvesicle = 30
    deg = 45
    rpa = Rparticle / Rvesicle
    
    # saved = read_best_params(filename, opt_params, rpa, deg)
    # if (rpa, deg) not in saved:
    #     print(f"Skipping. No saved params for rpa={rpa} deg={deg}. Please run a sweep first.")
    #     opt_params = [1.0, 0.01, 1.0, 1.0]
    #     opt_cost = np.inf
    # else:
    #     opt_params, opt_cost = saved[(rpa, deg)]
    
    # omega_opt, sigma_opt, u0_opt, u_contact_opt = opt_params
    
    psistar = np.pi + deg_to_rad(deg)
    ustar = 1.0 / rpa
    xstar = rpa * np.sin(deg_to_rad(deg))
    
    A = 4*np.pi*Rvesicle**2  # full vesicle
    A0 = 2*np.pi*Rparticle**2*(1-np.cos(deg_to_rad(deg)))  # wrapped area
    Astar = (A - A0)/Rvesicle**2  # unbound area dimensionless
    
    bc = [psistar, xstar, Astar]
    
    print(f"Exploring landscape for rpa={(rpa):.2f}, deg={deg}")
    print(f"Optimal params: {opt_params}")
    print(f"Optimal cost: {opt_cost}")
    
    n_points = 25
    
    # 1. Slice: omega vs sigma
    omega_range = np.linspace(0.1,2.0, n_points)
    # sigma can be negative sometimes, center around it 
    sigma_range = np.linspace(-0.1, 0.1, n_points)
    
    omega_grid, sigma_grid = np.meshgrid(omega_range, sigma_range)
    cost_grid_os = np.zeros_like(omega_grid)
    
    print("Computing omega vs sigma landscape on grid...")
    args_os = [(omega_grid[i, j], sigma_grid[i, j], u0_opt, u_contact_opt, deg, Rparticle, Rvesicle) for i in range(n_points) for j in range(n_points)]
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results_os = list(executor.map(eval_grid_point, args_os))
        
    idx = 0
    for i in range(n_points):
        for j in range(n_points):
            cost_grid_os[i, j] = results_os[idx]
            idx += 1
            
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(omega_grid, sigma_grid, np.log10(cost_grid_os + 1e-15), levels=50, cmap='viridis')
    plt.colorbar(cp, label='Log10 Cost')
    # plt.plot(omega_opt, sigma_opt, 'r*', markersize=15, label='Optimal')
    plt.xlabel('omega')
    plt.ylabel('sigma')
    plt.title(f'Cost Landscape (omega vs sigma)\nrpa={rpa:.2f}, deg={deg}')
    # plt.legend()
    plt.savefig('plot/landscape_omega_sigma.png')
    plt.close()
    
    # 2. Slice: u0 vs u_contact
    u0_range = np.linspace(7, 10.0, n_points)
    uc_range = np.linspace(0.5, 1.5, n_points)
    
    u0_grid, uc_grid = np.meshgrid(u0_range, uc_range)
    cost_grid_u = np.zeros_like(u0_grid)
    
    print("Computing u0 vs u_contact landscape on grid...")
    args_u = [(omega_opt, sigma_opt, u0_grid[i, j], uc_grid[i, j], deg, Rparticle, Rvesicle) for i in range(n_points) for j in range(n_points)]
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results_u = list(executor.map(eval_grid_point, args_u))
        
    idx = 0
    for i in range(n_points):
        for j in range(n_points):
            cost_grid_u[i, j] = results_u[idx]
            idx += 1
            
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(u0_grid, uc_grid, np.log10(cost_grid_u + 1e-15), levels=50, cmap='viridis')
    plt.colorbar(cp, label='Log10 Cost')
    # plt.plot(u0_opt, u_contact_opt, 'r*', markersize=15, label='Optimal')
    plt.xlabel('u0')
    plt.ylabel('u_contact')
    plt.title(f'Cost Landscape (u0 vs u_contact)\nrpa={rpa:.2f}, deg={deg}')
    # plt.legend()
    plt.savefig('plot/landscape_u0_ucontact.png')
    plt.close()
    
    print("Done. Saved in plot directory.")

if __name__ == '__main__':
    explore()
