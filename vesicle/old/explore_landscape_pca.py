import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from concurrent.futures import ProcessPoolExecutor
from sklearn.decomposition import PCA
# from VesicleShapesPietrov2 import optimize_for_angle, deg_to_rad, Residuales_DoubleShooting

from VesicleShapesPietrov2_numba import optimize_for_angle, deg_to_rad, Residuales_DoubleShooting

def run_optimization(args):
    omega_g, sigma_g, u0_g, uc_g, deg, Rparticle, Rvesicle = args
    initial_guess = [omega_g, sigma_g, u0_g, uc_g]
    
    psistar = np.pi + deg_to_rad(deg)
    xstar = (Rparticle/Rvesicle) * np.sin(deg_to_rad(deg))
    Astar = (4*np.pi*Rvesicle**2 - 2*np.pi*Rparticle**2*(1-np.cos(deg_to_rad(deg))))/Rvesicle**2
    bc = [psistar, xstar, Astar]
    
    # Calculate starting cost without optimization
    initial_res = Residuales_DoubleShooting(initial_guess, bc)
    initial_cost = sum(np.array(initial_res)**2)
    
    try:
        result_ds, _, _, _ = optimize_for_angle(deg, initial_guess, Rparticle, Rvesicle)
        opt_params = result_ds.x
        final_cost = result_ds.cost
        
        # If optimization fails badly or explodes
        if final_cost > 1e4:
            return (initial_guess, initial_cost, np.array([np.nan]*4), np.inf)
            
        return (initial_guess, initial_cost, opt_params, final_cost)
    except Exception as e:
        return (initial_guess, initial_cost, np.array([np.nan]*4), np.inf)


def explore_pca():
    os.makedirs('plot', exist_ok=True)
    Rparticle = 3
    Rvesicle = 30
    deg = 45
    rpa = Rparticle / Rvesicle
    
    # Define a 4D grid of initial guesses
    omega_vals = np.linspace(2.5, 3.5, 2)
    sigma_vals = np.linspace(-0.1, 0.1, 2)
    u0_vals = np.linspace(0.5, 3.0, 2)
    uc_vals = np.linspace(5.0, 10.0, 2)
    
    print("Generating 4D grid combinations...")
    args_list = []
    for w in omega_vals:
        for s in sigma_vals:
            for u in u0_vals:
                for uc in uc_vals:
                    args_list.append((w, s, u, uc, deg, Rparticle, Rvesicle))
                    
    print(f"Running {len(args_list)} optimizations from different initial guesses. Please be patient...")
    
    results = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(run_optimization, args_list))
        
    initials = []
    initial_costs = []
    valid_opts = []
    valid_costs = []
    
    # Extract
    for init_g, init_c, opt_p, final_c in results:
        initials.append(init_g)
        initial_costs.append(init_c)
        if not np.any(np.isnan(opt_p)) and final_c < 1e5:
            valid_opts.append(opt_p)
            valid_costs.append(final_c)
            
    initials = np.array(initials)
    initial_costs = np.array(initial_costs)
    valid_opts = np.array(valid_opts)
    valid_costs = np.array(valid_costs)
    
    if len(initials) < 2:
        print("Not enough points to PCA.")
        return
        
    # PCA on initial guesses (The true input energy landscape)
    pca_init = PCA(n_components=2)
    init_pca_coords = pca_init.fit_transform(initials)
    
    plt.figure(figsize=(9, 7))
    sc = plt.scatter(init_pca_coords[:, 0], init_pca_coords[:, 1], c=np.log10(initial_costs + 1e-15), cmap='viridis', alpha=0.8)
    plt.colorbar(sc, label='Log10 Initial Cost')
    plt.xlabel(f'PCA1 ({pca_init.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'PCA2 ({pca_init.explained_variance_ratio_[1]*100:.1f}%)')
    plt.title(f'Cost Landscape of 4D Initial Guesses in 2D PCA\nPhi={deg} deg')
    plt.savefig('plot/pca_landscape_initial.png')
    plt.close()
    
    # PCA on final optimized parameters (To identify multiple minima basins)
    opt_pca_coords_full = np.full((len(results), 2), np.nan)
    if len(valid_opts) > 2:
        pca_opt = PCA(n_components=2)
        opt_pca_coords = pca_opt.fit_transform(valid_opts)
        
        valid_idx = 0
        for i, (init_g, init_c, opt_p, final_c) in enumerate(results):
            if not np.any(np.isnan(opt_p)) and final_c < 1e5:
                opt_pca_coords_full[i] = opt_pca_coords[valid_idx]
                valid_idx += 1
                
        plt.figure(figsize=(9, 7))
        sc2 = plt.scatter(opt_pca_coords[:, 0], opt_pca_coords[:, 1], c=np.log10(valid_costs + 1e-15), cmap='plasma', alpha=0.8)
        plt.colorbar(sc2, label='Log10 Optimized Final Cost')
        plt.xlabel(f'PCA1 ({pca_opt.explained_variance_ratio_[0]*100:.1f}%)')
        plt.ylabel(f'PCA2 ({pca_opt.explained_variance_ratio_[1]*100:.1f}%)')
        plt.title(f'Clustering of Final Optimized Parameters in 2D PCA Space\n(Showing local minima basins for Phi={deg} deg)')
        plt.savefig('plot/pca_landscape_optimized.png')
        plt.close()
        
    csv_filename = 'plot/pca_landscape_results.csv'
    print(f"Saving intermediate results to {csv_filename} ...")
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['omega_init', 'sigma_init', 'u0_init', 'uc_init', 'initial_cost', 
                         'pca_init_1', 'pca_init_2',
                         'omega_opt', 'sigma_opt', 'u0_opt', 'uc_opt', 'final_cost',
                         'pca_opt_1', 'pca_opt_2'])
                         
        for i, (init_g, init_c, opt_p, final_c) in enumerate(results):
            row = [
                init_g[0], init_g[1], init_g[2], init_g[3], init_c,
                init_pca_coords[i, 0], init_pca_coords[i, 1],
                opt_p[0], opt_p[1], opt_p[2], opt_p[3], final_c,
                opt_pca_coords_full[i, 0], opt_pca_coords_full[i, 1]
            ]
            writer.writerow(row)
            
    print("Done. Saved PCA plots and results in plot/")

if __name__ == '__main__':
    explore_pca()
