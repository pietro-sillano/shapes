import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
from scipy.integrate import solve_ivp
from core_math import InitialArcLength, InitialValues, ShapeIntegrator, deg_to_rad

def compute_base_energies(row):
    rpa = float(row['rpa'])
    deg = float(row['deg'])
    omega = float(row['omega'])
    sigma = float(row['sigma'])
    u0 = float(row['u0'])
    u_contact = float(row['ustar'])
    
    phi = deg_to_rad(deg)
    psistar = np.pi + phi
    xstar = rpa * np.sin(phi)

    cos_psi = np.cos(psistar)
    if abs(cos_psi) < 1e-6:
        return np.nan, phi, rpa

    gamma_contact_ds = (2 * np.pi / cos_psi) * (sigma * xstar + 0.5 * (np.sin(psistar)**2)/xstar - 0.5 * xstar * u_contact**2)

    s_init = InitialArcLength(omega, u0)
    z_south_init = InitialValues(s_init, omega, u0, sigma, 1.0)
    
    s_mid = 0.5
    sol_south = solve_ivp(ShapeIntegrator, [s_init, s_mid], y0=z_south_init, args=(omega, sigma), method='Radau', rtol=1e-8, atol=1e-8, t_eval=np.linspace(s_init, s_mid, 500))
    
    z_contact_init = [psistar, u_contact, gamma_contact_ds, xstar]
    sol_contact = solve_ivp(ShapeIntegrator, [1.0, s_mid], y0=z_contact_init, args=(omega, sigma), method='Radau', rtol=1e-8, atol=1e-8, t_eval=np.linspace(1.0, s_mid, 500))
    
    if not sol_south.success or not sol_contact.success:
        return np.nan, phi, rpa

    t_contact = np.flip(sol_contact.t)[1:]
    y_contact = np.flip(sol_contact.y, axis=1)[:, 1:]
    
    t_full = np.concatenate((sol_south.t, t_contact))
    y_full = np.concatenate((sol_south.y, y_contact), axis=1)

    sort_idx = np.argsort(t_full)
    t_full = t_full[sort_idx]
    y_full = y_full[:, sort_idx]

    psi = y_full[0, :]
    u = y_full[1, :]
    gamma = y_full[2, :]
    x = y_full[3, :]

    integrand = x/2 * (u + np.sin(psi)/x)**2 + sigma*x + gamma*(x - np.cos(psi))
    Eun = 2 * np.pi * np.trapezoid(integrand, t_full)
    
    return Eun, phi, rpa

def main():
    target_rpa = 0.267
    file_path = "params.csv"
    df = pd.read_csv(file_path, dtype=np.float64)
    
    # Filter by rpa
    df_filtered = df[np.isclose(df['rpa'], target_rpa, atol=1e-1)].copy()
    
    base_results = []
    for index, row in df_filtered.iterrows():
        Eun, phi, rpa = compute_base_energies(row)
        if not np.isnan(Eun):
            base_results.append({'deg': row['deg'], 'phi': phi, 'Eun': Eun, 'rpa': rpa})
            
    df_base = pd.DataFrame(base_results)
    
    W_values = np.linspace(5, 20, 25)
    min_Etot_vals = []
    min_Eun_vals = []
    min_Ebo_vals = []
    min_degs = []
    
    for W in W_values:
        # compute Ebo and Etot for all degrees for this W
        Ebo_array = (-2 * np.pi * W * df_base['rpa']**2 + 4 * np.pi * df_base['rpa']**2) * (1 - np.cos(df_base['phi']))
        Etot_array = df_base['Eun'] + Ebo_array
        
        # Find combination that minimizes Etot
        min_idx = Etot_array.idxmin()
        min_Etot_vals.append(Etot_array[min_idx])
        min_Eun_vals.append(df_base.loc[min_idx, 'Eun'])
        min_Ebo_vals.append(Ebo_array[min_idx])
        min_degs.append(df_base.loc[min_idx, 'deg'])
        
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.plot(W_values, min_Etot_vals, 'k-', linewidth=2, label='Total Energy')
    ax1.plot(W_values, min_Eun_vals, 'b--', linewidth=2, label='Unbound Energy')
    ax1.plot(W_values, min_Ebo_vals, 'r-.', linewidth=2, label='Bound Energy')
    
    ax1.set_xlabel('Adhesive Coefficient (W)')
    ax1.set_ylabel('Energy')
    ax1.set_title(f'Energy Decomposition vs W (Optimal Degree) for RPA = {target_rpa:.3f}')
    ax1.grid(True)
    ax1.legend(loc='center left')
    
    # Plot the optimal degree on a secondary axis
    # ax2 = ax1.twinx()
    # ax2.plot(W_values, min_degs, 'g:', linewidth=2, label='Optimal Degree')
    # ax2.set_ylabel('Optimal Degree', color='g')
    # ax2.tick_params(axis='y', labelcolor='g')
    # ax2.legend(loc='center right')
    
    plt.tight_layout()
    plt.savefig(f'plot_energy_W_sweep_rpa_{target_rpa}.png', dpi=300, transparent=True)
    plt.close()

    # Second plot: Total Energy vs Angle phi for 5 different W values
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Pick 5 W values from the linspace array (or boundaries if array size < 5)
    indices = np.linspace(0, len(W_values)-1, 5, dtype=int)
    five_Ws = W_values[indices]
    
    df_base_sorted = df_base.sort_values(by="deg")
    
    for W in five_Ws:
        Ebo_array = (-2 * np.pi * W * df_base_sorted['rpa']**2 + 4 * np.pi * df_base_sorted['rpa']**2) * (1 - np.cos(df_base_sorted['phi']))
        Etot_array = df_base_sorted['Eun'] + Ebo_array
        ax.plot(df_base_sorted['deg'], Etot_array, marker='o', label=f'W = {W:.2f}')
        
    ax.set_xlabel('Wrap Angle $\\degree$')
    ax.set_ylabel('Total Energy ($E_{tot}$)')
    ax.set_title(f'Total Energy vs Wrap Angle for various W (RPA = {target_rpa:.3f})')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'plot_Etot_vs_phi_rpa_{target_rpa}.png', dpi=300, transparent=True)
    plt.close()

    # Third plot: Heatmap of Total Energy vs W and Wrap Angle
    # We create a 2D grid: rows represent W, columns represent Wrap Angle (deg)
    phis = df_base_sorted['phi'].values
    Euns = df_base_sorted['Eun'].values
    rp = df_base_sorted['rpa'].values[0]
    print(phis)
    Etot_grid = np.zeros((len(W_values), len(phis)))
    
    for i, W in enumerate(W_values):
        Ebo_row = (-2 * np.pi * W * rp**2 + 4 * np.pi * rp**2) * (1 - np.cos(phis))
        Etot_grid[i, :] = Euns + Ebo_row
        
    fig, ax = plt.subplots(figsize=(10, 8))
       
    deg_mesh,W_mesh = np.meshgrid(phis, W_values)
    
    # We use contourf for a smooth heatmap representation
    contours = ax.contourf(deg_mesh, W_mesh, Etot_grid, 50, cmap='viridis')
    cbar = fig.colorbar(contours, ax=ax)
    cbar.set_label('Total Energy ($E_{tot}$)', rotation=270, labelpad=15)
    
    # Optionally overlay the optimal degree line
    # ax.plot(min_degs, W_values, 'r--', linewidth=2, label='Optimal Path')
    
    ax.set_xlabel('Wrap Angle $\\degree$')
    ax.set_ylabel('Adhesive Coefficient (W)')
    ax.set_title(f'Heatmap of Total Energy (RPA = {target_rpa:.3f})')
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'plot_Etot_heatmap_rpa_{target_rpa}.png', dpi=300, transparent=True)
    plt.close()

if __name__ == "__main__":
    main()
