import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
from scipy.integrate import solve_ivp

# Math imports
from core_math import InitialArcLength, InitialValues, ShapeIntegrator, deg_to_rad

def compute_energies(row, W):
    rpa = float(row['rpa'])
    deg = float(row['deg'])
    omega = float(row['omega'])
    sigma = float(row['sigma'])
    u0 = float(row['u0'])
    u_contact = float(row['ustar'])
    
    phi = deg_to_rad(deg)
    psistar = np.pi + phi
    xstar = rpa * np.sin(phi)

    # Reconstruct shape using Double Shooting approach natively
    cos_psi = np.cos(psistar)
    if abs(cos_psi) < 1e-6:
        return np.nan, np.nan, np.nan

    gamma_contact_ds = (2 * np.pi / cos_psi) * (sigma * xstar + 0.5 * (np.sin(psistar)**2)/xstar - 0.5 * xstar * u_contact**2)

    s_init = InitialArcLength(omega, u0)
    z_south_init = InitialValues(s_init, omega, u0, sigma, 1.0)
    
    s_mid = 0.5
    sol_south = solve_ivp(ShapeIntegrator, [s_init, s_mid], y0=z_south_init, args=(omega, sigma), method='Radau', rtol=1e-8, atol=1e-8, t_eval=np.linspace(s_init, s_mid, 500))
    
    z_contact_init = [psistar, u_contact, gamma_contact_ds, xstar]
    sol_contact = solve_ivp(ShapeIntegrator, [1.0, s_mid], y0=z_contact_init, args=(omega, sigma), method='Radau', rtol=1e-8, atol=1e-8, t_eval=np.linspace(1.0, s_mid, 500))
    
    if not sol_south.success or not sol_contact.success:
        return np.nan, np.nan, np.nan

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

    # Calculate Energies
    # Eun: Bending energy of the unbound part
    integrand = x/2 * (u + np.sin(psi)/x)**2 + sigma*x + gamma*(x - np.cos(psi))
    Eun = 2 * np.pi * np.trapezoid(integrand, t_full)

    # Ebo: Bending + Adhesive energy for the bound part
    Ebo = (-2 * np.pi * W * rpa**2 + 4 * rpa**2) * (1 - np.cos(phi))
    
    Etot = Eun + Ebo
    return Etot, Eun, Ebo

def main():
    W = 10.0  # Adhesive coefficient
    file_path = "params.csv"
    df = pd.read_csv(file_path, dtype=np.float64)

    results = []

    # Filter out or handle invalid rpa/deg combinations if necessary
    for index, row in df.iterrows():
        Etot, Eun, Ebo = compute_energies(row, W)
        results.append((row['rpa'], row['deg'], Etot, Eun, Ebo))

    df_energies = pd.DataFrame(results, columns=['rpa', 'deg', 'Etot', 'Eun', 'Ebo'])

    fig, ax = plt.subplots(figsize=(8, 6))

    for rpa, group in df_energies.groupby("rpa"):
        group = group.sort_values(by="deg")
        # Filter NaNs
        group = group.dropna()
        ax.plot(group["deg"], group["Etot"], marker='o', label=f'RPA = {rpa:.2f}')

    ax.set_title(f"Total Energy vs Wrap Angle $\\degree$ (W={W})")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Total Energy")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"plot_energy_{W}.png", dpi=300, transparent=True)
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()
