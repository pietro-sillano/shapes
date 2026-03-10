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
        return (np.nan,) * 7

    gamma_contact_ds = (2 * np.pi / cos_psi) * (sigma * xstar + 0.5 * (np.sin(psistar)**2)/xstar - 0.5 * xstar * u_contact**2)

    s_init = InitialArcLength(omega, u0)
    z_south_init = InitialValues(s_init, omega, u0, sigma, 1.0)
    
    s_mid = 0.5
    sol_south = solve_ivp(ShapeIntegrator, [s_init, s_mid], y0=z_south_init, args=(omega, sigma), method='Radau', rtol=1e-8, atol=1e-8, t_eval=np.linspace(s_init, s_mid, 500))
    
    z_contact_init = [psistar, u_contact, gamma_contact_ds, xstar]
    sol_contact = solve_ivp(ShapeIntegrator, [1.0, s_mid], y0=z_contact_init, args=(omega, sigma), method='Radau', rtol=1e-8, atol=1e-8, t_eval=np.linspace(1.0, s_mid, 500))
    
    if not sol_south.success or not sol_contact.success:
        return (np.nan,) * 7

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
    integrand = x/2 * (u + np.sin(psi)/x)**2 + sigma*x
    Eun = 2 * np.pi * omega * np.trapezoid(integrand, t_full)

    # Ebo: Bending + Adhesive energy for the bound part
    Ebo = (-2 * np.pi * W * rpa**2 + 4 * np.pi * rpa**2) * (1 - np.cos(phi))
    
    Etot = Eun + Ebo

    # Calculate Area
    A_un = 2 * np.pi * omega * np.trapezoid(x, t_full)
    A_bo = 2 * np.pi * rpa**2 * (1 - np.cos(phi))

    # Calculate Volume
    V_un = np.pi * omega * np.trapezoid(x**2 * np.sin(psi), t_full)
    V_bo = (np.pi / 3) * rpa**3 * (1 - np.cos(phi))**2 * (2 + np.cos(phi))

    return Etot, Eun, Ebo, A_un, A_bo, V_un, V_bo

def main():
    W = 30.0  # Adhesive coefficient
    file_path = "params.csv"
    df = pd.read_csv(file_path, dtype=np.float64)

    results = []

    # Filter out or handle invalid rpa/deg combinations if necessary
    for index, row in df.iterrows():
        Etot, Eun, Ebo, A_un, A_bo, V_un, V_bo = compute_energies(row, W)
        results.append((row['rpa'], row['deg'], Etot, Eun, Ebo, A_un, A_bo, V_un, V_bo))

    df_energies = pd.DataFrame(results, columns=['rpa', 'deg', 'Etot', 'E_un', 'E_bo', 'A_un', 'A_bo', 'V_un', 'V_bo'])
    df_energies['E_tot'] = df_energies['E_un'] + df_energies['E_bo']
    df_energies['A_tot'] = df_energies['A_un'] + df_energies['A_bo']
    df_energies['V_tot'] = df_energies['V_un'] + df_energies['V_bo']

    # Plot Energy
    fig, ax = plt.subplots(figsize=(8, 6))
    for rpa, group in df_energies.groupby("rpa"):
        group = group.sort_values(by="deg").dropna()
        ax.plot(group["deg"], group["Etot"], marker='o', label=f'RPA = {rpa:.2f}')
    ax.set_title(f"Total Energy vs Wrap Angle $\\degree$ (W={W})")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Total Energy")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"plot_energy_tot_W_{W}.png", dpi=300, transparent=True)
    plt.close()

    # Plot Area
    fig, ax = plt.subplots(figsize=(8, 6))
    for rpa, group in df_energies.groupby("rpa"):
        group = group.sort_values(by="deg").dropna()
        ax.plot(group["deg"], group["A_tot"], marker='o', label=f'Total Area (RPA = {rpa:.2f})')
        ax.plot(group["deg"], group["A_un"], linestyle='--', color=ax.lines[-1].get_color())
        ax.plot(group["deg"], group["A_bo"], linestyle=':', color=ax.lines[-1].get_color())

    ax.set_title(f"Area vs Wrap Angle $\\degree$")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Area")
    ax.grid(True)
    # Put legend outside to avoid cluttering
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"plot_area.png", dpi=300, transparent=True)
    plt.close()

    # Plot Volume
    fig, ax = plt.subplots(figsize=(8, 6))
    for rpa, group in df_energies.groupby("rpa"):
        group = group.sort_values(by="deg").dropna()
        ax.plot(group["deg"], group["V_tot"], 'o',label=f'Total Volume (RPA = {rpa:.2f})')
        ax.plot(group["deg"], group["V_un"], linestyle='--', color=ax.lines[-1].get_color())
        ax.plot(group["deg"], group["V_bo"], linestyle=':', color=ax.lines[-1].get_color())

    ax.set_title(f"Volume vs Wrap Angle $\\degree$")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Volume")
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"plot_volume.png", dpi=300, transparent=True)
    plt.close()


    # Plot Energy
    fig, ax = plt.subplots(figsize=(8, 6))
    for rpa, group in df_energies.groupby("rpa"):
        group = group.sort_values(by="deg").dropna()
        ax.plot(group["deg"], group["E_tot"], marker='o', label=f'Total Energy (RPA = {rpa:.2f})')
        ax.plot(group["deg"], group["E_un"], linestyle='--', color=ax.lines[-1].get_color())
        ax.plot(group["deg"], group["E_bo"], linestyle=':', color=ax.lines[-1].get_color())

    ax.set_title(f"Energy vs Wrap Angle $\\degree$")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Energy")
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"plot_energy_W_{W}.png", dpi=300, transparent=True)
    plt.close()

if __name__ == "__main__":
    main()
