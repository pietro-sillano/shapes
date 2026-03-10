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

    integrand = x/2 * (u + np.sin(psi)/x)**2 + sigma*x
    Eun = 2 * np.pi * omega * np.trapezoid(integrand, t_full)
    
    return Eun, phi, rpa


COMPUTE=False


if COMPUTE:
    file_path = "params.csv"
    # Load the full dataset
    df = pd.read_csv(file_path, dtype=np.float64)

    base_results = []
    
    # Iterate through every row in the original dataframe
    for index, row in df.iterrows():
        Eun, phi, rpa = compute_base_energies(row)
        
        if not np.isnan(Eun):
            base_results.append({
                'deg': row['deg'], 
                'phi': phi, 
                'Eun': Eun, 
                'rpa': rpa  # This now stores the specific RPA from each row
            })
            
    df_base = pd.DataFrame(base_results)
    
    # Optional: Sort by rpa and deg to keep the output file organized
    df_base = df_base.sort_values(by=['rpa', 'deg'])
    
    df_base.to_csv('energies.csv', index=False)



df_base = pd.read_csv('energies.csv')
W = 15

# Get unique RPA values
rpas = df_base['rpa'].unique()

plt.figure(figsize=(10, 6))

for rpa in rpas:
    # 1. Filter for the current RPA
    mask = np.isclose(df_base['rpa'], rpa, atol=1e-2)
    df_sub = df_base[mask].copy()

    # 2. SORT by 'deg' to ensure the line connects points in order
    df_sub = df_sub.sort_values(by='deg')

    # 3. Perform calculations on the sorted subset
    E_adh = (-2 * W * np.pi * df_sub['rpa']**2) * (1 - np.cos(df_sub['phi']))
    Ebe_bo = (4 * np.pi * df_sub['rpa']**2) * (1 - np.cos(df_sub['phi']))
    
    E_tot = E_adh + Ebe_bo + df_sub['Eun']
    E_tot = E_tot - E_tot.iloc[0]

    # 4. Plot with a line ('-o') to see the smooth transition
    plt.plot(df_sub['deg'], E_tot, '-o', label=f"RPA: {rpa:.2f}", markersize=4)

# Formatting
plt.title(fr"Adhesive strength density $W = {W}$ [$\kappa k_B T / l^2$]")
plt.xlabel("Contact Angle [deg]")
plt.ylabel("Total Energy $E_{tot}$")
plt.ylim(-20,12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(title="RPA Values", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

plt.savefig(f'energy_rpa_{W}.png', dpi=300, transparent=True)
plt.show()
