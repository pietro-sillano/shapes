import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
from shape_solver import *

W = 0.25

df_base = pd.read_csv('results/energies.csv')

# Get unique RPA values
rpas = df_base['rpa'].unique()

plt.figure(figsize=(10, 6))

# rpa,phi_deg,F_me_un,F_me_bo,F_ad,cost


for rpa in rpas:
    # 1. Filter for the current RPA
    mask = np.isclose(df_base['rpa'], rpa, atol=1e-2)
    df_sub = df_base[mask].copy()

    # 2. SORT by 'deg' to ensure the line connects points in order
    df_sub = df_sub.sort_values(by='phi_deg')
    
    F_tot = df_sub['F_me_un'] + df_sub['F_me_bo'] + W * df_sub['F_ad']

    # 4. Plot with a line ('-o') to see the smooth transition
    plt.plot(df_sub['phi_deg'], F_tot, '-o', label=f"tot RPA: {rpa:.2f}", markersize=4)
    plt.plot(df_sub['phi_deg'], df_sub['F_me_un'], '--', label=f"unb RPA: {rpa:.2f}", markersize=4)

# Formatting
plt.title(fr"Adhesive strength density $W = {W}$ [$\kappa k_B T / l^2$]")
plt.xlabel(r"Contact Angle $\phi$ [deg]")
plt.ylabel("Total Energy $E_{tot}$")
# plt.ylim(-50,50)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(title="RPA Values", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

plt.savefig(f'energy_rpa_{W}.png', dpi=300, transparent=True)
plt.show()