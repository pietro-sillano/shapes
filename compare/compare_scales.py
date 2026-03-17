import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

# Create comparison plots directory
os.makedirs("results", exist_ok=True)

# ── Extract and Reconstruct parameters from existing results ──
# 1. Vesicle Data
vesicle_file = "../vesicle/energies.csv"
try:
    df_vesicle = pd.read_csv(vesicle_file)
except OSError:
    print(f"Could not read {vesicle_file}. Ensure you have run energy calculations in the vesicle directory first.")
    df_vesicle = pd.DataFrame()

# 2. Planar Data
planar_file = "../planar/results/energies.csv"
try:
    df_planar = pd.read_csv(planar_file)
except OSError:
    print(f"Could not read {planar_file}. Ensure you have run shape_solver.py in the planar directory first.")
    df_planar = pd.DataFrame()


def plot_vesicle_vs_planar():
    """
    Overlays total energy from planar vs finite vesicle limits using dimensionless mappings.
    The parameters are mapped directly using lam = sqrt(kappa/sigma) = R_ve / sqrt(sigma_tilde)
    """
    if df_vesicle.empty or df_planar.empty:
        return
        
    # Example constants (Must match exact parameters run in the simulators)
    # Vesicle specific dimensionless tension parameter limits
    kappa = 1.0
    sigma = 0.1
    W = 0.05 # Standard test parameter
    
    # Intrinsic scaling properties mappings
    lam = np.sqrt(kappa / sigma)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Vesicle Plots (using scaled values)
    # The vesicle data uses pure length ratio rpa = R_pa / R_ve
    # We will overlay using intrinsic dimensional lengths (R_pa unscaled from lam mapping)
    for rpa, group in df_vesicle.groupby("rpa"):
        group = group.sort_values(by="phi" if "phi" in group.columns else ("phi_deg" if "phi_deg" in group.columns else "deg")).dropna()
        
        # Calculate derived energies natively if not stored
        phi_arr = group["phi"] if "phi" in group.columns else np.radians(group["phi_deg" if "phi_deg" in group.columns else "deg"])
        phi_deg = np.degrees(phi_arr)
        
        # In vesicle, E_bo dimensional matching:
        # According to shape_equations.tex, the strictly dimensionless bound energy (scaled by kappa) is:
        # \bar{E}_{bo} = (-2 \pi w + 4 \pi) (1 - \cos \phi)
        # where w is intrinsically the dimensionless adhesive energy relative to the particle size
        # We need an assumption for R_ve to relate the unscaled values from the vesicle sweep:
        R_ve = 30.0 # Common dimensional reference typically used for these models
        w_vesicle = W * (rpa * R_ve)**2 / kappa 
        E_bo_dimless = (-2 * np.pi * w_vesicle + 4 * np.pi) * (1 - np.cos(phi_arr))
        
        E_tot_dimless = group["Eun"] + E_bo_dimless
        # Normalize relative to start
        E_tot_dimless = E_tot_dimless - E_tot_dimless.iloc[0]
        ax.plot(phi_deg, E_tot_dimless, marker='o', linestyle='dashed', alpha=0.7,
                label=f'Finite Vesicle (rpa={rpa:.2f})')
        
    # Planar Plots
    for R_pa_unscaled, group in df_planar.groupby("rpa"): # Note: The planar CSV saves the UNSCALED dimensional R_pa for comparison compatibility
        group = group.sort_values(by="phi_deg" if "phi_deg" in group.columns else "deg").dropna()
        rpa_pl = R_pa_unscaled / lam
        
        # Planar outputs are completely dimensional, so divide by kappa to non-dimensionalize them natively
        E_tot_dimless = (group["F_me_un"] + group["F_me_bo"] + group["F_ad"]) / kappa
        
        # Normalize relative to start
        E_tot_dimless = E_tot_dimless - E_tot_dimless.iloc[0]
        
        phi_col = "phi_deg" if "phi_deg" in group.columns else "deg"
        ax.plot(group[phi_col], E_tot_dimless, marker='s', linewidth=2,
                label=f'Infinite Planar ($r_{{pa}}^{{pl}}$={rpa_pl:.2f})')

    ax.set_title("Vesicle vs Planar Membrane Engulfment Energy Landscape (Fixed W)")
    ax.set_xlabel("Wrap Angle $\\phi$ (Degrees)")
    ax.set_ylabel("Dimensionless Total Free Energy $\\bar{F}_{tot} = F_{tot} / \\kappa$")
    ax.grid(True, alpha=0.3)
    
    # Legend formatting
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("results/comparison_energy_scaling.png", dpi=300)
    print("Saved comparison plot to results/comparison_energy_scaling.png")
    plt.show()
    plt.close()

def plot_energy_components():
    """
    Creates a 4-panel figure comparing individual energy components:
    E_un, E_bo, E_ad, and E_tot.
    """
    if df_vesicle.empty or df_planar.empty:
        return
        
    kappa = 1.0
    sigma = 0.1
    W = 0.05
    lam = np.sqrt(kappa / sigma)
    R_ve = 10.0

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_un, ax_bo, ax_ad, ax_tot = axes.flatten()

    # --- Vesicle ---
    for rpa, group in df_vesicle.groupby("rpa"):
        group = group.sort_values(by="phi" if "phi" in group.columns else ("phi_deg" if "phi_deg" in group.columns else "deg")).dropna()
        
        phi_arr = group["phi"] if "phi" in group.columns else np.radians(group["phi_deg" if "phi_deg" in group.columns else "deg"])
        phi_deg = np.degrees(phi_arr)
        
        w_vesicle = W * (rpa * R_ve)**2 / kappa 
        print(f"Vesicle A-dimensional adhesive energy density: {w_vesicle}")

        
        # Vesicle components (Dimensionless)
        E_un_dimless = group["Eun"] - group["Eun"].iloc[0]  # Normalized
        
        # Need to split E_bo and E_ad for vesicle based on the formula: (-2 * pi * w + 4 * pi) * (1 - cos)
        # E_ad is the part with w, E_bo_pure is the part with 4*pi
        E_ad_dimless = -2 * np.pi * w_vesicle * (1 - np.cos(phi_arr))
        E_bo_pure_dimless = 4 * np.pi * (1 - np.cos(phi_arr))
        
        # Total
        E_tot_dimless = group["Eun"] + E_ad_dimless + E_bo_pure_dimless
        E_tot_dimless = E_tot_dimless - E_tot_dimless.iloc[0] # Normalized
        
        # Plotting Vesicle
        kwargs = {'marker': 'o', 'linestyle': 'dashed', 'alpha': 0.7, 'label': f'Vesicle (rpa={rpa:.2f})'}
        ax_un.plot(phi_deg, E_un_dimless, **kwargs)
        ax_bo.plot(phi_deg, E_bo_pure_dimless, **kwargs)
        ax_ad.plot(phi_deg, E_ad_dimless, **kwargs)
        ax_tot.plot(phi_deg, E_tot_dimless, **kwargs)

    # --- Planar ---
    for R_pa_unscaled, group in df_planar.groupby("rpa"):

        w_planar = W * R_pa_unscaled**2 / kappa 
        print(f"Planar A-dimensional adhesive energy density: {w_planar}")
        group = group.sort_values(by="phi_deg" if "phi_deg" in group.columns else "deg").dropna()
        rpa_pl = R_pa_unscaled / lam
        phi_col = "phi_deg" if "phi_deg" in group.columns else "deg"
        phi_deg = group[phi_col]
        
        # Planar components (Divided by kappa to be dimensionless, some normalized)
        E_un_dimless = group["F_me_un"] / kappa
        E_un_dimless = E_un_dimless - E_un_dimless.iloc[0] # Normalized
        
        E_bo_pure_dimless = group["F_me_bo"] / kappa # Already essentially starts at 0 for phi=0
        E_ad_dimless = w_planar * group["F_ad"] / kappa       # Already starts at 0 for phi=0
        
        E_tot_dimless = (group["F_me_un"] + group["F_me_bo"] + group["F_ad"]) / kappa
        E_tot_dimless = E_tot_dimless - E_tot_dimless.iloc[0] # Normalized
        
        # Plotting Planar
        kwargs = {'marker': 's', 'linewidth': 2, 'label': f'Planar ($r_{{pa}}^{{pl}}$={rpa_pl:.2f})'}
        ax_un.plot(phi_deg, E_un_dimless, **kwargs)
        ax_bo.plot(phi_deg, E_bo_pure_dimless, **kwargs)
        ax_ad.plot(phi_deg, E_ad_dimless, **kwargs)
        ax_tot.plot(phi_deg, E_tot_dimless, **kwargs)

    # --- Formatting ---
    titles = [("Unbound Energy $E_{un}$ (Normalized)", ax_un),
              ("Bound Bending Energy $E_{bo}$", ax_bo),
              ("Adhesion Energy $E_{ad}$", ax_ad),
              ("Total Energy $E_{tot}$ (Normalized)", ax_tot)]
              
    for title, ax in titles:
        ax.set_title(title)
        ax.set_xlabel("Wrap Angle $\\phi$ (Degrees)")
        ax.set_ylabel("Dimensionless Energy ($E/\\kappa$)")
        ax.grid(True, alpha=0.3)
        if ax == ax_ad: # Put legend on bottom left for adhesion since it goes negative
            ax.legend()
        else:
             ax.legend()

    plt.suptitle("Energy Component Comparison (Vesicle vs Planar)", fontsize=16)
    plt.tight_layout()
    plt.savefig("results/comparison_energy_components.png", dpi=300)
    print("Saved components plot to results/comparison_energy_components.png")
    plt.show()
    plt.close()

if __name__ == "__main__":
    plot_vesicle_vs_planar()
    plot_energy_components()
