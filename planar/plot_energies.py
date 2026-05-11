import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import argrelmin

COST_THRESHOLD = 1.0

dfs = [pd.read_csv(f) for f in glob.glob("results/sigma_*/energies.csv")]
df = pd.concat(dfs, ignore_index=True)
df = df[df['cost'] < COST_THRESHOLD].copy()
df['F_total'] = df['F_me_un'] + df['F_me_bo'] + df['F_ad']  # at W=1

rpas   = sorted(df['rpa'].unique())
sigmas = sorted(df['sigma'].unique())

# ── Plot 1: F_total vs phi for each rpa, coloured by sigma (W=1) ─────────────
fig, axs = plt.subplots(1, len(rpas), figsize=(5 * len(rpas), 5))
fig.suptitle("Total energy vs wrapping angle  (W = 1)")

for ax, rpa in zip(axs, rpas):
    sub = df[np.isclose(df['rpa'], rpa)]
    for sigma in sigmas:
        g = sub[np.isclose(sub['sigma'], sigma)].sort_values('phi_deg')
        if g.empty:
            continue
        ax.plot(g['phi_deg'], g['F_total'], marker='o', label=f'σ={sigma:.3g}')
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_title(f'rpa = {rpa}')
    ax.set_xlabel('φ (deg)')
    ax.grid(True)
    ax.legend()


for ax in axs:
    ax.set_ylabel('$F_{\\mathrm{total}}$ / $\\kappa$')
plt.tight_layout()
plt.savefig('plot_energies_vs_phi.png', dpi=150, transparent=True)
print("Saved plot_energies_vs_phi.png")
plt.close()

# ── Plot 2: min_phi(F_total) vs W, one line per sigma, per rpa ───────────────
W_arr = np.linspace(0.1, 1.0, 60)

fig, axs = plt.subplots(1, len(rpas), figsize=(5 * len(rpas), 5), sharey=True)
fig.suptitle("Minimum total energy over φ vs adhesion strength W")

for ax, rpa in zip(axs, rpas):
    sub = df[np.isclose(df['rpa'], rpa)]
    for sigma in sigmas:
        g = sub[np.isclose(sub['sigma'], sigma)]
        if g.empty:
            continue
        F_wb = (g['F_me_un'] + g['F_me_bo']).values
        F_ad = g['F_ad'].values  # computed at W=1; scales linearly with W
        min_ftot = [np.min(F_wb + F_ad * W) for W in W_arr]
        ax.plot(W_arr, min_ftot, label=f'σ={sigma:.3g}')
        ax.set_ylim(-5,5)

    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_title(f'rpa = {rpa}')
    ax.set_xlabel('W')
    ax.grid(True)
    ax.legend()

axs[0].set_ylabel('$\\min_\\phi\\, F_{\\mathrm{total}}$ / $\\kappa$')
plt.tight_layout()
plt.savefig('plot_energy_switching.png', dpi=150, transparent=True)
print("Saved plot_energy_switching.png")
plt.close()

# ── Plot 3: Phase diagram sigma vs W, coloured by spontaneous (F_total < 0) ──
fig, axs = plt.subplots(1, len(rpas), figsize=(5 * len(rpas), 4))
fig.suptitle("Phase diagram: spontaneous engulfment")

for ax, rpa in zip(axs, rpas):
    sub = df[np.isclose(df['rpa'], rpa)]
    sigma_vals, W_vals, spont = [], [], []
    for sigma in sigmas:
        g = sub[np.isclose(sub['sigma'], sigma)]
        if g.empty:
            continue
        F_wb = (g['F_me_un'] + g['F_me_bo']).values
        F_ad = g['F_ad'].values
        for W in W_arr:
            sigma_vals.append(sigma)
            W_vals.append(W)
            spont.append(float(np.min(F_wb + F_ad * W) < 0))
    sc = ax.scatter(W_vals, sigma_vals, c=spont, cmap='RdYlGn',
                    s=30, vmin=0, vmax=1, edgecolors='none')
    ax.set_title(f'rpa = {rpa}')
    ax.set_xlabel('W')
    ax.grid(True, alpha=0.3)

axs[0].set_ylabel('σ')
plt.colorbar(sc, ax=axs[-1], label='Spontaneous (1=yes)')
plt.tight_layout()
plt.savefig('plot_phase_diagram.png', dpi=150, transparent=True)
print("Saved plot_phase_diagram.png")
plt.close()

# ── Plot 4: F_me_un vs phi, one subplot per rpa, coloured by sigma ────────────
fig, axs = plt.subplots(1, len(rpas), figsize=(5 * len(rpas), 5))
fig.suptitle("Unbounded membrane energy vs wrapping angle")

for ax, rpa in zip(axs, rpas):
    sub = df[np.isclose(df['rpa'], rpa)]
    for sigma in sigmas:
        g = sub[np.isclose(sub['sigma'], sigma)].sort_values('phi_deg')
        if g.empty:
            continue
        ax.plot(g['phi_deg'], g['F_me_un'], marker='o', label=f'σ={sigma:.3g}')
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_title(f'rpa = {rpa}')
    ax.set_xlabel('φ (deg)')
    ax.set_ylabel('$F_{\\mathrm{me,un}}$ / $\\kappa$')
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.savefig('plot_F_me_un_vs_phi.png', dpi=150, transparent=True)
print("Saved plot_F_me_un_vs_phi.png")
plt.close()

# ── Plot 5: local and absolute minima of F_total vs phi ───────────────────────
fig, axs = plt.subplots(1, len(rpas), figsize=(5 * len(rpas), 5))
fig.suptitle("Local and absolute minima of total energy (W = 1)")

colors = plt.cm.tab10(np.linspace(0, 1, len(sigmas)))

for ax, rpa in zip(axs, rpas):
    sub = df[np.isclose(df['rpa'], rpa)]
    for color, sigma in zip(colors, sigmas):
        g = sub[np.isclose(sub['sigma'], sigma)].sort_values('phi_deg').reset_index(drop=True)
        if len(g) < 3:
            continue
        vals = g['F_total'].values
        phis = g['phi_deg'].values

        # local minima (order=1: compare each point to its neighbours)
        local_idx = argrelmin(vals, order=1)[0]
        # absolute minimum
        abs_idx = np.argmin(vals)

        # combine local minima and absolute minimum, sorted by phi
        all_idx = np.unique(np.concatenate([local_idx, [abs_idx]]))
        all_idx = all_idx[np.argsort(phis[all_idx])]

        ax.plot(phis[all_idx], vals[all_idx], marker='o', color=color,
                label=f'σ={sigma:.3g}')

    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_title(f'rpa = {rpa}')
    ax.set_xlabel('φ (deg)')
    ax.set_ylabel('$F_{\\mathrm{total}}$ / $\\kappa$')
    ax.grid(True)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('plot_minima_F_total.png', dpi=150, transparent=True)
print("Saved plot_minima_F_total.png")
plt.close()
