import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COST_THRESHOLD = 1.0

dfs = [pd.read_csv(f) for f in glob.glob("results/sigma_*/geometry.csv")]
df = pd.concat(dfs, ignore_index=True)
df = df[df['cost'] < COST_THRESHOLD].copy()

rpas   = sorted(df['rpa'].unique())
sigmas = sorted(df['sigma'].unique())

fig, axs = plt.subplots(1, len(rpas), figsize=(5 * len(rpas), 5))
fig.suptitle("Particle centre penetration vs wrapping angle")

for ax, rpa in zip(axs, rpas):
    sub = df[np.isclose(df['rpa'], rpa)]
    for sigma in sigmas:
        g = sub[np.isclose(sub['sigma'], sigma)].sort_values('phi_deg')
        if g.empty:
            continue
        ax.plot(g['phi_deg'], g['z_center'], marker='o', label=f'σ={sigma:.3g}')
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_title(f'rpa = {rpa}')
    ax.set_xlabel('φ (deg)')
    ax.set_ylabel('$z_{\\mathrm{center}}$ (dimensional)')
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.savefig('plot_z_penetration.png', dpi=150, transparent=True)
print("Saved plot_z_penetration.png")
plt.close()
