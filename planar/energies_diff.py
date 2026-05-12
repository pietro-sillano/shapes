import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

COST_THRESHOLD = 1.0
POLY_DEG = 4
OUTLIER_SIGMA = 2.5

dfs = [pd.read_csv(f) for f in glob.glob("results/sigma_*/energies.csv")]
df = pd.concat(dfs, ignore_index=True)
df = df[(df['cost'] < COST_THRESHOLD) & (df['rpa'] > 1.0)].copy()

W = np.linspace(0.1, 2.0, 5)

rpas   = sorted(df['rpa'].unique())
sigmas = sorted(df['sigma'].unique())


def fit_poly_robust(x, y, deg=POLY_DEG, n_sigma=OUTLIER_SIGMA):
    """Iterative sigma-clipping polynomial fit. Returns (coeffs, inlier_mask)."""
    mask = np.ones(len(x), dtype=bool)
    for _ in range(5):
        if mask.sum() <= deg + 1:
            break
        coeffs = np.polyfit(x[mask], y[mask], deg)
        residuals = y - np.polyval(coeffs, x)
        threshold = n_sigma * residuals[mask].std()
        new_mask = np.abs(residuals) < threshold
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask
    return coeffs, mask


def monotonicity(coeffs, x_min, x_max, n_pts=300):
    """Return +1 (increasing), -1 (decreasing), or 0 (non-monotonic)."""
    deriv = np.polyder(coeffs)
    vals = np.polyval(deriv, np.linspace(x_min, x_max, n_pts))
    if np.all(vals >= 0):
        return +1
    if np.all(vals <= 0):
        return -1
    return 0


MONO_SYMBOL = {+1: '↑', -1: '↓', 0: '~'}
KAPPA = 1.0

all_phase_points = []  # (w, sigma_rescaled, mono)

for rpa in rpas:
    fig, axs = plt.subplots(1, len(W), figsize=(5 * len(W), 5))
    fig.suptitle(f"Total energy vs wrapping angle, Rpa: {rpa}")
    sub = df[np.isclose(df['rpa'], rpa)].copy()

    mono_results = {}  # (w, sigma) → +1 / -1 / 0

    for ax, w in zip(axs, W):
        w_bar = w * rpa**2 / KAPPA
        sub['F_total'] = sub['F_me_un'] + sub['F_me_bo'] + w * sub['F_ad']

        for sigma in sigmas:
            g = sub[np.isclose(sub['sigma'], sigma)].sort_values('phi_deg')
            if g.empty:
                continue

            sigma_bar = sigma * rpa**2 / KAPPA

            x = g['phi_deg'].to_numpy(dtype=float)
            y = g['F_total'].to_numpy(dtype=float)

            line, = ax.plot(x, y, marker='o', label=None, alpha=0.4, ms=4)
            colour = line.get_color()

            if len(x) > POLY_DEG + 1:
                coeffs, inliers = fit_poly_robust(x, y)
                mono = monotonicity(coeffs, x.min(), x.max())

                # highlight outliers
                if (~inliers).any():
                    ax.scatter(x[~inliers], y[~inliers],
                               marker='x', color=colour, s=60, zorder=5, alpha=0.6)

                x_smooth = np.linspace(x.min(), x.max(), 300)
                ax.plot(x_smooth, np.polyval(coeffs, x_smooth),
                        color=colour, lw=2, ls='-')
            else:
                mono = 0

            mono_results[(w, sigma)] = mono
            ax.plot([], [], color=colour,
                    label=f'σ̄={sigma_bar:.3g} {MONO_SYMBOL[mono]}')

        ax.axhline(0, color='k', lw=0.8, ls='--')
        ax.set_title(f'w̄ = {w_bar:.3g}')
        ax.set_xlabel('φ (deg)')
        ax.grid(True)
        ax.legend(fontsize=7)

    for ax in axs:
        ax.set_ylabel('$F_{\\mathrm{total}}$ / $\\kappa$')
    plt.tight_layout()
    plt.savefig(f'test_plot/diff_w_rpa_{rpa}.png', dpi=150, transparent=True)
    plt.close()

    # accumulate points for the combined phase plot (both axes rescaled by rpa²/κ)
    for sigma in sigmas:
        sr = sigma * rpa**2 / KAPPA
        for w in W:
            mono = mono_results.get((w, sigma), None)
            if mono is not None:
                wr = w * rpa**2 / KAPPA
                all_phase_points.append((wr, sr, mono))

# ── Combined phase scatter plot ───────────────────────────────────────────────
if all_phase_points:
    pts = np.array(all_phase_points)  # shape (N, 3): w, sigma_rescaled, mono
    wx, yr, mc = pts[:, 0], pts[:, 1], pts[:, 2]

    cmap = ListedColormap(['#4477AA', '#BBBBBB', '#CC4444'])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(wx, yr, c=mc, cmap=cmap, norm=norm, s=120,
                    edgecolors='k', linewidths=0.4, zorder=3)

    cbar = fig.colorbar(sc, ax=ax, ticks=[-1, 0, 1])
    cbar.ax.set_yticklabels(['↓ decr.', '~ non-mono', '↑ incr.'])

    ax.set_xlabel(r'$\bar{w} = W\,R_{pa}^2/\kappa$')
    ax.set_ylabel(r'$\bar{\sigma} = \sigma\,R_{pa}^2/\kappa$')
    ax.set_title('Monotonicity phase diagram (all rpa)')
    ax.grid(True, ls='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig('test_plot/phase_combined.png', dpi=150, transparent=True)
    plt.close()
