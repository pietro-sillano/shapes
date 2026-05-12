import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COST_THRESHOLD = 1.0
KAPPA = 1.0
LABEL_FILE = 'results/monotonicity_labels.csv'

# ── Rectangle bounds in (w_bar, sigma_bar) phase-plot coordinates ─────────────
# w_bar     = 2 * W * rpa² / kappa   (matches plot_monotonicity.py)
# sigma_bar = sigma * rpa² / kappa
W_BAR_MIN,     W_BAR_MAX     = 3.85,4.25
SIGMA_BAR_MIN, SIGMA_BAR_MAX = 0.0, 1.0
N_W_SAMPLES = 5   # how many W values to sample per (rpa, sigma) inside the box

# ── Load data ─────────────────────────────────────────────────────────────────
dfs = [pd.read_csv(f) for f in glob.glob("results/sigma_*/energies.csv")]
df = pd.concat(dfs, ignore_index=True)
df = df[(df['cost'] < COST_THRESHOLD) & (df['rpa'] > 1.0)].copy()

rpas   = sorted(df['rpa'].unique())
sigmas = sorted(df['sigma'].unique())

# ── Load existing labels ──────────────────────────────────────────────────────
if os.path.exists(LABEL_FILE):
    labels_df = pd.read_csv(LABEL_FILE)
    labeled = set(zip(labels_df['rpa'], labels_df['w'], labels_df['sigma']))
else:
    labels_df = pd.DataFrame(columns=['rpa', 'w', 'sigma', 'label'])
    labeled = set()

# ── Build list of unlabeled triples inside the rectangle ─────────────────────
todo = []
for rpa in rpas:
    sub = df[np.isclose(df['rpa'], rpa)].copy()
    for sigma in sigmas:
        g = sub[np.isclose(sub['sigma'], sigma)]
        if g.empty:
            continue

        sigma_bar = sigma * rpa**2 / KAPPA
        if not (SIGMA_BAR_MIN <= sigma_bar <= SIGMA_BAR_MAX):
            continue

        # invert w_bar = 2*w*rpa²/kappa to get the w range
        w_min = W_BAR_MIN * KAPPA / (2 * rpa**2)
        w_max = W_BAR_MAX * KAPPA / (2 * rpa**2)
        W_local = np.linspace(w_min, w_max, N_W_SAMPLES)

        for w in W_local:
            key = (rpa, round(w, 10), sigma)
            if key not in labeled:
                todo.append((rpa, w, sigma))

print(f"Rectangle  w_bar∈[{W_BAR_MIN},{W_BAR_MAX}]  σ_bar∈[{SIGMA_BAR_MIN},{SIGMA_BAR_MAX}]")

total_todo = len(todo)
if total_todo == 0:
    print("All curves already labeled.")
    raise SystemExit

print(f"{total_todo} curves to label. Keys: p=pos. monotonic | n=neg. monotonic | x=not monotonic | q=quit\n")

LABEL_MAP = {'p': +1, 'n': -1, 'x': 0, 'z': np.nan}


def _show_curve(rpa, w, sigma, idx, total_todo):
    sub = df[np.isclose(df['rpa'], rpa)].copy()
    norm = KAPPA * np.pi
    sub['F_total'] = (sub['F_me_un'] + sub['F_me_bo'] + w * sub['F_ad']) / norm

    g_target = sub[np.isclose(sub['sigma'], sigma)].sort_values('phi_deg')
    x_t = g_target['phi_deg'].to_numpy(dtype=float)
    y_t = g_target['F_total'].to_numpy(dtype=float)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 4))
    w_bar     = 2 * w * rpa**2 / KAPPA
    sigma_bar = sigma * rpa**2 / KAPPA
    fig.suptitle(
        f'[{idx+1}/{total_todo}]  rpa={rpa}  W={w:.3g}  σ={sigma:.3g}'
        f'     →  w̄={w_bar:.3g}  σ̄={sigma_bar:.3g}',
        fontsize=11, fontweight='bold'
    )

    ax_left.plot(x_t, y_t, color='crimson', lw=2, ms=5)
    ax_left.axhline(0, color='k', lw=0.8, ls='--')
    ax_left.set_xlabel('φ (deg)')
    ax_left.set_ylabel(r'$\bar{E} = F_\mathrm{total} / (\kappa\pi)$')
    ax_left.set_title(f'σ={sigma:.3g}  (target)')
    ax_left.grid(True)
    ax_left.set_xlim(0, 180)

    for s in sigmas:
        g = sub[np.isclose(sub['sigma'], s)].sort_values('phi_deg')
        if g.empty:
            continue
        if np.isclose(s, sigma):
            ax_right.plot(g['phi_deg'], g['F_total'], color='crimson', lw=2.5,
                          marker='o', ms=4, label=f'σ={s:.3g} ◄', zorder=5)
        else:
            ax_right.plot(g['phi_deg'], g['F_total'], marker='o', ms=3,
                          alpha=0.35, lw=1, label=f'σ={s:.3g}')
    ax_right.axhline(0, color='k', lw=0.8, ls='--')
    ax_right.set_xlabel('φ (deg)')
    ax_right.set_ylabel(r'$\bar{E} = F_\mathrm{total} / (\kappa\pi)$')
    ax_right.set_title(f'Context  (rpa={rpa}, W={w:.3g})')
    ax_right.legend(fontsize=7)
    ax_right.grid(True)
    ax_right.set_xlim(0, 180)

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.2)


def _remove_from_csv(rpa, w, sigma):
    if not os.path.exists(LABEL_FILE):
        return
    ldf = pd.read_csv(LABEL_FILE)
    mask = ~(np.isclose(ldf['rpa'], rpa) &
             np.isclose(ldf['w'],   w)   &
             np.isclose(ldf['sigma'], sigma))
    ldf[mask].to_csv(LABEL_FILE, index=False)


# ── Main labeling loop ────────────────────────────────────────────────────────
session_history = []  # (rpa, w, sigma) saved in this session, in order
idx = 0

while idx < len(todo):
    rpa, w, sigma = todo[idx]
    _show_curve(rpa, w, sigma, idx, total_todo)

    while True:
        raw = input(f"  [{total_todo - idx} left]  p / n / x / b (back) / q: ").strip().lower()

        if raw == 'q':
            plt.close()
            print("Saved progress. Exiting.")
            raise SystemExit

        if raw == 'b':
            if not session_history:
                print("  Already at the first curve of this session.")
                continue
            plt.close()
            prev = session_history.pop()
            _remove_from_csv(*prev)
            labeled.discard((prev[0], round(prev[1], 10), prev[2]))
            idx -= 1
            break

        if raw in LABEL_MAP:
            label = LABEL_MAP[raw]
            plt.close()
            row = pd.DataFrame([{'rpa': rpa, 'w': w, 'sigma': sigma, 'label': label}])
            write_header = not os.path.exists(LABEL_FILE)
            row.to_csv(LABEL_FILE, mode='a', header=write_header, index=False)
            labeled.add((rpa, round(w, 10), sigma))
            session_history.append((rpa, w, sigma))
            idx += 1
            break

        print("  Invalid key. Use p, n, x, b, z or q.")

print("All curves labeled.")
