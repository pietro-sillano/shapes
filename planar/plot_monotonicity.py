import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

KAPPA = 1.0

df = pd.read_csv("results/monotonicity_labels.csv")
df['w_bar']     = 2 * df['w'] * df['rpa']**2 / KAPPA
df['sigma_bar'] = df['sigma'] * df['rpa']**2 / KAPPA

marker_map = {1: 'o', -1: 's', 0: '^'}
color_map  = {1: '#2ca02c', -1: '#d62728', 0: '#1f77b4'}
label_map  = {1: 'free', -1: 'full', 0: 'partially'}

# ── Plot 1: combined, rescaled axes ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
for lv in [1, -1, 0]:
    sub = df[df['label'] == lv]
    if sub.empty:
        continue
    ax.scatter(sub['w_bar'], sub['sigma_bar'],
               marker=marker_map[lv], color=color_map[lv],
               label=label_map[lv], s=40, alpha=0.6,
               edgecolors='white', linewidths=0.5)
ax.set_xlabel(r'$\bar{w} = W\,R_{pa}^2/\kappa$')
ax.set_ylabel(r'$\bar{\sigma} = \sigma\,R_{pa}^2/\kappa$')
ax.set_title('Monotonicity phase diagram (all rpa)')
ax.legend(title='Labels')
ax.grid(True, ls='--', alpha=0.4)
plt.xlim(0,12)
plt.tight_layout()
plt.savefig('test_plot/monot_rescaled.png', dpi=150, transparent=True)
plt.show()
plt.close()

# ── Plot 2: rpa/lam vs w/sigma ───────────────────────────────────────────────
# lam = sqrt(kappa/sigma)  →  rpa/lam = rpa * sqrt(sigma/kappa)
fig, ax = plt.subplots(figsize=(8, 6))
df['rpa_over_lam'] = df['rpa'] * np.sqrt(df['sigma'] / KAPPA)
df['w_over_sigma'] = df['w'] / df['sigma']
for lv in [1, -1, 0]:
    sub = df[df['label'] == lv]
    if sub.empty:
        continue
    ax.scatter(sub['rpa_over_lam'], sub['w_over_sigma'],
               marker=marker_map[lv], color=color_map[lv],
               label=label_map[lv], s=40, alpha=0.6,
               edgecolors='white', linewidths=0.5)
ax.set_xlabel(r'$R_{pa}/\lambda$  $(\lambda = \sqrt{\kappa/\sigma})$')
ax.set_ylabel(r'$W/\sigma$')
ax.set_xscale('log')
ax.set_title('Monotonicity phase diagram (all rpa)')
ax.legend(title='Labels')
ax.grid(True, ls='--', alpha=0.4)
plt.tight_layout()
plt.savefig('test_plot/monot_rpa_lam.png', dpi=150, transparent=True)
plt.show()
plt.close()

# # ── Plot 3: subplots per rpa, rescaled axes ───────────────────────────────────
# rpa_values = sorted(df['rpa'].unique())
# cols = 3
# rows = (len(rpa_values) + cols - 1) // cols

# fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4),
#                          sharex=True, sharey=True)
# axes = axes.flatten() if len(rpa_values) > 1 else [axes]

# for i, rpa in enumerate(rpa_values):
#     ax = axes[i]
#     sub_rpa = df[np.isclose(df['rpa'], rpa)]
#     for lv in [1, -1, 0]:
#         sub = sub_rpa[sub_rpa['label'] == lv]
#         if sub.empty:
#             continue
#         ax.scatter(sub['w_bar'], sub['sigma_bar'],
#                    marker=marker_map[lv], color=color_map[lv],
#                    label=label_map[lv], s=70, alpha=0.6,
#                    edgecolors='white', linewidths=0.5)
#     ax.set_title(f'Rpa = {rpa}')
#     ax.grid(True, ls='--', alpha=0.4)
#     if i >= (rows - 1) * cols:
#         ax.set_xlabel(r'$\bar{w} = W\,R_{pa}^2/\kappa$')
#     if i % cols == 0:
#         ax.set_ylabel(r'$\bar{\sigma} = \sigma\,R_{pa}^2/\kappa$')

# for j in range(i + 1, len(axes)):
#     fig.delaxes(axes[j])

# handles, labels = axes[0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper right', title='Labels', bbox_to_anchor=(1.0, 0.9))
# fig.suptitle('Monotonicity phase diagram by Rpa', fontsize=14)
# plt.tight_layout(rect=[0, 0.03, 0.9, 0.95])
# plt.savefig('test_plot/monot_subplots.png', dpi=150, transparent=True)
# plt.show()
# plt.close()
