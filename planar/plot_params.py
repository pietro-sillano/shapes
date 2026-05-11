import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

file_path = "results/params.csv"
df = pd.read_csv(file_path, dtype=np.float64)

variables = [
    ("u0",    "u0"),
    ("omega", "omega"),
    ("cost",  "cost"),
    ("psi1",  "psi1"),
]

for sigma_val, sigma_group in df.groupby("sigma"):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Parameters by $\\phi$ and RPA  —  $\\Sigma = {sigma_val:.4g}$")

    for rpa, group in sigma_group.groupby("rpa"):
        group = group.sort_values(by="deg")
        for (var, label), ax in zip(variables, axs.flat):
            if var in ("cost", "psi1"):
                ax.semilogy(group["deg"], group[var], marker='o', label=f'RPA = {rpa:.2f}')
            else:
                ax.plot(group["deg"], group[var], marker='o', label=f'RPA = {rpa:.2f}')

    for (var, label), ax in zip(variables, axs.flat):
        ax.set_xlabel("deg")
        ax.set_ylabel(label)
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    out = f"plot_params_sigma_{sigma_val:.4g}.png"
    plt.savefig(out, dpi=300, transparent=True)
    print(f"Saved {out}")
    plt.close()
