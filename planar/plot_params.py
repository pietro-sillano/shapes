import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 

file_path = "results/params.csv"
df = pd.read_csv(file_path,dtype=np.float32)

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Parameters by $\\degree$ and RPA")

variables = [
    ("u0", axs[0, 0]),
    ("omega", axs[0, 1]),
    ("cost", axs[1, 0]),
    ("psi1", axs[1, 1]),

]

# Plot deg vs each parameter with different color for different rpa
for rpa, group in df.groupby("rpa"):
    group = group.sort_values(by="deg")
    for var, ax in variables:
        if var =='cost' or var =='psi1':
            ax.semilogy(group["deg"], group[var], marker='o', label=f'RPA = {rpa:.2f}')
        else:
            ax.plot(group["deg"], group[var], marker='o', label=f'RPA = {rpa:.2f}')


for var, ax in variables:
    ax.set_xlabel("deg")
    ax.set_ylabel(var)
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.savefig("plot_params.png",dpi=300,transparent=True)
plt.show()
plt.close()
