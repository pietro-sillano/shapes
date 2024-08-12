import numpy as np
import matplotlib.pyplot as plt

path = '/home/pietro/Desktop/numerical_shape/code/cluster/seq_omega/'

path = '/home/pietro/Desktop/numerical_shape/code/cluster/seq_omega_2/'

params = np.load(path+'params.npy')
init_params = np.load(path+'init_params.npy')
cost = np.load(path+'cost.npy')

# idxs = np.nonzero(params)
# params = params[idxs]

# idxs = np.nonzero(init_params)
# init_params = init_params[idxs]


# idxs = np.nonzero(cost)
# cost = cost[idxs]
plt.cla()
plt.semilogy(init_params[:, 0], cost,  'o', label="omega")
plt.semilogy(init_params[:, 1], cost,  'o', label="sigma")
plt.semilogy(init_params[:, 2], cost,  'o', label="u0")
plt.legend()
plt.xlabel("init params")
plt.ylabel("cost")
plt.savefig(path+f"init_params_cost.png", dpi=200)
plt.show()
plt.close()


for x, std in zip(np.mean(params, axis=0), np.std(params, axis=0)):
    print(f"{x:.2f} +- {std:.2f}")

plt.cla()
plt.loglog(params[:, 0], cost,  'o', label="omega")
plt.plot(params[:, 1], cost,  'o', label="sigma")
plt.plot(params[:, 2], cost,  'o', label="u0")
plt.xlabel("params")
plt.ylabel("cost")
plt.legend()
plt.savefig(path+f"params_cost.png", dpi=200)
plt.show()
plt.close()

plt.cla()
plt.plot(init_params[:, 0],  params[:, 0],  '-o')
# plt.plot(init_params[:, 1],  params[:, 1],  '-o')
# plt.plot(init_params[:, 2],  params[:, 2],  '-o')
# plt.plot(init_params[:, 0],  init_params[:, 0],  '--k')
plt.xlabel("initial omega")
plt.ylabel("final omega")
plt.savefig(path+f"omega.png", dpi=200)
plt.show()
plt.close()

plt.cla()
plt.plot(init_params[:, 1],  params[:, 1],  'o')
plt.plot(init_params[:, 1],  init_params[:, 1],  '--k')
plt.xlabel("initial sigma")
plt.ylabel("final sigma")
plt.savefig(path+f"sigma.png", dpi=200)
plt.show()
plt.close()

plt.cla()
plt.plot(init_params[:, 2],  params[:, 2],  'o')
plt.plot(init_params[:, 2],  init_params[:, 2],  '--k')
plt.xlabel("initial u0")
plt.ylabel("final u0")
plt.savefig(path+f"u0.png", dpi=200)
plt.close()
plt.show()


plt.cla()
plt.hist(params[:, 0], bins=20)
plt.savefig(path+f"hist_omega.png", dpi=200)
plt.close()
plt.show()
