import numpy as np
import matplotlib.pyplot as plt

path = '/home/pietro/Desktop/numerical_shape/code/cluster/seq_omega/'

path = '/home/pietro/Desktop/numerical_shape/code/cluster/seq_omega_2/'

params = np.load(path+'params.npy')
init_params = np.load(path+'init_params.npy')
cost = np.load(path+'cost.npy')


plt.plot(init_params[:, 0], cost,  'o', label="omega")
plt.plot(init_params[:, 1], cost,  'o', label="sigma")
plt.plot(init_params[:, 2], cost,  'o', label="u0")
plt.legend()
plt.xlabel("init params")
plt.ylabel("cost")

plt.show()


for x, std in zip(np.mean(params, axis=0), np.std(params, axis=0)):
    print(f"{x:.2f} +- {std:.2f}")


plt.loglog(params[:, 0], cost,  'o', label="omega")
# plt.plot(params[:, 1], cost,  'o', label="sigma")
# plt.plot(params[:, 2], cost,  'o', label="u0")
plt.xlabel("params")
plt.ylabel("cost")
plt.legend()
plt.show()


plt.plot(init_params[:, 0],  params[:, 0],  'o')
plt.plot(init_params[:, 0],  init_params[:, 0],  '--k')
plt.xlabel("initial omega")
plt.ylabel("final omega")
plt.show()


plt.plot(init_params[:, 1],  params[:, 1],  'o')
plt.plot(init_params[:, 1],  init_params[:, 1],  '--k')
plt.xlabel("initial sigma")
plt.ylabel("final sigma")
plt.show()


plt.plot(init_params[:, 2],  params[:, 2],  'o')
plt.plot(init_params[:, 2],  init_params[:, 2],  '--k')
plt.xlabel("initial u0")
plt.ylabel("final u0")
plt.show()
