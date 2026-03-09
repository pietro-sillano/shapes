"""
Monge-gauge axisymmetric solver (linearized Helfrich)
Solves: kappa * Delta^2 h - sigma * Delta h + p = 0
in axisymmetric coordinates on r in [0, R].
Uses scipy.integrate.solve_bvp.

Variables:
 y0 = h
 y1 = h'
 y2 = u = Delta h = h'' + (1/r) h'
 y3 = u' = d/dr (Delta h)

System:
 y0' = y1
 y1' = h'' = u - h'/r = y2 - y1/r
 y2' = y3
 y3' = u'' = -u'/r + alpha * u - beta = -y3/r + alpha*y2 - beta
"""

import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

def monge_ode(r, y, kappa=1.0, sigma=0.1, p=0.0):
    # r: 1D array shape (N,)
    # y: array shape (4, N)
    # returns dy/dr
    alpha = sigma / kappa
    beta = p / kappa

    y0 = y[0]
    y1 = y[1]
    y2 = y[2]
    y3 = y[3]

    # prevent division by zero at r=0 by using a small r (implementation detail)
    # but solve_bvp will pass r[0]==0; algebraic handling below
    dy = np.zeros_like(y)
    dy[0] = y1
    # y1' = y2 - y1/r
    # at r=0 use limit: h''(0) = u(0) (since h'/r -> 0 as r->0 if h'~O(r))
    small = 1e-12
    invr = 1.0 / np.where(r == 0.0, small, r)
    dy[1] = y2 - y1 * invr

    dy[2] = y3
    # y3' = -y3/r + alpha*y2 - beta
    dy[3] = -y3 * invr + alpha * y2 - beta

    return dy

def bc(ya, yb):
    # ya = solution at r=0, yb = solution at r=R
    # Regularity at r=0: h'(0)=0 and u'(0)=0  -> ya[1] = 0, ya[3]=0
    # Outer boundary: h(R)=0 and h'(R)=0 -> yb[0]=0, yb[1]=0
    # h' is the slope of the height
    # h,h',u,u'

    return np.array([ya[1],   # h'(0)=0 # regularity
                     ya[3],   # u'(0)=0 # regularity
                     yb[0] - 1,   # h(R)=0
                     yb[1]])  # h'(R)=0

def initial_guess(r, R, amp=0.1):
    # Simple initial guess: small Gaussian bump
    # h_guess = amp * np.exp(- (r/(0.2*R+1e-12))**2)
    h_guess = r**2

    dh = np.gradient(h_guess, r)
    # compute u_guess approximately
    lap = np.gradient(np.gradient(h_guess, r), r) + (1.0/np.where(r==0,1e-12,r)) * dh
    du = np.gradient(lap, r)
    return np.vstack([h_guess, dh, lap, du])

def solve_monge(R=10.0, kappa=1.0, sigma=0.1, p=0.0, nr=200):
    r = np.linspace(0.0, R, nr)
    y_init = initial_guess(r, R, amp=0.1)

    sol = solve_bvp(lambda rr, yy: monge_ode(rr, yy, kappa=kappa, sigma=sigma, p=p),
                    bc, r, y_init, max_nodes=100000,verbose=2)

    if not sol.success:
        raise RuntimeError("BVP solver failed: " + sol.message)

    return sol

if __name__ == "__main__":
    # Example parameters
    R = 10.0
    kappa = 1.0
    sigma = 0.05
    p = 0.0  # pressure loading; set nonzero to see bulging

    sol = solve_monge(R=R, kappa=kappa, sigma=sigma, p=p, nr=400)
    r_plot = np.linspace(0, R, 400)
    h = sol.sol(r_plot)[0]
    u = sol.sol(r_plot)[2]

    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.plot(r_plot, h)
    # plt.ylim(0,0.15)
    plt.xlabel('r')
    plt.ylabel('h(r)')
    plt.title('Height profile')

    plt.subplot(1,2,2)
    plt.plot(r_plot, u)
    plt.xlabel('r')
    plt.ylabel('Delta h (u)')
    plt.title('Laplacian of h')
    plt.tight_layout()
    plt.show()
