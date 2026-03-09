import numpy as np
from numba import njit
from scipy.interpolate import InterpolatedUnivariateSpline


@njit
def deg_to_rad(deg):
    return np.pi*deg/180


@njit
def DPsi_DS(omega, u):
    return omega*u


@njit
def DU_DS(omega, psi, u, x, k, gamma):
    a1 = - u/x*np.cos(psi)
    a2 = (np.sin(psi)*np.cos(psi))/x**2
    a3 = (np.sin(psi)*gamma)/(2*np.pi*x*k)
    return omega*(a1+a2+a3)


@njit
def DGamma_DS(omega, u, psi, x, k, sigma):
    a1 = np.pi*k*(u**2-np.sin(psi)**2/x**2)
    a2 = 2*np.pi*sigma*k
    return omega*(a1+a2)


@njit
def DX_DS(omega, psi):
    return omega*np.cos(psi)


@njit
def South_X(s, omega, u0):
    x1 = omega
    x3 = -omega**3*u0**2
    return x1*s+1/6*x3*s**3


@njit
def South_Psi(s, omega, u0, sigma):
    psi1 = omega*u0
    psi3 = omega**3*(3*sigma*u0-2*u0**3)
    return psi1*s+1/6*psi3*s**3


@njit
def South_U(s, omega, u0, sigma):
    return u0+1/2*omega**2*0.5*(3*u0*sigma-2*u0**3)*s**2


@njit
def South_Gamma(s, omega, u0, sigma, k):
    gamma1 = omega*(2 * np.pi * sigma * k)
    gamma3 = 4/3 * np.pi * k * u0 * omega**3*(3 * sigma * u0 - 2*u0**3)
    return gamma1*s+1/6*gamma3*s**3


@njit
def InitialArcLength(omega, u0):
    threshold = 0.035
    n = 1
    delta_s = 0.0001
    while True:
        if South_X(n*delta_s, omega, u0) > threshold:
            break
        elif n > 10000:
            break
        else:
            n += 1
    return n*delta_s


@njit
def InitialValues(s_init, omega, u0, sigma, k):
    return np.array([South_Psi(s_init, omega, u0, sigma),
                     South_U(s_init, omega, u0, sigma),
                     South_Gamma(s_init, omega, u0, sigma, k),
                     South_X(s_init, omega, u0),
                     ], dtype=np.float64)


@njit
def ShapeIntegrator(s, z, omega, sigma):
    psi, u, gamma, x = z[0], z[1], z[2], z[3]
    k = 1.0
    return np.array([DPsi_DS(omega, u),
                     DU_DS(omega, psi, u, x, k, gamma),
                     DGamma_DS(omega, u, psi, x, k, sigma),
                     DX_DS(omega, psi)
                     ], dtype=np.float64)


@njit
def ShapeJacobian(s, z, omega, sigma):
    psi, u, gamma, x = z

    k = 1.0
    a12 = omega
    a11, a13, a14, a15, a16 = 0.0, 0.0, 0.0, 0.0, 0.0

    a21 = omega * (u / x * np.sin(psi) + 1 / x**2 * np.cos(psi)**2 - np.sin(psi)
                   ** 2 / x**2 * gamma / (2 * np.pi * k * x) * np.cos(psi))
    a22 = -omega/x*np.cos(psi)  # dudu
    a23 = omega*np.sin(psi)/(2*k*np.pi*x)  # dudgamma
    a24 = omega * (u / x**2 * np.cos(psi) - 2 * np.cos(psi) * np.sin(psi) /
                   x**3 - gamma * np.sin(psi) / (2 * np.pi * k * x**2))  # dudx

    a31 = omega*(-np.pi*k*np.cos(psi)/x**2)  # dgammadpsi
    a32 = omega*np.pi*k*2*u  # dgammadu
    a34 = 2*omega*np.pi*k*np.sin(psi)/x**3  # dgammadx
    a33 = 0.0

    a41 = -omega*np.sin(psi)  # dxdpsi
    a42, a43, a44 = 0.0, 0.0, 0.0

    return np.array([[a11, a12, a13, a14],
                     [a21, a22, a23, a24],
                     [a31, a32, a33, a34],
                     [a41, a42, a43, a44],
                     ], dtype=np.float64)


def handling_instabilities(t, y,  omega, sigma):
    if np.abs(y[0]) > 10 or np.abs(y[1]) > 10 or np.abs(y[3]) > 10:
        return 1
    return -1

handling_instabilities.terminal = True


def hamiltonian(sol, sigma):
    s = sol.t
    psi = sol.y[0, :]
    u = sol.y[1, :]
    gamma = sol.y[2, :]
    x = sol.y[3, :]

    H = 0.5*u**2 * x + gamma * \
        np.cos(psi)/(2*np.pi)-x*sigma- 0.5 * np.sin(psi) ** 2 / x
    return H


def calc_energies(sol, best_parameters, phi_deg, W, rpa):
    s = sol.t
    psi = sol.y[0, :]
    u = sol.y[1, :]
    gamma = sol.y[2, :]
    x = sol.y[3, :]

    omega, sigma, u0 = best_parameters
    phi = deg_to_rad(phi_deg)

    Eun = 2*np.pi*np.trapezoid(x/2*(u+np.sin(psi)/x)**2 +
                           sigma*x+gamma*(x-np.cos(psi)), s)

    Ebo = (-2*np.pi*W*rpa**2 + 4*rpa**2)*(1-np.cos(phi))
    Etot = Eun + Ebo

    return Ebo, Eun, Etot
