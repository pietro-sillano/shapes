############################################################
# Python code for the shape calculation of endocytosis process of a colloidal particle
# written by Pietro Sillano, in collaboration with Mijke Dijke at TU Delft, May 2024, adapting the code by Felix Frey from Membrane Area Gain and Loss during Cytokinesis, Phys. Rev. E (2022)
############################################################

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import least_squares


def deg_to_rad(deg):
    return np.pi*deg/180


def array_to_dat(time_array, data_array, filename):
    a = np.column_stack((time_array, data_array))
    np.savetxt(filename, a, delimiter=',')
    return a.shape


# integrate 4 ODEs
def DPsi_DS(omega, u):
    return omega*u


def DU_DS(omega, psi, u, x, k, gamma):
    a1 = - u/x*np.cos(psi)
    a2 = (np.sin(psi)*np.cos(psi))/x**2
    a3 = (np.sin(psi)*gamma)/(2*np.pi*x*k)
    return omega*(a1+a2+a3)


def DGamma_DS(omega, u, psi, x, k, sigma):
    a1 = np.pi*k*(u**2-np.sin(psi)**2/x**2)
    a2 = 2*np.pi*sigma*k
    return omega*(a1+a2)


def DX_DS(omega, psi):
    return omega*np.cos(psi)

# South Pole expansion


def South_X(s, omega, u0):
    x1 = omega
    x3 = -omega**3*u0**2
    return x1*s+1/6*x3*s**3


def South_Psi(s, omega, u0, sigma):
    psi1 = omega*u0
    psi3 = omega**3*(3*sigma*u0-2*u0**3)
    return psi1*s+1/6*psi3*s**3


def South_U(s, omega, u0, sigma):
    return u0+1/2*omega**2*0.5*(3*u0*sigma-2*u0**3)*s**2


def South_Gamma(s, omega, u0, sigma, k):
    gamma1 = omega*(2 * np.pi * sigma * k)
    gamma3 = 4/3 * np.pi * k * u0 * omega**3*(3 * sigma * u0 - 2*u0**3)
    return gamma1*s+1/6*gamma3*s**3


def InitialArcLength(p):
    # Find the minimimum length s_init to not have a divergence in x(s).
    omega = p[0]
    u0 = p[1]
    # changed from 0.01 to make the code more stable (because you are more far from evaluating x in s=n*delta_s)
    threshold = 0.035
    n = 1
    delta_s = 0.0001
    while (True):
        if South_X(n*delta_s, omega, u0) > threshold:
            break
        else:
            n += 1
    return n*delta_s

# calculate the initial values at the south pole


def InitialValues(s_init, p):
    omega = p[0]
    u0 = p[1]
    sigma = p[2]
    k = p[3]

    return [South_Psi(s_init, omega, u0, sigma),
            South_U(s_init, omega, u0, sigma),
            South_Gamma(s_init, omega, u0, sigma, k),
            South_X(s_init, omega, u0),
            ]


# integrate ODEs
# this is the system of equation to integrate
def ShapeIntegrator(s, z, omega, sigma):
    psi, u, gamma, x = z
    k = 1
    return [DPsi_DS(omega, u),
            DU_DS(omega, psi, u, x, k, gamma),
            DGamma_DS(omega, u, psi, x, k, sigma),
            DX_DS(omega, psi),
            ]

# Jacobian of the shape equations


def ShapeJacobian(s, z, omega, sigma):
    psi, u, gamma, x = z

    k = 1
    a12 = omega
    a11, a13, a14, a15, a16 = 0, 0, 0, 0, 0

    a21 = omega * (u / x * np.sin(psi) + 1 / x**2 * np.cos(psi)**2 - np.sin(psi)
                   # dudpsi
                   ** 2 / x**2 * gamma / (2 * np.pi * k * x) * np.cos(psi))
    a22 = -omega/x*np.cos(psi)  # dudu
    a23 = omega*np.sin(psi)/(2*k*np.pi*x)  # dudgamma
    a24 = omega * (u / x**2 * np.cos(psi) - 2 * np.cos(psi) * np.sin(psi) /
                   x**3 - gamma * np.sin(psi) / (2 * np.pi * k * x**2))  # dudx
    a25, a26 = 0, 0

    a31 = omega*(-np.pi*k*np.cos(psi)/x**2)  # dgammadpsi
    a32 = omega*np.pi*k*2*u  # dgammadu
    a34 = 2*omega*np.pi*k*np.sin(psi)/x**3  # dgammadx
    a33, a35, a36 = 0, 0, 0

    a41 = -omega*np.sin(psi)  # dxdpsi
    a42, a43, a44, a45, a46 = 0, 0, 0, 0, 0

    a54 = omega  # dAdx
    a51, a52, a53, a55, a56 = 0, 0, 0, 0, 0

    a61 = 3/4*omega*x**2*np.cos(psi)  # dVdpsi
    a64 = 3/2*omega*x*np.sin(psi)  # dVdx
    a62, a63, a65, a66 = 0, 0, 0, 0

    return np.array([[a11, a12, a13, a14],
                    [a21, a22, a23, a24],
                    [a31, a32, a33, a34],
                    [a41, a42, a43, a44],
                     ])


def handling_instabilities(t, y,  omega, sigma):
    if np.abs(y[0]) > 10 or np.abs(y[1]) > 10 or np.abs(y[3]) > 10:
        # print("high values")
        return 1
    return -1

# calculate residuals


def Residuales(parameters, boundary_conditions):
    k = 1
    omega, sigma, u0 = parameters
    print(f"free params:{parameters}")
    # calculate initial arc length
    s_init = InitialArcLength((omega, u0))

    # calculate initial values
    z_init = InitialValues(s_init, (omega, u0, sigma, k))
    print(f"approx init values:{z_init}")

    # evaluation points for the solution
    s = np.linspace(s_init, 1, 1000)

    handling_instabilities.terminal = True
    sol = solve_ivp(ShapeIntegrator, t_span=[s_init, 1], y0=z_init, jac=ShapeJacobian, args=(
        omega, sigma), t_eval=s, method='Radau', events=handling_instabilities)

    # sol=solve_ivp(ShapeIntegrator, t_span=[s_init,omega], y0=z_init,jac=ShapeJacobian,args=(omega,sigma),t_eval=s,method='RK23',max_step=0.001)
    # print(sol)
    # print(sol.t.shape)

    if sol.status == -1:
        #     # raise ValueError('error in integration')
        #     raise Warning('error in integration')
        # return 'error in integration'
        print("integration failed")
        return [1e3, 1e3, 1e3]

    z_fina_num = sol.y[:, -1]

    # print(sol.y[:, -1])
    psif, uf, xf, rpa, phi = boundary_conditions
    gammaf = -2 * np.pi * rpa * sigma * np.tan(phi)
    psi = z_fina_num[0]-boundary_conditions[0]
    u = z_fina_num[1]-boundary_conditions[1]
    gamma = z_fina_num[2]-gammaf
    x = z_fina_num[3]-boundary_conditions[2]

    print(psi, u, gamma, x)

    return [psi, u, gamma, x]


def Residuales_Area(parameters, boundary_conditions):
    k = 1
    omega, sigma, u0 = parameters
    # print(f"free params:{parameters}")
    # calculate initial arc length
    s_init = InitialArcLength((omega, u0))

    # calculate initial values
    z_init = InitialValues(s_init, (omega, u0, sigma, k))
    # print(f"approx init values:{z_init}")

    # evaluation points for the solution
    s = np.linspace(s_init, 1, 1000)

    handling_instabilities.terminal = True
    sol = solve_ivp(ShapeIntegrator, t_span=[s_init, 1], y0=z_init, jac=ShapeJacobian, args=(
        omega, sigma), t_eval=s, method='Radau', events=handling_instabilities)

    # sol=solve_ivp(ShapeIntegrator, t_span=[s_init,omega], y0=z_init,jac=ShapeJacobian,args=(omega,sigma),t_eval=s,method='RK23',max_step=0.001)
    # print(sol)
    # print(sol.t.shape)

    if sol.status == -1:
        #     # raise ValueError('error in integration')
        #     raise Warning('error in integration')
        # return 'error in integration'
        print("integration failed")
        return [1e3, 1e3, 1e3]

    z_fina_num = sol.y[:, -1]

    # print(sol.y[:, -1])
    # psif, uf, xf = boundary_conditions
    x = sol.y[3, :]
    max_idx = len(x.nonzero()[0])
    # print(x.shape)

    Astar = 2*np.pi*np.trapz(x, s[:max_idx])
    print(Astar)

    psi = z_fina_num[0]-boundary_conditions[0]
    u = z_fina_num[1]-boundary_conditions[1]
    x = z_fina_num[3]-boundary_conditions[2]
    a = Astar-boundary_conditions[3]

    res = psi**2+u**2+x**2+a**2

    print(f"Omega: {omega:.5f}  Sigma: {
          sigma:.5f}  u0: {u0:.5f}  Err:{res:.5g}")

    print(f"psi: {psi:.3f}  u: {u:.3f}  x: {x:.3f} a: {a:.3f}  Err:{res:.5g}")
    # return [psi, u, x,  a]
    return [psi, u, x,  10*a]  # i am not sure if ustar is defined

    # return [psi, 10*u, 10*x,  a]


def calc_energies(sol, best_parameters, phi_deg, W, rpa):
    s = sol.t
    psi = sol.y[0, :]
    u = sol.y[1, :]
    gamma = sol.y[2, :]
    x = sol.y[3, :]

    omega, sigma, u0 = best_parameters
    phi = deg_to_rad(phi_deg)

    Eun = 2*np.pi*np.trapz(x/2*(u+np.sin(psi)/x)**2 +
                           sigma*x+gamma*(x-np.cos(psi)), s)

    Ebo = (-2*np.pi*W*rpa**2 + 4*rpa**2)*(1-np.cos(phi))
    Etot = Eun + Ebo

    return Ebo, Eun, Etot


def hamiltonian(sol, sigma):
    s = sol.t
    psi = sol.y[0, :]
    u = sol.y[1, :]
    gamma = sol.y[2, :]
    x = sol.y[3, :]

    H = 0.5*u**2 * x + gamma * \
        np.cos(psi)/(2*np.pi)-x*sigma-np.sin(psi) ** 2 / x
    return H


def ShapeCalculator(parameters):
    omega, sigma, u0 = parameters
    k = 1
    s_init = InitialArcLength((omega, u0))
    z_init = InitialValues(s_init, (omega, u0, sigma, k))
    s = np.linspace(s_init, 1, 1000)
    sol = solve_ivp(ShapeIntegrator, t_span=[s_init, 1], y0=z_init, jac=ShapeJacobian, args=(
        omega, sigma), t_eval=s, method='Radau')
    return sol


def ZCoordinate(paras, psi, s):
    # to convert s to z
    omega, u0, sigma = paras
    # k=1 gives linear interpolation
    f = InterpolatedUnivariateSpline(s, omega*np.sin(psi), k=1)
    z = np.array([])
    for s_max in s:
        z = np.append(z, f.integral(s[0], s_max))
        # print(z)
    return z


def PlotShapes(sol, best_parameters, rpa, deg, savefig=True):
    z = ZCoordinate(best_parameters, sol.y[0], sol.t)
    fig = plt.figure(figsize=(7, 7))
    sub1 = fig.add_subplot(111)
    sub1.plot(sol.y[3], z, 'b-')
    sub1.plot(-sol.y[3], z, 'r-')

    if deg > 90:
        y_center = z[-1] - rpa*np.sin(deg_to_rad(deg-90))

    else:
        y_center = z[-1] + rpa*np.sin(deg_to_rad(90-deg))

    circle1 = plt.Circle((0, y_center), rpa, color='k', facecolor=None)
    sub1.add_patch(circle1)

    # sub1.axis('off')
    # plt.xlim([-2.2, 2.2])
    # plt.ylim([-0.2, 5])

    plt.grid()
    sub1.set_aspect("equal")
    if savefig:
        plt.savefig(f"plot/wrapped_rpa{rpa:.2f}_phi{deg:.2f}.png", dpi=200)
    # plt.show()
    return None


def save_best_params(best_parameters, rpa, deg):
    with open("params.dat", "a+") as file:
        file.write(
            f"{rpa:.5f} {deg:.5f} {best_parameters[0]:.5f} {best_parameters[1]:.5f} {best_parameters[2]:.5f}\n")
    return best_parameters


def read_best_params(filename='params.dat'):
    with open(filename, "r") as file:
        params_dict = {}
        for a in file.readlines():
            # print(a)
            a = a.strip()
            b = a.split(' ')
            params_dict[float(b[0]), float(b[1])] = [
                float(b[2]), float(b[3]), float(b[4])]
    return params_dict


def save_coords_file(sol, rpa, deg):
    np.save(f"data/wrapped_rpa{rpa:.2f}_phi{deg:.2f}.npy", sol)

############################################################
############## Main function ###############################
############################################################


def main():
    # constitutive relations
    Rparticle = 3
    Rvesicle = 30
    rpa = Rparticle / Rvesicle

    # wrapping angle
    # phi = np.pi/2
    deg = 30
    phi = deg_to_rad(deg)

    # free parameters initial values
    omega = 3
    sigma = 0.01
    u0 = 0.01

    # Boundary conditions
    psistar = np.pi + phi
    psistar = phi

    ustar = 1/rpa
    xstar = rpa*np.sin(phi)

    A = 4*np.pi*Rvesicle**2  # full vesicle
    A0 = 2*np.pi*Rparticle**2*(1-np.cos((np.pi - phi)))  # wrapped area
    Astar = (A - A0)/Rvesicle**2  # unbound area
    print(A0, A, Astar)

    V = 4/3*np.pi*Rvesicle**3  # full vesicle
    # wrapped area
    V0 = np.pi/3*Rparticle**3*(2+np.cos(np.pi - phi)) * \
        (1-np.cos(np.pi - phi))**2
    Vstar = (V - V0)/Rvesicle**3  # unbound area
    print(V0, V, Vstar)

    # check on xstar value
    if xstar < 0.035:
        print(f"xstar:{xstar}")
        raise ValueError('xstar too small, can lead to divergences')
    else:
        print(f"xstar:{xstar}")

    boundary_conditions = [psistar, ustar, xstar, rpa, phi]
    free_params_extended = [omega, sigma, u0]

    # shoting algorithm and solver
    # result = least_squares(Residuales, free_params_extended, args=(
    #     [boundary_conditions]), method='lm', verbose=2, xtol=1e-14, gtol=1e-14, max_nfev=50000)

    # using area constraint

    boundary_conditions = [psistar, ustar, xstar, Astar]

    result = least_squares(Residuales_Area, free_params_extended, args=(
        [boundary_conditions]), method='lm', verbose=2, xtol=1e-12, gtol=1e-12, max_nfev=50000)

    # # # with bounds
    # result = least_squares(Residuales_Area, free_params_extended, args=(
    #     [boundary_conditions]), method='trf', bounds=((0, -1, 0), (3.14, 0.01, 100)), verbose=2, xtol=1e-12, gtol=1e-12, max_nfev=50000)

    print(f"Err: {result.cost}")

    best_parameters = result.x
    sol = ShapeCalculator(best_parameters)
    print(best_parameters)
    H = hamiltonian(sol, sigma)
    plt.plot(sol.t, H)
    plt.show()

    PlotShapes(sol, best_parameters, rpa, deg)
    plt.show()

    # save_best_params(best_parameters, rpa, deg)
    # save_coords_file(sol, rpa, deg)


# Execute the main function only if the script is run directly
if __name__ == "__main__":
    main()
