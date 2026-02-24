############################################################
# Python code for the shape calculation of endocytosis process of a colloidal particle
# written by Pietro Sillano, in collaboration with Mijke Dijke at TU Delft, May 2024, adapting the code by Felix Frey from Membrane Area Gain and Loss during Cytokinesis, Phys. Rev. E (2022)
############################################################

import numpy as np
import os
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares, differential_evolution
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import least_squares
from utilities import *


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
        elif n > 10000:
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
            DX_DS(omega, psi)
            ]

    return [dpsi, du, dgamma, dx]


# Jacobian of the shape equations


def ShapeJacobian(s, z, omega, sigma):
    psi, u, gamma, x = z

    k = 1
    a12 = omega
    a11, a13, a14, a15, a16 = 0, 0, 0, 0, 0

    a21 = omega * (u / x * np.sin(psi) + 1 / x**2 * np.cos(psi)**2 - np.sin(psi)
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

    Astar = 2*np.pi*omega*np.trapezoid(x, s[:max_idx])
    print(Astar)

    psi = z_fina_num[0]-boundary_conditions[0]
    u = z_fina_num[1]-boundary_conditions[1]
    x = z_fina_num[3]-boundary_conditions[2]
    a = Astar-boundary_conditions[3]

    res = psi**2+u**2+x**2+a**2

    print(f"Omega: {omega:.5f}  Sigma: {sigma:.5f}  u0: {u0:.5f}  Err:{res:.5g}")

    print(f"psi: {psi:.3f}  u: {u:.3f}  x: {x:.3f} a: {a:.3f}  Err:{res:.5g}")
    # return [psi, u, x,  a]
    return [psi, u, x,  a]  # i am not sure if ustar is defined

    # return [psi, 10*u, 10*x,  a]



def Residuales_DoubleShooting(parameters, boundary_conditions, s_mid=0.5):
    k = 1
    omega, sigma, u0, u_contact = parameters
    
    psistar, xstar, Astar = boundary_conditions

    # Determine gamma_contact physically from Hamiltonian conservation H=0
    cos_psi = np.cos(psistar)
    if abs(cos_psi) < 1e-6:
        return [1e5]*4 # Prevent division by zero if psistar is near 90 deg
        
    gamma_contact = (2 * np.pi / cos_psi) * (sigma * xstar + 0.5 * (np.sin(psistar)**2)/xstar - 0.5 * xstar * u_contact**2)

    # sigma can be negative! Only omega must be strictly positive.
    if omega <= 0:
        return [1e5]*4
        
    s_init = InitialArcLength((omega, u0))
    if s_init >= s_mid:
        return [1e5]*4
        
    z_south_init = InitialValues(s_init, (omega, u0, sigma, k))
    
    # We use Radau with default tolerances to avoid hanging on stiff equations near limits
    sol_south = solve_ivp(ShapeIntegrator, [s_init, s_mid], y0=z_south_init, args=(omega, sigma), method='Radau', rtol=1e-5, atol=1e-5)
    
    z_contact_init = [psistar, u_contact, gamma_contact, xstar]
    
    # solve_ivp natively supports backward integration with ShapeIntegrator when t_span is reversed. 
    # Do NOT negate the derivatives, otherwise it integrates backwards in coordinate space as well.
    sol_contact = solve_ivp(ShapeIntegrator, [1.0, s_mid], y0=z_contact_init, args=(omega, sigma), method='Radau', rtol=1e-5, atol=1e-5)
    
    if not sol_south.success or not sol_contact.success:
        return [1e5]*4
        
    z_mid_south = sol_south.y[:, -1]
    z_mid_contact = sol_contact.y[:, -1]
    
    A_south = 2*np.pi*omega*np.trapezoid(sol_south.y[3, :], sol_south.t)
    # Both x and y need to be flipped so the trapezoid rule computes a positive area using increasing x values!
    A_contact = 2*np.pi*omega*np.trapezoid(np.flip(sol_contact.y[3, :]), np.flip(sol_contact.t)) 
    A_total = A_south + A_contact
    
    diff_psi = z_mid_south[0] - z_mid_contact[0]
    diff_u   = z_mid_south[1] - z_mid_contact[1]
    diff_x   = z_mid_south[3] - z_mid_contact[3]
    a_diff   = A_total - Astar
    
    return [diff_psi, diff_u, diff_x, a_diff]

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


def hamiltonian(sol, sigma):
    s = sol.t
    psi = sol.y[0, :]
    u = sol.y[1, :]
    gamma = sol.y[2, :]
    x = sol.y[3, :]

    H = 0.5*u**2 * x + gamma * \
        np.cos(psi)/(2*np.pi)-x*sigma- 0.5 * np.sin(psi) ** 2 / x
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






############################################################
############## Main function ###############################
############################################################


def optimize_for_angle(deg, initial_guess, Rparticle, Rvesicle):
    rpa = Rparticle / Rvesicle
    phi = deg_to_rad(deg)

    psistar = np.pi + phi
    ustar = 1.0 / rpa
    xstar = rpa*np.sin(phi)

    A = 4*np.pi*Rvesicle**2  # full vesicle
    A0 = 2*np.pi*Rparticle**2*(1-np.cos(phi))  # wrapped area
    Astar = (A - A0)/Rvesicle**2  # unbound area dimensionless

    V = 4/3*np.pi*Rvesicle**3  # full vesicle
    V0 = np.pi/3*Rparticle**3*(2+np.cos(phi)) * \
        (1-np.cos(phi))**2
    Vstar = (V - V0)/Rvesicle**3  # unbound volume dimensionless

    # check on xstar value
    if xstar < 0.035:
        print(f"xstar:{xstar} (< 0.035). Proceeding with integration with exact pole handling via Radau.")
        
    print(f"\n--- Optimizing for phi = {deg} degrees ---")
    boundary_conditions_ds = [psistar, xstar, Astar]
    
    # initial_guess: [omega, sigma, u0, u_contact]
    result_ds = least_squares(Residuales_DoubleShooting, initial_guess, args=([boundary_conditions_ds]), method='lm', verbose=0, max_nfev=15000, xtol=1e-7, gtol=1e-7)
    
    # if result_ds.cost > 1e-4:
    #     print("Local optimization failed to converge. Attempting differential evolution global search...")
        
    #     def de_cost(params):
    #         res = Residuales_DoubleShooting(params, boundary_conditions_ds)
    #         return np.sum(np.array(res)**2)
            
    #     # Define bounds centered roughly round realistic scales
    #     bounds = [(0.1, 80.0), (-0.5, 5.0), (-5.0, 5.0), (-5.0, 30.0)]
    #     de_res = differential_evolution(de_cost, bounds, maxiter=50, popsize=15, updating='deferred', workers=-1)
        
    #     print(f"Global search found params: {de_res.x} with cost {de_res.fun}")
        
    #     # Polish it with least_squares
    #     result_ds = least_squares(Residuales_DoubleShooting, de_res.x, args=([boundary_conditions_ds]), method='lm', verbose=0, max_nfev=5000, xtol=1e-7, gtol=1e-7)
        
    print(f"Double Shooting Best Params: {result_ds.x}")
    print(f"Double Shooting Final Cost: {result_ds.cost}")
    
    return result_ds, boundary_conditions_ds, (A, A0, Astar), rpa

def reconstruct_and_save_plot(deg, opt_params, boundary_conditions_ds, rpa):
    omega_ds, sigma_ds, u0_ds, u_contact_ds = opt_params
    psistar, xstar, Astar = boundary_conditions_ds
    
    gamma_contact_ds = (2 * np.pi / np.cos(psistar)) * (sigma_ds * xstar + 0.5 * (np.sin(psistar)**2)/xstar - 0.5 * xstar * u_contact_ds**2)

    s_init_ds = InitialArcLength((omega_ds, u0_ds))
    z_south_init = InitialValues(s_init_ds, (omega_ds, u0_ds, sigma_ds, 1))
    
    s_mid = 0.5
    sol_south = solve_ivp(ShapeIntegrator, [s_init_ds, s_mid], y0=z_south_init, args=(omega_ds, sigma_ds), t_eval=np.linspace(s_init_ds, s_mid, 500), method='Radau', rtol=1e-8, atol=1e-8)
    
    z_contact_init = [psistar, u_contact_ds, gamma_contact_ds, xstar]
    
    if xstar < 0.035:
        # Use expansion-like offset if perfectly at tip to avoid 1/x stiffness. (For now Radau handles generic stiff regions robustly with tight tol).
        pass
        
    sol_contact = solve_ivp(ShapeIntegrator, [1.0, s_mid], y0=z_contact_init, args=(omega_ds, sigma_ds), t_eval=np.linspace(1.0, s_mid, 500), method='Radau', rtol=1e-8, atol=1e-8)
    
    # Remove duplicate point at s_mid from one of the arrays
    t_contact = np.flip(sol_contact.t)
    y_contact = np.flip(sol_contact.y, axis=1)
    
    t_contact = t_contact[1:]
    y_contact = y_contact[:, 1:]
    
    # Concatenate the solutions for plotting
    t_full = np.concatenate((sol_south.t, t_contact))
    y_full = np.concatenate((sol_south.y, y_contact), axis=1)

    # Sort just in case there are minor floating point mismatches
    sort_idx = np.argsort(t_full)
    t_full = t_full[sort_idx]
    y_full = y_full[:, sort_idx]
    
    # Enforce strictly increasing by adding epsilon to duplicates
    for i in range(1, len(t_full)):
        if t_full[i] <= t_full[i-1]:
            t_full[i] = t_full[i-1] + 1e-12

    class DummySol:
        pass
    sol_full = DummySol()
    sol_full.t = t_full
    sol_full.y = y_full
    
    os.makedirs("plot", exist_ok=True)
    filename = f"rpa_{rpa:.2f}_phi_{deg:1f}.png"
    PlotShapes(sol_full, [omega_ds, u0_ds, sigma_ds], rpa, deg, filename)
    plt.close()

def main():
    # constitutive relations
    Rparticle = 5
    Rvesicle = 30
    
    deg = 35
    rpa = Rparticle / Rvesicle
    
    initial_guess = [3.0, 0.08, 1.0, Rvesicle/Rparticle]
    # initial_guess = [9.84205599e+00, 1.23920584e-04, 3.18059165e-01, 1.06206568e+01]    
    
    saved_params = read_best_params(filename, opt_params, rpa, deg)
    # Search for matching parameters with a small tolerance for floating point comparisons
    for (saved_rpa, saved_deg), (params, cost) in saved_params.items():
        if abs(saved_rpa - rpa) < 1e-4 and abs(saved_deg - deg) < 1e-4:
            print(f"Found saved parameters for rpa={rpa:.5f} and deg={deg:.5f} with cost {cost:.3e}: {params}")
            initial_guess = params
            break
            
    result_ds, boundary_conditions_ds, areas, rpa = optimize_for_angle(deg, initial_guess, Rparticle, Rvesicle)
    
    save_best_params(result_ds.x, rpa, deg, result_ds.cost)
    reconstruct_and_save_plot(deg, result_ds.x, boundary_conditions_ds, rpa)

# Execute the main function only if the script is run directly
if __name__ == "__main__":
    main()