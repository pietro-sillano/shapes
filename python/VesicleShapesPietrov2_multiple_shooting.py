############################################################
# Python code for the shape calculation of endocytosis process of a colloidal particle
# Multiple Shooting Procedure based on double shooting
############################################################

import numpy as np
import os
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from numba import njit
import time
from utilities import *



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





def Residuales_MultipleShooting(parameters, boundary_conditions, num_segments=4):
    M = num_segments - 1
    omega, sigma, u0, u_contact = parameters[0:4]
    
    psistar, xstar, Astar = boundary_conditions

    cos_psi = np.cos(psistar)
    if abs(cos_psi) < 1e-6:
        return [1e5]*(4 + 4*M) 
        
    gamma_contact = (2 * np.pi / cos_psi) * (sigma * xstar + 0.5 * (np.sin(psistar)**2)/xstar - 0.5 * xstar * u_contact**2)

    if omega <= 0:
        return [1e5]*(4 + 4*M)
        
    s_init = InitialArcLength(omega, u0)
    s_nodes = np.linspace(s_init, 1.0, num_segments + 1)
    
    if s_init >= s_nodes[1]:
        return [1e5]*(4 + 4*M)
        
    z_south_init = InitialValues(s_init, omega, u0, sigma, 1.0)
    
    internal_states = np.array(parameters[4:]).reshape(M, 4)
    residuals = []
    A_total = 0.0
    
    for i in range(M):
        s_start = s_nodes[i]
        s_end = s_nodes[i+1]
        
        z_init = z_south_init if i == 0 else internal_states[i-1]
            
        sol = solve_ivp(ShapeIntegrator, [s_start, s_end], y0=z_init, args=(omega, sigma), method='Radau', rtol=1e-5, atol=1e-5)
        
        if not sol.success:
            return [1e5]*(4 + 4*M)
            
        z_final = sol.y[:, -1]
        z_target = internal_states[i]
        residuals.extend((z_final - z_target).tolist())
        
        A_total += 2*np.pi*omega*np.trapezoid(sol.y[3, :], sol.t)
        
    # Last segment implicitly backward
    s_start = 1.0
    s_end = s_nodes[M]
    z_contact_init = [psistar, u_contact, gamma_contact, xstar]
    
    sol_contact = solve_ivp(ShapeIntegrator, [1.0, s_end], y0=z_contact_init, args=(omega, sigma), method='Radau', rtol=1e-5, atol=1e-5)
    
    if not sol_contact.success:
        return [1e5]*(4 + 4*M)
        
    z_mid_contact = sol_contact.y[:, -1]
    z_target = internal_states[M-1]
    
    diff_psi = z_mid_contact[0] - z_target[0]
    diff_u   = z_mid_contact[1] - z_target[1]
    diff_x   = z_mid_contact[3] - z_target[3]
    
    A_contact = 2*np.pi*omega*np.trapezoid(np.flip(sol_contact.y[3, :]), np.flip(sol_contact.t)) 
    A_total += A_contact
    
    a_diff = A_total - Astar
    
    residuals.extend([diff_psi, diff_u, diff_x, a_diff])
    
    return residuals

def generate_initial_guess_ms(base_params, num_segments, boundary_conditions):
    psistar, xstar, Astar = boundary_conditions
    omega, sigma, u0, u_contact = base_params
    
    M = num_segments - 1
    s_init = InitialArcLength(omega, u0)
    s_nodes = np.linspace(s_init, 1.0, num_segments + 1)
    
    internal_states = np.zeros((M, 4))
    
    s_mid = 0.5
    z_south_init = InitialValues(s_init, omega, u0, sigma, 1.0)
    sol_south = solve_ivp(ShapeIntegrator, [s_init, s_mid], y0=z_south_init, args=(omega, sigma), dense_output=True, method='Radau')
    
    cos_psi = np.cos(psistar)
    gamma_contact_val = (2 * np.pi / cos_psi) * (sigma * xstar + 0.5 * (np.sin(psistar)**2)/xstar - 0.5 * xstar * u_contact**2)
    z_contact_init = [psistar, u_contact, gamma_contact_val, xstar]
    sol_contact = solve_ivp(ShapeIntegrator, [1.0, s_mid], y0=z_contact_init, args=(omega, sigma), dense_output=True, method='Radau')
    
    for i in range(M):
        s_target = s_nodes[i+1] # s_1 to s_M
        if s_target <= s_mid:
            if sol_south.success and s_target >= s_init:
                state = sol_south.sol(s_target)
            else:
                state = z_south_init
        else:
            if sol_contact.success and s_target <= 1.0:
                state = sol_contact.sol(s_target)
            else:
                state = z_contact_init
        internal_states[i] = state
        
    return np.concatenate((base_params, internal_states.flatten()))

def optimize_for_angle_ms(deg, initial_guess_base, Rparticle, Rvesicle, num_segments=4):
    rpa = Rparticle / Rvesicle
    phi = deg_to_rad(deg)

    psistar = np.pi + phi
    ustar = 1.0 / rpa
    xstar = rpa*np.sin(phi)

    A = 4*np.pi*Rvesicle**2
    A0 = 2*np.pi*Rparticle**2*(1-np.cos(phi))
    Astar = (A - A0)/Rvesicle**2

    print(f"--- Optimizing for phi = {deg} degrees (Multiple Shooting, segments={num_segments}) ---")
    boundary_conditions = [psistar, xstar, Astar]
    
    initial_guess_full = generate_initial_guess_ms(initial_guess_base, num_segments, boundary_conditions)
    # print(f"prepared initial guess: {initial_guess_full}")io
    t0 = time.time()
    result_ms = least_squares(Residuales_MultipleShooting, initial_guess_full, 
                              args=(boundary_conditions, num_segments), method='lm', 
                              verbose=0, max_nfev=25000, xtol=1e-7, gtol=1e-7)
    t1 = time.time()
    
    print(f"Optimization finished in {t1-t0:.2f}s")
    print(f"Base Params: {result_ms.x[:4]}")
    print(f"Final Cost: {result_ms.cost}")
    
    return result_ms, boundary_conditions, rpa, num_segments

def reconstruct_and_save_plot_ms(deg, opt_params, boundary_conditions, rpa, num_segments):
    M = num_segments - 1
    omega_ms, sigma_ms, u0_ms, u_contact_ms = opt_params[0:4]
    internal_states = np.array(opt_params[4:]).reshape(M, 4)
    psistar, xstar, Astar = boundary_conditions
    
    gamma_contact = (2 * np.pi / np.cos(psistar)) * (sigma_ms * xstar + 0.5 * (np.sin(psistar)**2)/xstar - 0.5 * xstar * u_contact_ms**2)

    s_init_ms = InitialArcLength(omega_ms, u0_ms)
    s_nodes = np.linspace(s_init_ms, 1.0, num_segments + 1)
    
    t_full, y_full = [], []
    
    for i in range(M):
        s_start, s_end = s_nodes[i], s_nodes[i+1]
        z_init = InitialValues(s_init_ms, omega_ms, u0_ms, sigma_ms, 1.0) if i == 0 else internal_states[i-1]
        sol = solve_ivp(ShapeIntegrator, [s_start, s_end], y0=z_init, args=(omega_ms, sigma_ms), method='Radau', rtol=1e-8, atol=1e-8, t_eval=np.linspace(s_start, s_end, 200))
        t_full.append(sol.t[:-1])
        y_full.append(sol.y[:, :-1])
        
    z_contact_init = [psistar, u_contact_ms, gamma_contact, xstar]
    sol_contact = solve_ivp(ShapeIntegrator, [1.0, s_nodes[M]], y0=z_contact_init, args=(omega_ms, sigma_ms), method='Radau', rtol=1e-8, atol=1e-8, t_eval=np.linspace(1.0, s_nodes[M], 200))
    
    t_full.append(np.flip(sol_contact.t))
    y_full.append(np.flip(sol_contact.y, axis=1))
    
    t_full = np.concatenate(t_full)
    y_full = np.concatenate(y_full, axis=1)
    
    sort_idx = np.argsort(t_full)
    t_full = t_full[sort_idx]
    y_full = y_full[:, sort_idx]
    
    for i in range(1, len(t_full)):
        if t_full[i] <= t_full[i-1]: t_full[i] = t_full[i-1] + 1e-12

    class DummySol: pass
    sol_full = DummySol()
    sol_full.t, sol_full.y = t_full, y_full
    
    os.makedirs("plot", exist_ok=True)
    filename = f"ms_rpa_{rpa:.2f}_phi_{deg:1f}.png"
    PlotShapes(sol_full, [omega_ms, u0_ms, sigma_ms], rpa, deg, filename)
    plt.close()

def main():
    Rparticle = 5
    Rvesicle = 30
    deg = 60
    
    initial_guess_base = [8.76545115, 0.21125935, 0.21337616, 6.39789426] # this works for rpa = 0.17 and deg=60
    num_segments = 4
    
    result_ms, bc, rpa, n_seg = optimize_for_angle_ms(deg, initial_guess_base, Rparticle, Rvesicle, num_segments)
    reconstruct_and_save_plot_ms(deg, result_ms.x, bc, rpa, n_seg)

if __name__ == "__main__":
    main()
