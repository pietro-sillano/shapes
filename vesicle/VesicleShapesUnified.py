import numpy as np
import os
import time
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# Import common math and utilities
from core_math import InitialArcLength, InitialValues, ShapeIntegrator, ShapeJacobian, handling_instabilities, deg_to_rad
from utilities import save_best_params, read_best_params, PlotShapes


# =============================================================================
# SINGLE SHOOTING
# =============================================================================

def Residuales(parameters, boundary_conditions):
    omega, sigma, u0 = parameters
    # print(f"free params:{parameters}")
    
    s_init = InitialArcLength(omega, u0)
    z_init = InitialValues(s_init, omega, u0, sigma, 1.0)
    
    s = np.linspace(s_init, 1, 1000)
    handling_instabilities.terminal = True
    sol = solve_ivp(ShapeIntegrator, t_span=[s_init, 1], y0=z_init, jac=ShapeJacobian, args=(omega, sigma), t_eval=s, method='Radau', events=handling_instabilities)

    if sol.status == -1:
        print("integration failed")
        return [1e3, 1e3, 1e3]

    z_fina_num = sol.y[:, -1]
    psif, uf, xf, rpa, phi = boundary_conditions
    gammastar = -2 * np.pi * rpa * sigma * np.tan(phi)
    
    psi = z_fina_num[0] - boundary_conditions[0]
    u = z_fina_num[1] - boundary_conditions[1]
    gamma = z_fina_num[2] - gammastar
    x = z_fina_num[3] - boundary_conditions[2]
    
    return [psi, u, gamma, x]

def Residuales_Area(parameters, boundary_conditions):
    omega, sigma, u0 = parameters
    s_init = InitialArcLength(omega, u0)
    z_init = InitialValues(s_init, omega, u0, sigma, 1.0)
    
    s = np.linspace(s_init, 1, 1000)
    handling_instabilities.terminal = True
    sol = solve_ivp(ShapeIntegrator, t_span=[s_init, 1], y0=z_init, jac=ShapeJacobian, args=(omega, sigma), t_eval=s, method='Radau', events=handling_instabilities)

    if sol.status == -1:
        print("integration failed")
        return [1e3, 1e3, 1e3]

    z_fina_num = sol.y[:, -1]
    x = sol.y[3, :]
    max_idx = len(x.nonzero()[0])
    
    Astar = 2*np.pi*omega*np.trapezoid(x, s[:max_idx])
    
    psi = z_fina_num[0] - boundary_conditions[0]
    u = z_fina_num[1] - boundary_conditions[1]
    x_err = z_fina_num[3] - boundary_conditions[2]
    a_err = Astar - boundary_conditions[3]
    
    return [psi, u, x_err, a_err]


# =============================================================================
# DOUBLE SHOOTING
# =============================================================================

def Residuales_DoubleShooting(parameters, boundary_conditions, s_mid=0.5):
    omega, sigma, u0, u_contact = parameters
    psistar, xstar, Astar = boundary_conditions

    cos_psi = np.cos(psistar)
    if abs(cos_psi) < 1e-6:
        return [1e5]*4 
        
    gamma_contact = (2 * np.pi / cos_psi) * (sigma * xstar + 0.5 * (np.sin(psistar)**2)/xstar - 0.5 * xstar * u_contact**2)

    if omega <= 0: return [1e5]*4
        
    s_init = InitialArcLength(omega, u0)
    if s_init >= s_mid: return [1e5]*4
        
    z_south_init = InitialValues(s_init, omega, u0, sigma, 1.0)
    
    sol_south = solve_ivp(ShapeIntegrator, [s_init, s_mid], y0=z_south_init, args=(omega, sigma), method='Radau', rtol=1e-5, atol=1e-5)
    z_contact_init = [psistar, u_contact, gamma_contact, xstar]
    sol_contact = solve_ivp(ShapeIntegrator, [1.0, s_mid], y0=z_contact_init, args=(omega, sigma), method='Radau', rtol=1e-5, atol=1e-5)
    
    if not sol_south.success or not sol_contact.success:
        return [1e5]*4
        
    z_mid_south = sol_south.y[:, -1]
    z_mid_contact = sol_contact.y[:, -1]
    
    A_south = 2*np.pi*omega*np.trapezoid(sol_south.y[3, :], sol_south.t)
    A_contact = 2*np.pi*omega*np.trapezoid(np.flip(sol_contact.y[3, :]), np.flip(sol_contact.t)) 
    A_total = A_south + A_contact
    
    diff_psi = z_mid_south[0] - z_mid_contact[0]
    diff_u   = z_mid_south[1] - z_mid_contact[1]
    diff_x   = z_mid_south[3] - z_mid_contact[3]
    a_diff   = A_total - Astar
    
    return [diff_psi, diff_u, diff_x, a_diff]

def optimize_for_angle_ds(deg, initial_guess, Rparticle, Rvesicle):
    rpa = Rparticle / Rvesicle
    phi = deg_to_rad(deg)

    psistar = np.pi + phi
    ustar = 1.0 / rpa
    xstar = rpa*np.sin(phi)

    A = 4*np.pi*Rvesicle**2  
    A0 = 2*np.pi*Rparticle**2*(1-np.cos(phi))  
    Astar = (A - A0)/Rvesicle**2  

    print(f"\n--- Optimizing for phi = {deg} degrees (Double Shooting) ---")
    boundary_conditions_ds = [psistar, xstar, Astar]
    
    t0 = time.time()
    result_ds = least_squares(Residuales_DoubleShooting, initial_guess, args=([boundary_conditions_ds]), method='lm', verbose=0, max_nfev=15000, xtol=1e-7, gtol=1e-7)
    t1 = time.time()
    
    print(f"Double Shooting Best Params: {result_ds.x}")
    print(f"Double Shooting Final Cost: {result_ds.cost}")
    print(f"Optimization finished in {t1-t0:.2f}s")
    
    return result_ds, boundary_conditions_ds, rpa

def reconstruct_and_save_plot_ds(deg, opt_params, boundary_conditions_ds, rpa, filename="ds"):
    omega_ds, sigma_ds, u0_ds, u_contact_ds = opt_params
    psistar, xstar, Astar = boundary_conditions_ds
    
    gamma_contact_ds = (2 * np.pi / np.cos(psistar)) * (sigma_ds * xstar + 0.5 * (np.sin(psistar)**2)/xstar - 0.5 * xstar * u_contact_ds**2)

    s_init_ds = InitialArcLength(omega_ds, u0_ds)
    z_south_init = InitialValues(s_init_ds, omega_ds, u0_ds, sigma_ds, 1.0)
    
    s_mid = 0.5
    sol_south = solve_ivp(ShapeIntegrator, [s_init_ds, s_mid], y0=z_south_init, args=(omega_ds, sigma_ds), t_eval=np.linspace(s_init_ds, s_mid, 500), method='Radau', rtol=1e-8, atol=1e-8)
    
    z_contact_init = [psistar, u_contact_ds, gamma_contact_ds, xstar]
    sol_contact = solve_ivp(ShapeIntegrator, [1.0, s_mid], y0=z_contact_init, args=(omega_ds, sigma_ds), t_eval=np.linspace(1.0, s_mid, 500), method='Radau', rtol=1e-8, atol=1e-8)
    
    t_contact = np.flip(sol_contact.t)[1:]
    y_contact = np.flip(sol_contact.y, axis=1)[:, 1:]
    
    t_full = np.concatenate((sol_south.t, t_contact))
    y_full = np.concatenate((sol_south.y, y_contact), axis=1)

    sort_idx = np.argsort(t_full)
    t_full = t_full[sort_idx]
    y_full = y_full[:, sort_idx]
    
    for i in range(1, len(t_full)):
        if t_full[i] <= t_full[i-1]: t_full[i] = t_full[i-1] + 1e-12

    class DummySol: pass
    sol_full = DummySol()
    sol_full.t = t_full
    sol_full.y = y_full
    
    os.makedirs("plot", exist_ok=True)
    full_filename = f"{filename}_rpa_{rpa:.2f}_phi_{deg:1f}.png"
    PlotShapes(sol_full, [omega_ds, u0_ds, sigma_ds], rpa, deg, full_filename)
    plt.close()


# =============================================================================
# MULTIPLE SHOOTING
# =============================================================================

def Residuales_MultipleShooting(parameters, boundary_conditions, num_segments=4):
    M = num_segments - 1
    omega, sigma, u0, u_contact = parameters[0:4]
    psistar, xstar, Astar = boundary_conditions

    cos_psi = np.cos(psistar)
    if abs(cos_psi) < 1e-6:
        return [1e5]*(4 + 4*M) 
        
    gamma_contact = (2 * np.pi / cos_psi) * (sigma * xstar + 0.5 * (np.sin(psistar)**2)/xstar - 0.5 * xstar * u_contact**2)

    if omega <= 0: return [1e5]*(4 + 4*M)
        
    s_init = InitialArcLength(omega, u0)
    s_nodes = np.linspace(s_init, 1.0, num_segments + 1)
    
    if s_init >= s_nodes[1]: return [1e5]*(4 + 4*M)
        
    z_south_init = InitialValues(s_init, omega, u0, sigma, 1.0)
    
    internal_states = np.array(parameters[4:]).reshape(M, 4)
    residuals = []
    A_total = 0.0
    
    for i in range(M):
        s_start = s_nodes[i]
        s_end = s_nodes[i+1]
        z_init = z_south_init if i == 0 else internal_states[i-1]
            
        sol = solve_ivp(ShapeIntegrator, [s_start, s_end], y0=z_init, args=(omega, sigma), method='Radau', rtol=1e-5, atol=1e-5)
        if not sol.success: return [1e5]*(4 + 4*M)
            
        z_final = sol.y[:, -1]
        z_target = internal_states[i]
        residuals.extend((z_final - z_target).tolist())
        A_total += 2*np.pi*omega*np.trapezoid(sol.y[3, :], sol.t)
        
    s_end = s_nodes[M]
    z_contact_init = [psistar, u_contact, gamma_contact, xstar]
    
    sol_contact = solve_ivp(ShapeIntegrator, [1.0, s_end], y0=z_contact_init, args=(omega, sigma), method='Radau', rtol=1e-5, atol=1e-5)
    if not sol_contact.success: return [1e5]*(4 + 4*M)
        
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
        s_target = s_nodes[i+1]
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
    xstar = rpa*np.sin(phi)
    A = 4*np.pi*Rvesicle**2
    A0 = 2*np.pi*Rparticle**2*(1-np.cos(phi))
    Astar = (A - A0)/Rvesicle**2

    print(f"--- Optimizing for phi = {deg} degrees (Multiple Shooting, segments={num_segments}) ---")
    boundary_conditions = [psistar, xstar, Astar]
    
    initial_guess_full = generate_initial_guess_ms(initial_guess_base, num_segments, boundary_conditions)
    t0 = time.time()
    result_ms = least_squares(Residuales_MultipleShooting, initial_guess_full, 
                              args=(boundary_conditions, num_segments), method='lm', 
                              verbose=0, max_nfev=25000, xtol=1e-7, gtol=1e-7)
    t1 = time.time()
    
    print(f"Optimization finished in {t1-t0:.2f}s")
    print(f"Base Params: {result_ms.x[:4]}")
    print(f"Final Cost: {result_ms.cost}")
    
    return result_ms, boundary_conditions, rpa, num_segments

def reconstruct_and_save_plot_ms(deg, opt_params, boundary_conditions, rpa, num_segments, filename="ms"):
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
    full_filename = f"{filename}_rpa_{rpa:.2f}_phi_{deg:1f}.png"
    PlotShapes(sol_full, [omega_ms, u0_ms, sigma_ms], rpa, deg, full_filename)
    plt.close()
