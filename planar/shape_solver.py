############################################################
# Planar membrane engulfment solver
# Multiple Shooting with least_squares (adapted from vesicle code)
############################################################

import numpy as np
import warnings
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import os
import warnings
import pandas as pd
from utilities import read_best_params, save_best_params, save_energies

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ── Physical parameters ──────────────────────────────────
kappa = 1.0
Sigma = 0.2       # mechanical tension
m     = 0.0       # spontaneous curvature
sigma = Sigma + 2 * kappa * m**2
W     = 1.0       # adhesion energy density |W|

# ── ODE system  ──────────────────────────────────────────
# State vector:  Y = [psi, u, x, z]
# F_me is computed post-hoc by a separate integration
# This avoids stiffness from coupling the energy integral
# during the shooting optimization.

def shape_rhs(s, Y, omega):
    """RHS of the shape equations on s in [0, 1]."""
    psi, u, x, z = Y

    sin_psi = np.sin(psi)
    cos_psi = np.cos(psi)

    if x < 1e-8:
        x = 1e-8
    if abs(cos_psi) < 1e-8:
        cos_psi = np.sign(cos_psi + 1e-15) * 1e-8

    dpsi = omega * u

    du = omega * (
        -(cos_psi / x) * u
        + (sin_psi * cos_psi) / x**2
        + (sin_psi / (2 * cos_psi)) * (
            sigma / kappa - u**2 + sin_psi**2 / x**2
            + 4 * m * sin_psi / x
        )
    )

    dx = omega * cos_psi
    dz = -omega * sin_psi

    return [dpsi, du, dx, dz]


def shape_jac(s, Y, omega):
    """Analytical Jacobian df/dY for the shape equations.
    Providing this avoids costly finite-difference Jacobian estimation
    inside the Radau solver (saves ~4 RHS evaluations per step)."""
    psi, u, x, z = Y

    sin_psi = np.sin(psi)
    cos_psi = np.cos(psi)

    if x < 1e-8:
        x = 1e-8
    if abs(cos_psi) < 1e-8:
        cos_psi = np.sign(cos_psi + 1e-15) * 1e-8

    sp = sin_psi
    cp = cos_psi
    sig_k = sigma / kappa

    # Precompute bracket term for du
    bracket = sig_k - u**2 + sp**2 / x**2 + 4 * m * sp / x

    # ── Row 0: d(dpsi)/d[psi, u, x, z]
    # dpsi = omega * u
    J00 = 0.0
    J01 = omega
    J02 = 0.0
    J03 = 0.0

    # ── Row 1: d(du)/d[psi, u, x, z]
    # du = omega * ( -(cp/x)*u + (sp*cp)/x^2 + (sp/(2*cp)) * bracket )
    # where bracket = sig_k - u^2 + sp^2/x^2 + 4*m*sp/x

    # d(du)/d(psi):
    #  term1: d/dpsi[-(cp/x)*u] = (sp/x)*u
    #  term2: d/dpsi[(sp*cp)/x^2] = (cp^2 - sp^2)/x^2
    #  term3: d/dpsi[(sp/(2*cp))*bracket]
    #       = (1/(2*cp^2))*bracket  +  (sp/(2*cp)) * d(bracket)/dpsi
    #  d(bracket)/dpsi = 2*sp*cp/x^2 + 4*m*cp/x
    J10 = omega * (
        (sp / x) * u
        + (cp**2 - sp**2) / x**2
        + bracket / (2 * cp**2)
        + (sp / (2 * cp)) * (2 * sp * cp / x**2 + 4 * m * cp / x)
    )

    # d(du)/d(u):
    #  term1: -(cp/x)
    #  term3: (sp/(2*cp)) * (-2*u)
    J11 = omega * (-(cp / x) - sp * u / cp)

    # d(du)/d(x):
    #  term1: d/dx[-(cp/x)*u] = (cp/x^2)*u
    #  term2: d/dx[(sp*cp)/x^2] = -2*(sp*cp)/x^3
    #  term3: (sp/(2*cp)) * d(bracket)/dx
    #  d(bracket)/dx = -2*sp^2/x^3 - 4*m*sp/x^2
    J12 = omega * (
        (cp / x**2) * u
        - 2 * (sp * cp) / x**3
        + (sp / (2 * cp)) * (-2 * sp**2 / x**3 - 4 * m * sp / x**2)
    )

    J13 = 0.0

    # ── Row 2: d(dx)/d[psi, u, x, z]
    # dx = omega * cos_psi
    J20 = -omega * sp
    J21 = 0.0
    J22 = 0.0
    J23 = 0.0

    # ── Row 3: d(dz)/d[psi, u, x, z]
    # dz = -omega * sin_psi
    J30 = -omega * cp
    J31 = 0.0
    J32 = 0.0
    J33 = 0.0

    return [
        [J00, J01, J02, J03],
        [J10, J11, J12, J13],
        [J20, J21, J22, J23],
        [J30, J31, J32, J33],
    ]




def _divergence_event(s, Y, omega):
    """Terminate integration if psi diverges beyond physical bounds."""
    return 100.0 - abs(Y[0])  # triggers when |psi| > 100
_divergence_event.terminal = True


def residuals_multiple_shooting(params, phi, rpa, num_segments=4):
    """Compute residuals for the multiple shooting method.
    
    Parameters:
      params = [omega, u0, *internal_states]
      where internal_states has (num_segments-1) * 4 values
      (psi, u, x, z at each internal node)
    
    Boundary conditions:
      s=0: psi=phi, x=rpa*sin(phi), z=rpa*(1-cos(phi))   [3 known]
           u=u0 (shooting parameter)
      s=1: psi=0, u=0                                       [2 targets]
    
    Unknowns: omega, u0, + 4*(num_segments-1) internal states
    Residuals: 4*(num_segments-1) continuity + 2 boundary = 4*num_segments - 2
    """
    omega = params[0]
    u0 = params[1]
    M = num_segments - 1  # number of internal nodes
    
    # Omega must be positive and bounded
    omega_max = 10 * np.sqrt(kappa / sigma) * rpa
    # omega_max = 50 * rpa
    # print(f"Omega upper bound: {omega_max}")
    if omega <= 0 or omega > omega_max:
        return np.full(4 * M + 2, 1e5)
    
    # Internal node states
    if M > 0:
        internal_states = params[2:].reshape(M, 4)
    
    s_nodes = np.linspace(0, 1, num_segments + 1)
    
    residuals = []
    
    for i in range(num_segments):
        s_start = s_nodes[i]
        s_end = s_nodes[i + 1]
        
        if i == 0:
            # Initial conditions at s=0
            y_init = [phi, u0, rpa * np.sin(phi), rpa * (1 - np.cos(phi))]
        else:
            y_init = internal_states[i - 1].tolist()
        
        try:
            sol = solve_ivp(shape_rhs, [s_start, s_end], y_init,
                           args=(omega,), method='Radau',
                           rtol=1e-5, atol=1e-5)
            if not sol.success:
                return np.full(4 * M + 2, 1e5)
            y_end = sol.y[:, -1]
        except Exception:
            return np.full(4 * M + 2, 1e5)
        
        if i < num_segments - 1:
            # Continuity: end of segment i must match start of segment i+1
            target = internal_states[i]
            residuals.extend((y_end - target).tolist())
        else:
            # Last segment: enforce far-field BCs
            residuals.append(y_end[0])     # psi(1) = 0
            residuals.append(y_end[1])     # u(1)   = 0
    
    return np.array(residuals)


def generate_initial_guess_ms(phi, rpa, omega_guess, u0_guess, num_segments=4):
    """Generate initial guess for multiple shooting by forward integration."""
    M = num_segments - 1
    s_nodes = np.linspace(0, 1, num_segments + 1)
    
    y0 = [phi, u0_guess, rpa * np.sin(phi), rpa * (1 - np.cos(phi))]
    
    # Try forward integration to get states at internal nodes
    internal_states = np.zeros((M, 4))
    
    try:
        sol = solve_ivp(shape_rhs, [0, 1], y0, args=(omega_guess,),
                       method='Radau',
                       rtol=1e-5, atol=1e-5, dense_output=True)
        
        if sol.success:
            for i in range(M):
                s_target = s_nodes[i + 1]
                internal_states[i] = sol.sol(s_target)
        else:
            # Use linearized exponential decay
            alpha = omega_guess * np.sqrt(sigma / kappa)
            for i in range(M):
                s = s_nodes[i + 1]
                psi_s = phi * np.exp(-alpha * s)
                u_s = -(alpha / omega_guess) * phi * np.exp(-alpha * s)
                x_s = rpa * np.sin(phi) + omega_guess * s
                z_s = rpa * (1 - np.cos(phi)) - omega_guess * (phi / alpha) * (1 - np.exp(-alpha * s))
                internal_states[i] = [psi_s, u_s, x_s, z_s]
    except Exception:
        alpha = omega_guess * np.sqrt(sigma / kappa)
        for i in range(M):
            s = s_nodes[i + 1]
            psi_s = phi * np.exp(-alpha * s)
            u_s = -(alpha / omega_guess) * phi * np.exp(-alpha * s)
            x_s = rpa * np.sin(phi) + omega_guess * s
            z_s = rpa * (1 - np.cos(phi)) - omega_guess * (phi / alpha) * (1 - np.exp(-alpha * s))
            internal_states[i] = [psi_s, u_s, x_s, z_s]
    
    return np.concatenate([[omega_guess, u0_guess], internal_states.flatten()])


def compute_energy(sol_full_t, sol_full_y, omega):
    """Compute F_me by integrating the energy equation post-hoc using Simpson's rule."""
    from scipy.integrate import simpson
    
    psi = sol_full_y[0]
    u = sol_full_y[1]
    x = np.maximum(sol_full_y[2], 1e-8)
    
    sin_psi = np.sin(psi)
    cos_psi = np.cos(psi)

    dF_ds = omega * np.pi * kappa * x * (
        (u + sin_psi / x) * (u + sin_psi / x + 4 * m)
        + (2 * sigma / kappa) * (1 - cos_psi)
    )
    
    F_me_un = simpson(dF_ds, x=sol_full_t)
    return F_me_un


def solve_for_angle(phi, rpa, guess_params, num_segments=4):
    """Solve for a given wrapping angle using multiple shooting."""
    M = num_segments - 1
    
    try:
        result = least_squares(
            residuals_multiple_shooting,
            guess_params,
            args=(phi, rpa, num_segments),method='lm', verbose=0, max_nfev=15000, xtol=1e-7, gtol=1e-7)
        
        if result.cost < 100:  # relaxed tolerance for far-field BCs
            print(f"  Optimization converged with cost={result.cost:.2e}")
            omega_opt = result.x[0]
            u0_opt = result.x[1]
            
            # Reconstruct full solution for energy and plotting
            s_nodes = np.linspace(0, 1, num_segments + 1)
            internal_states = result.x[2:].reshape(M, 4) if M > 0 else []
            
            t_full = []
            y_full = []
            
            for i in range(num_segments):
                s_start = s_nodes[i]
                s_end = s_nodes[i + 1]
                if i == 0:
                    y_init = [phi, u0_opt, rpa * np.sin(phi), rpa * (1 - np.cos(phi))]
                else:
                    y_init = internal_states[i - 1].tolist()
                
                sol = solve_ivp(shape_rhs, [s_start, s_end], y_init,
                               args=(omega_opt,), method='Radau',
                               rtol=1e-10, atol=1e-10,
                               t_eval=np.linspace(s_start, s_end, 100))
                
                if i < num_segments - 1:
                    t_full.append(sol.t[:-1])
                    y_full.append(sol.y[:, :-1])
                else:
                    t_full.append(sol.t)
                    y_full.append(sol.y)
            
            t_full = np.concatenate(t_full)
            y_full = np.concatenate(y_full, axis=1)
            
            # Compute energy post-hoc
            F_me_un = compute_energy(t_full, y_full, omega_opt)
            
            # Package as a simple object
            class Sol:
                pass
            sol_obj = Sol()
            sol_obj.t = t_full
            sol_obj.y = y_full
            
            # Extract final values u(1) and psi(1)
            psi1_opt = y_full[0, -1]
            u1_opt = y_full[1, -1]
            
            return u0_opt, omega_opt, sol_obj, F_me_un, True, result.x, result.cost, u1_opt, psi1_opt
        else:
            print(f"  Optimization converged but cost={result.cost:.2e} too high")
            print(result.x[0],result.x[1])
            return None, None, None, 0.0, False, None, result.cost, None, None
            
    except Exception as e:
        print(f"  Error: {e}")
        return None, None, None, 0.0, False, None, 1e5, None, None


# ── Energy calculations ──────────────────────────────────

def calculate_energies(phi, F_me_un, R_pa, kappa, sigma, m):

    F_me_bo = (4 * np.pi * kappa * (1 + 2 * m * R_pa) * (1 - np.cos(phi))
               + sigma * np.pi * R_pa**2 * (1 - np.cos(phi))**2)
    F_ad = -2 * np.pi * R_pa**2 * (1 - np.cos(phi))
    return F_me_bo, F_ad


# ── Shape plotting ────────────────────────────────────────

def reconstruct_solution(phi, rpa, u0, omega, num_segments=4):
    """Reconstruct the full ODE solution from optimized u0 and omega.
    Uses a single forward integration since u0/omega are already good.
    Returns a Sol-like object with .t and .y, or None if integration fails."""
    y0 = [phi, u0, rpa * np.sin(phi), rpa * (1 - np.cos(phi))]

    sol = solve_ivp(shape_rhs, [0, 1], y0, args=(omega,),
                   method='Radau', rtol=1e-10, atol=1e-10,
                   max_step=0.01,
                   t_eval=np.linspace(0, 1, 200))

    if not sol.success:
        return None

    class Sol:
        pass
    sol_obj = Sol()
    sol_obj.t = sol.t
    sol_obj.y = sol.y
    return sol_obj


def plot_shape(sol, omega, rpa, deg, filename):
    """Plot the axisymmetric membrane cross-section with the particle."""
    x_vals = sol.y[2]

    # Shift z so that z(s=1) = 0
    z_at_infinity = sol.y[3, -1]
    z_vals = -(sol.y[3] - z_at_infinity)  # flip so membrane bends downward relative to z=0

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(x_vals, z_vals, 'b-', linewidth=2, label='membrane')
    ax.plot(-x_vals, z_vals, 'b-', linewidth=2)

    phi_rad = np.radians(deg)
    z_contact = -(sol.y[3, 0] - z_at_infinity)  # flipped and shifted contact z
    z_center = z_contact + rpa * np.cos(phi_rad)  # particle center above

    circle = plt.Circle((0, z_center), rpa, edgecolor='k',
                         facecolor='lightgray', alpha=0.3, linewidth=1.5)
    ax.add_patch(circle)

    plt.xlim(-40,40)
    plt.ylim(-5*rpa,10)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Planar Wrapping: $\\phi$ = {deg:.1f}°")
    plt.tight_layout()
    plt.savefig(f"plot/{filename}", dpi=100)
    plt.close()


# ── Plot-only mode ────────────────────────────────────────

def plot_from_params(params_file="results/params.csv", cost_threshold=1e-1, num_segments=4):
    """Reconstruct and plot shapes from saved parameters without re-optimizing.
    Only processes entries with cost < cost_threshold."""
    os.makedirs('plot', exist_ok=True)

    df = pd.read_csv(params_file)
    print(f"Loaded {len(df)} entries from {params_file}")

    for _, row in df.iterrows():
        rpa = row['rpa']
        deg = row['deg']
        u0 = row['u0']
        omega = row['omega']
        cost = row.get('cost', np.nan)
        success = row.get('success', True)

        if not success or cost > cost_threshold:
            continue

        phi = np.radians(deg)
        print(f"  Reconstructing R_pa={rpa}, phi={deg}° (cost={cost:.2e})")

        sol = reconstruct_solution(phi, rpa, u0, omega, num_segments)
        if sol is not None:
            F_me_un = compute_energy(sol.t, sol.y, omega)
            print(f"    F_me_un={F_me_un:.4f}")
            plot_shape(sol, omega, rpa, deg,
                      f"shape_rpa_{rpa:.1f}_phi_{deg:.1f}.png")
        else:
            print(f"    FAILED to reconstruct")

    print("Done plotting from saved parameters.")


# ── Main sweep ────────────────────────────────────────────

def main():
    num_segments = 4
    params_file = "results/params.csv"
    energies_file = "results/energies.csv"
    
    os.makedirs('results', exist_ok=True)
    os.makedirs('plot', exist_ok=True)

    # Exploration ranges
    rpa_vals = [5.0]
    angles_deg = np.arange(10, 70, 10)
    
    lam = np.sqrt(kappa / sigma)  # characteristic length ~3.16
    print(f"characteristic length: {lam}")
    omega_guess = 10 * lam

    for rpa in rpa_vals:
        print(f"\n======== R_pa = {rpa} ========")
        prev_params = None  # reset continuation for each R_pa
        rpa_alt=rpa+1.0

        
        for deg in angles_deg:

            if deg < 85 or deg > 95 :
                phi = np.radians(deg)
                print(f"\n── phi = {deg:.1f}° ──")

                # 1. Check if we already have good scalar params for this combination
                saved_vals = read_best_params(params_file, rpa, deg)

                
                if saved_vals is not None:
                    u0_saved, omega_saved = saved_vals
                    print(f"  Found saved parameters for R_pa={rpa}, phi={deg}°")
                    guess = generate_initial_guess_ms(phi, rpa, omega_saved, u0_saved, num_segments)
                
                elif prev_params is not None:
                    # Use continuation from previous angle if available
                    print("use previous params")
                    guess = prev_params
                else:
                    # Default initial guess
                    u0_guess = -np.sqrt(sigma / kappa) * phi
                    guess = generate_initial_guess_ms(phi, rpa, omega_guess, u0_guess, num_segments)

                u0_opt, omega_opt, sol, F_me_un, success, opt_params, cost, u1_opt, psi1_opt = \
                    solve_for_angle(phi, rpa, guess, num_segments)

                if success:
                    # Save the optimized scalar params including boundary values
                    save_best_params(params_file, u0_opt, omega_opt, rpa, deg, cost, success, u1_opt, psi1_opt)

                    # Now calculate energies using current W (decoupled)
                    F_me_bo, F_ad = calculate_energies(
                        phi, F_me_un, rpa, kappa, sigma, m)
                    
                    save_energies(energies_file, rpa, deg, F_me_un, F_me_bo, F_ad, cost)

                    prev_params = opt_params  # update continuation
                    print(f"  u0={u0_opt:.6f}, omega={omega_opt:.4f}")
                    print(f"  u(1)={u1_opt:.6e}, psi(1)={psi1_opt:.6e}")
                    print(f"  F_me_un={F_me_un:.4f}")

                    plot_shape(sol, omega_opt, rpa, deg,
                            f"shape_rpa_{rpa:.1f}_phi_{deg:.1f}.png")
                else:
                    print(f"  FAILED for R_pa={rpa}, phi = {deg:.1f}°.")
                    print(opt_params)
                    prev_params = None  # reset continuation

    if os.path.exists(energies_file):
        df_results = pd.read_csv(energies_file)
        print(f"\nSummary plot using results from {energies_file}")

        plt.figure(figsize=(10, 6))
        rpa_vals_in_data = df_results['rpa'].unique()
        for rpa_val in rpa_vals_in_data:
            sub = df_results[df_results['rpa'] == rpa_val].sort_values('phi_deg')
            if not sub.empty:
                plt.plot(sub['phi_deg'], sub['F_me_un'], marker='o', label=f'R_pa={rpa_val}')
        
        plt.xlabel(r'Wrapping angle $\phi$ (degrees)')
        plt.ylabel('Total Energy')
        plt.title('Energy Landscape: Planar Membrane Engulfment')
        plt.legend()
        plt.grid(True)
        plt.savefig('results/energy_sweep.png', dpi=300)
        plt.close()
        print("Summary plot saved to results/energy_sweep.png")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        plot_from_params()
    else:
        main()
