import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
from scipy.interpolate import InterpolatedUnivariateSpline
from numba import njit


@njit
def deg_to_rad(deg):
    return np.pi*deg/180

def ZCoordinate(paras, psi, s):
    omega, u0, sigma = paras
    f = InterpolatedUnivariateSpline(s, omega*np.sin(psi), k=1)
    z = np.array([])
    for s_max in s:
        z = np.append(z, f.integral(s[0], s_max))
    return z

def save_coords_file(sol, filename):
    np.save(f"data/{filename}.npy", sol)

def PlotShapes(sol, best_parameters, rpa, deg, filename):
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

    plt.xlim([-2,2])
    plt.ylim([-0.1, 3.0])

    plt.grid()
    sub1.set_aspect("equal")
    plt.savefig(f"plot/{filename}", dpi=200)
    return None


def save_best_params(filename, opt_params, rpa, deg, cost):
    file_path = f"{filename}"
    columns = ['rpa', 'deg', 'omega', 'sigma', 'u0', 'ustar', 'cost']
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        df = pd.DataFrame(columns=columns)
        
    if not df.empty:
        mask = np.isclose(df['rpa'].astype(float), float(rpa), atol=1e-5) & np.isclose(df['deg'].astype(float), float(deg), atol=1e-5)
    else:
        mask = np.array([False])
        
    if mask.any():
        idx = df[mask].index[0]
        if cost < df.at[idx, 'cost']:
            df.loc[idx, ['omega', 'sigma', 'u0', 'ustar', 'cost']] = [
                opt_params[0], opt_params[1], opt_params[2], opt_params[3], cost
            ]
    else:
        new_row = pd.DataFrame([{
            'rpa': rpa, 'deg': deg,
            'omega': opt_params[0], 'sigma': opt_params[1],
            'u0': opt_params[2], 'ustar': opt_params[3],
            'cost': cost
        }])
        df = pd.concat([df, new_row], ignore_index=True) if not df.empty else new_row
        
    df.to_csv(file_path, index=False)

def read_best_params(filename, rpa, deg):
    file_path = f"{filename}"
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        return None
    if df.empty:
        return None
        
    mask = np.isclose(df['rpa'].astype(float), float(rpa), atol=1e-5) & np.isclose(df['deg'].astype(float), float(deg), atol=1e-5)
    
    if mask.any():
        idx = df[mask].index[0]
        saved_cost = df.at[idx, 'cost']
        if saved_cost < 10:
            saved_params = [df.at[idx, 'omega'], df.at[idx, 'sigma'], df.at[idx, 'u0'], df.at[idx, 'ustar']]
            return saved_params
            
    return None