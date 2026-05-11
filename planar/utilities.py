import numpy as np
import pandas as pd
import os

def save_best_params(file_path, u0, omega, rpa, deg, cost, success, sigma, u1=None, psi1=None):
    """
    Save optimized scalar parameters to a CSV file, including far-field boundary values.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    columns = ['sigma', 'rpa', 'deg', 'u0', 'omega', 'psi1', 'u1', 'cost', 'success']

    try:
        df = pd.read_csv(file_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        df = pd.DataFrame(columns=columns)

    if not df.empty:
        mask = np.isclose(df['sigma'].astype(float), float(sigma), rtol=1e-6) & \
               np.isclose(df['rpa'].astype(float), float(rpa), atol=1e-5) & \
               np.isclose(df['deg'].astype(float), float(deg), atol=1e-5)
    else:
        mask = np.array([False])

    if mask.any():
        idx = df[mask].index[0]
        # Overwrite if success is True and cost is better, or if previously failed
        if (success and cost < df.at[idx, 'cost']) or not df.at[idx, 'success']:
            df.at[idx, 'u0'] = u0
            df.at[idx, 'omega'] = omega
            df.at[idx, 'psi1'] = psi1
            df.at[idx, 'u1'] = u1
            df.at[idx, 'cost'] = cost
            df.at[idx, 'success'] = success
    else:
        new_row = pd.DataFrame([{
            'sigma': sigma, 'rpa': rpa, 'deg': deg,
            'u0': u0, 'omega': omega,
            'psi1': psi1, 'u1': u1,
            'cost': cost,
            'success': success
        }])
        df = pd.concat([df, new_row], ignore_index=True)

    df.to_csv(file_path, index=False)

def read_closest_params(file_path, rpa, sigma, deg, cost_threshold=100):
    """
    Find the saved entry with the closest phi (deg) for the same sigma and rpa.
    Returns (u0, omega) if a match with cost < cost_threshold exists, else None.
    """
    try:
        df = pd.read_csv(file_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return None

    if df.empty:
        return None

    mask = np.isclose(df['sigma'].astype(float), float(sigma), rtol=1e-6) & \
           np.isclose(df['rpa'].astype(float), float(rpa), atol=1e-5) & \
           df['success'].astype(bool) & \
           (df['cost'].astype(float) < cost_threshold) & \
           (df['deg'].astype(float) != 0 ) & \
           (df['deg'].astype(float) != 180 )


    candidates = df[mask]
    if candidates.empty:
        return None

    closest_idx = (candidates['deg'].astype(float) - float(deg)).abs().idxmin()
    row = candidates.loc[closest_idx]

    return row['rpa'],row['deg'],row['cost'],row['u0'], row['omega']


def read_best_params(file_path, rpa, deg, sigma, cost_threshold=100):
    """
    Read optimized scalar parameters from a CSV file.
    Returns (u0, omega) if success is True and cost is below threshold, else None.
    """
    try:
        df = pd.read_csv(file_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return None

    if df.empty:
        return None

    mask = np.isclose(df['sigma'].astype(float), float(sigma), rtol=1e-6) & \
           np.isclose(df['rpa'].astype(float), float(rpa), atol=1e-5) & \
           np.isclose(df['deg'].astype(float), float(deg), atol=1e-5)

    if mask.any():
        idx = df[mask].index[0]
        if df.at[idx, 'success'] and df.at[idx, 'cost'] < cost_threshold:
            return df.at[idx, 'u0'], df.at[idx, 'omega']

    return None

def save_geometry(file_path, rpa, deg, z_center, cost, sigma):
    """
    Save particle centre height z_center (dimensional) to a CSV file.
    z_center > 0  → centre is below the far-field membrane plane (engulfed side).
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    columns = ['sigma', 'rpa', 'phi_deg', 'z_center', 'cost']

    try:
        df = pd.read_csv(file_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        df = pd.DataFrame(columns=columns)

    if not df.empty:
        mask = np.isclose(df['sigma'].astype(float), float(sigma), rtol=1e-6) & \
               np.isclose(df['rpa'].astype(float), float(rpa), atol=1e-5) & \
               np.isclose(df['phi_deg'].astype(float), float(deg), atol=1e-5)
    else:
        mask = np.array([False])

    if mask.any():
        idx = df[mask].index[0]
        if cost < float(df.at[idx, 'cost']):
            df.loc[idx, ['z_center', 'cost']] = [z_center, cost]
    else:
        new_row = pd.DataFrame([{
            'sigma': sigma, 'rpa': rpa, 'phi_deg': deg,
            'z_center': z_center, 'cost': cost
        }])
        df = pd.concat([df, new_row], ignore_index=True)

    df.to_csv(file_path, index=False)


def save_energies(file_path, rpa, deg, F_me_un, F_me_bo, F_ad, cost, sigma):
    """
    Save energy results incrementally to a CSV file.
    Updates existing row if current solution has a lower cost.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    columns = ['sigma', 'rpa', 'phi_deg', 'F_me_un', 'F_me_bo', 'F_ad', 'cost']

    try:
        df = pd.read_csv(file_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        df = pd.DataFrame(columns=columns)

    if not df.empty:
        mask = np.isclose(df['sigma'].astype(float), float(sigma), rtol=1e-6) & \
               np.isclose(df['rpa'].astype(float), float(rpa), atol=1e-5) & \
               np.isclose(df['phi_deg'].astype(float), float(deg), atol=1e-5)
    else:
        mask = np.array([False])

    if mask.any():
        idx = df[mask].index[0]
        df.loc[idx, ['F_me_un', 'F_me_bo', 'F_ad', 'cost']] = [F_me_un, F_me_bo, F_ad, cost]
    else:
        new_row = pd.DataFrame([{
            'sigma': sigma, 'rpa': rpa, 'phi_deg': deg,
            'F_me_un': F_me_un,
            'F_me_bo': F_me_bo, 'F_ad': F_ad,
            'cost': cost
        }])
        df = pd.concat([df, new_row], ignore_index=True)

    df.to_csv(file_path, index=False)
