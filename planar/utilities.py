import numpy as np
import pandas as pd
import os

def save_best_params(file_path, u0, omega, rpa, deg, cost, success, u1=None, psi1=None):
    """
    Save optimized scalar parameters to a CSV file, including far-field boundary values.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    columns = ['rpa', 'deg', 'u0', 'omega', 'psi1', 'u1', 'cost', 'success']
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        df = pd.DataFrame(columns=columns)
        
    if not df.empty:
        mask = np.isclose(df['rpa'].astype(float), float(rpa), atol=1e-5) & \
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
            'rpa': rpa, 'deg': deg,
            'u0': u0, 'omega': omega,
            'psi1': psi1, 'u1': u1,
            'cost': cost,
            'success': success
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        
    df.to_csv(file_path, index=False)

def read_best_params(file_path, rpa, deg, cost_threshold=100):
    """
    Read optimized scalar parameters from a CSV file.
    Returns (u0, omega) if success is True and cost is below threshold, else None.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        return None
        
    if df.empty:
        return None
        
    mask = np.isclose(df['rpa'].astype(float), float(rpa), atol=1e-5) & \
           np.isclose(df['deg'].astype(float), float(deg), atol=1e-5)
    
    if mask.any():
        idx = df[mask].index[0]
        if df.at[idx, 'success'] and df.at[idx, 'cost'] < cost_threshold:
            return df.at[idx, 'u0'], df.at[idx, 'omega']
            
    return None

def save_energies(file_path, rpa, deg, F_me_un, F_me_bo, F_ad, cost):
    """
    Save energy results incrementally to a CSV file.
    Updates existing row if current solution has a lower cost.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    columns = ['rpa', 'phi_deg', 'F_me_un', 'F_me_bo', 'F_ad', 'cost']
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        df = pd.DataFrame(columns=columns)
        
    if not df.empty:
        mask = np.isclose(df['rpa'].astype(float), float(rpa), atol=1e-5) & \
               np.isclose(df['phi_deg'].astype(float), float(deg), atol=1e-5)
    else:
        mask = np.array([False])
        
    if mask.any():
        idx = df[mask].index[0]
        # Only update if cost is better (if cost is available in the CSV)
        # if 'cost' in df.columns:
        #     if cost < df.at[idx, 'cost']:

        # always overwrite energies
        df.loc[idx, ['F_me_un', 'F_me_bo', 'F_ad', 'cost']] = [F_me_un, F_me_bo, F_ad, cost]
        # else:
        #     # If cost column is missing for some reason, just overwrite
        #     df.loc[idx, ['F_me_un', 'F_me_bo', 'F_ad', 'cost']] = [
        #          F_me_un, F_me_bo, F_ad, cost
        #     ]
    else:
        new_row = pd.DataFrame([{
            'rpa': rpa, 'phi_deg': deg,
            'F_me_un': F_me_un,
            'F_me_bo': F_me_bo, 'F_ad': F_ad,
            'cost': cost
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        
    df.to_csv(file_path, index=False)
