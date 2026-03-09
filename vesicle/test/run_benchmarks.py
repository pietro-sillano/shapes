#!/usr/bin/env python3
import os
import sys
import subprocess
import pandas as pd
import numpy as np

def run_benchmark(script_name):
    print(f"Running {script_name}...")
    result = subprocess.run([sys.executable, script_name], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {script_name}:\n{result.stderr}")
        return False
    print(f"Finished {script_name}.")
    return True

def compare_params(test_file, ref_file, tolerance=1e-2):
    if not os.path.exists(test_file):
        print(f"Test params file {test_file} not found.")
        return False
    if not os.path.exists(ref_file):
        print(f"Reference params file {ref_file} not found.")
        return False
        
    df_test = pd.read_csv(test_file)
    df_ref = pd.read_csv(ref_file)
    
    passed = True
    
    for _, test_row in df_test.iterrows():
        rpa = test_row['rpa']
        deg = test_row['deg']
        
        # Find corresponding row in reference
        mask = np.isclose(df_ref['rpa'].astype(float), float(rpa), atol=1e-5) & np.isclose(df_ref['deg'].astype(float), float(deg), atol=1e-5)
        
        if not mask.any():
            print(f"No reference data for rpa={rpa}, deg={deg}")
            continue
            
        ref_row = df_ref[mask].iloc[0]
        
        # Compare parameters
        params = ['omega', 'sigma', 'u0', 'ustar']
        for p in params:
            test_val = test_row[p]
            ref_val = ref_row[p]
            if not np.isclose(test_val, ref_val, rtol=tolerance, atol=tolerance):
                print(f"Mismatch for rpa={rpa}, deg={deg} on parameter {p}: test={test_val:.5f}, ref={ref_val:.5f}")
                passed = False
                
    if passed:
        print("All tested parameters matched reference values within tolerance!")
    else:
        print("Some parameters did not match reference values.")
        
    return passed

def main():
    # Set working directory to the test directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run the benchmarks
    scripts = ['benchmark_multiple_shooting.py', 'benchmark_double_shooting.py']
    for script in scripts:
        if not run_benchmark(script):
            print("Tests failed during execution.")
            sys.exit(1)
            
    # Compare generated parameters with the reference ones (saved in the parent dir)
    original_params = '../params.csv'
    
    print("\n--- Comparing Multiple Shooting parameters with reference ---")
    ms_passed = compare_params('test_params_ms.csv', original_params)
    
    print("\n--- Comparing Double Shooting parameters with reference ---")
    ds_passed = compare_params('test_params_ds.csv', original_params)
    
    if ms_passed and ds_passed:
        print("\nALL TESTS PASSED")
        sys.exit(0)
    else:
        print("\nSOME TESTS FAILED")
        if not ms_passed:
            print("- Multiple Shooting had mismatches or missing data.")
        if not ds_passed:
            print("- Double Shooting had mismatches or missing data.")
        sys.exit(1)

if __name__ == "__main__":
    main()
