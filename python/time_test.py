import time
import os

print("--- Original ---")
t0 = time.time()
from VesicleShapesPietrov2 import main as main_py
main_py()
t1 = time.time()
print(f"Original Py Run: {t1-t0:.2f}s")
print("==================\n")

print("--- Numba ---")
t2 = time.time()
from VesicleShapesPietrov2_numba import main as main_numba
main_numba()
t3 = time.time()
main_numba()
t4 = time.time()
print(f"Numba First run (compilation): {t3-t2:.2f}s")
print(f"Numba Second run (cached): {t4-t3:.2f}s")
