---
description: Project Knowledge and Experimental Findings
---

# Numerical Shape Integration Project Knowledge

## Problem Context
- The project integrates an axisymmetric vesicle shape ODE system derived from Helfrich energy.
- The physics model describes a colloidal particle binding to and getting wrapped by a membrane, governed by the balance between adhesion energy gain and membrane bending cost.
- The system is a Boundary Value Problem (BVP) with boundaries at the south pole and the north pole (boundary with the wrapped particle).

## Current State & Issues
- **Method**: Currently using a **Multiple Shooting** approach. The Numba version is the most stable and performant method. (Previous versions used Double Shooting).
- **Languages**: Python (optimized with Numba).

## Testing Methodology
- **Regression Testing**: A testing methodology must ALWAYS be used to check if edits were disruptive. Any changes to integration or equations should be verified by running the tests to ensure numerical stability and correctness are maintained.



## Further Reading
- `protocol_canalejo.md`: Contains MATLAB-based scholarly suggestions.
- `shape_notes/shapes.pdf`: Mathematical derivations in detail.
- `Mathematica_scripts/`: Mathematica notebooks for math validation.
- `tube/`: Preliminary (unstable) code for a related problem regarding membrane tube extraction from a plane.
