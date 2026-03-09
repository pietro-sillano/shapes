# Numerical Shape Integration Project Knowledge
## Problem Context
- The project integrates an axisymmetric vesicle shape ODE system derived from Helfrich energy.
- The physics model describes a colloidal particle binding to and getting wrapped by a membrane, governed by the balance between adhesion energy gain and membrane bending cost.
- The system is a Boundary Value Problem (BVP) with boundaries at the south pole and the north pole (boundary with the wrapped particle).
## Current State & Issues
- **Method**: Currently using a  **multi Shooting** because it is the most stable method.
- **Languages**: last update is in Python. Numba implementation is faster. Use it as default.
## Further Reading
- `protocol_canalejo.md`: Contains MATLAB-based scholarly suggestions.
- `shape_notes/shapes.pdf`: Mathematical derivations in detail.
- `Mathematica_scripts/`: Mathematica notebooks for math validation.
- `tube/`: Preliminary (unstable) code for a related problem regarding membrane tube extraction from a plane.


# Short TODOs
- make a python or bash script that can produce gif from the png

# Open Questions
-

# Long term steps
- generalize the method to extracting a membrane tube \cite{derenyiFormationInteractionMembrane2002}. We already have a (non stable) code for the tube in folder "tube". In principle the tube integration should be easier.
