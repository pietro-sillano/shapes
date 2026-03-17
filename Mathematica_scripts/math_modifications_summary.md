# Summary of Mathematical Corrections

This document summarizes the mathematical corrections implemented across the vesicle simulation codebase (`vesicle/core_math.py`) and the corresponding theoretical documentation (`shape_notes/shapes.tex`).

## LaTeX Documentation (`shapes.tex`) Updates
1. **Hamiltonian Equation**: Added a missing `1/2` factor to the $-\frac{\sin^2 \psi}{x}$ term.
2. **Contact Angle Boundary Condition**: Corrected the dimensional $R_{pa}$ to the dimensionless $r_{pa}$ (and added overbars to $\bar{\gamma}^\star$ and $\bar{\Sigma}$) to ensure dimensional consistency.
3. **Bound Energy ($E_{bo}$)**: Replaced replacing $4\pi k (1+mR_{pa})^2$ with the dimensionless $4 r_{pa}^2$ form, and then further corrected it to the mathematically exact $4\pi$ (which is the dimensionless bending energy of a spherical cap independent of particle radius).
4. **Unbound Energy ($E_{un}$)**: Filled in the previously blank equation with the explicit Helfrich energy integral formula.
5. **South Pole Expansion ($u(s)$)**: 
   - Removed the ad-hoc $-2ku_0^3$ pressure proxy term to reflect the true zero-pressure analytical limit of the ODE system.
   - Inserted the missing $\Omega^2$ scale factor required by the non-dimensionalization step.
6. **Jacobian Matrix ($a_{21}$)**: Corrected a typo where two terms were being multiplied instead of added ($+$ replaced $\times$).

## Python Codebase (`vesicle/core_math.py`) Updates
1. **Hamiltonian Dimensioning**: Added division by the bending rigidity `k` to the `gamma * np.cos(psi)` term so it correctly evaluates the dimensional energy.
2. **Bound Energy ($E_{bo}$)**: 
   - Fixed a bug where the bending energy was calculated as `4 * rpa**2` (missing $\pi$ and mixing length scales) instead of the exact theoretical `4 * np.pi`. 
   - Renamed `W` to `w` to explicitly denote the dimensionless adhesive strength.
3. **South Pole Expansion**: 
   - Removed an erroneous `0.5 *` multiplier in the $u_2$ Taylor expansion term `South_U` that arbitrarily halved the slope at the boundary.
   - Stripped out the $-2u_0^3$ pressure offset terms from `South_Psi`, `South_U`, and `South_Gamma` to align with the strictly zero-pressure physical model.
4. **Jacobian Matrix ($a_{21}$)**: Mirrored the LaTeX fix by changing a `*` to a `+` between the last two terms in `a21`, correcting the analytical derivative for the solver.

---

## Suggested Git Commit Message

```text
Fix dimensional errors and zero-pressure math limits in shape equations

- core_math.py: Correct E_bo bending energy to exact 4*pi term, rename W to w.
- core_math.py: Remove spurious 0.5 factor in South_U Taylor expansion.
- core_math.py: Strip auxiliary pressure terms (-2*u_0^3) from pole expansions for strict P=0 physics.
- core_math.py/shapes.tex: Fix a21 Jacobian analytical derivative typo (multiplication to addition).
- shapes.tex: Fix missing 1/2 factor in Hamiltonian energy integral.
- shapes.tex: Ensure gamma^* boundary condition uses strict dimensionless symbols (r_pa, \bar{\Sigma}).
- shapes.tex: Document the required \Omega^2 scaling factor in the u(s) pole expansion.
```
