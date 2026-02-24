# Curvature Jump at the Contact Point

In theoretical modeling of vesicles partially wrapping a rigid particle (like during endocytosis), the **curvature of the free membrane at the contact point ($u_{contact}$)** is treated as a free parameter, rather than being fixed to the curvature of the particle ($1/R_{particle}$).

This is physically and mathematically necessary for two main reasons:

## 1. The Physics (Curvature Jump at the Contact Line)
In the classic Helfrich membrane equilibrium model, when a membrane adheres to and wraps around a rigid sphere, there is generally a **jump in the mean curvature** across the detachment contact line.
*   **On the sphere (bound membrane):** The membrane curvature is forcibly matched to $1/R_{particle}$.
*   **Off the sphere (free membrane):** The membrane curvature $u_{contact}$ is free to relax to minimize the overall bending energy, adapting to how strongly the membrane is being pulled or tensioned.

The mathematical difference between these two curvatures is directly proportional to the adhesion energy $W$ per unit area that binds the membrane to the particle (refer to works like Deserno 2004). Because our code specifically restricts the total free **Area** available (searching for the unique shape that accommodates a predetermined partial volume/wrapping degree $\phi$), the implicit adhesion energy adjusts to whatever tension value establishes that wrapping state. Consequently, $u_{contact}$ must be unconstrained to find this equilibrium!

## 2. The Math (Degrees of Freedom in Double Shooting)
To perform numerical optimization using the Double Shooting method, we require the two independent integrated trajectories (from the South Pole up, and from the Contact Point down) to cleanly intersect and stitch together at the equator midpoint.

At this integration midpoint, we must satisfy exactly **4 matching conditions**:
1. Continuity of the tangent angle ($\psi$)
2. Continuity of the radius ($x$)
3. Continuity of the curvature ($u$)
4. The Unbound Area integrating exactly to the correct target ($A_{star}$)

To simultaneously solve 4 strict mathematical equations, the optimizer absolutely needs exactly **4 independent free parameters**:
*   From the **South Pole**, we have 3 structural parameters: `omega` (scale length), `sigma` (surface tension), and `u0` (south pole tip curvature).
*   From the **Contact point**, we fix $x^*$ and $\psi^*$ purely from static geometry matching the particle's side, and we calculate $\gamma^*$ analytically from the Hamiltonian conservation ($H=0$).
*   This leaves us needing precisely **1 more parameter** from the Contact side to solve the $4 \times 4$ system.

If we artificially locked $u_{contact} \equiv 1/R_{particle}$, we would only have 3 active parameters to satisfy 4 independent boundary conditions. The ODE solver would be mathematically over-constrained and broadly incapable of discovering continuous solutions that fit targeted vesicle Area requirements.
