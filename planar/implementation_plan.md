# Implementation Plan: Membrane Engulfment Boundary Value Problem

## 1. Overview and Objective
The goal is to compute the total free energy landscape $F_{tot}(\phi)$ of a planar membrane engulfing a spherical particle. Since the unbound membrane's shape requires solving a Boundary Value Problem (BVP) with an unknown domain size, we will leverage single/double/multiple shooting routines.

## 2. State Vector and ODE Definition
The system will be integrated over a fixed, dimensionless domain $s \in [0, 1]$.
The state vector is $\mathbf{Y}(s) = [\psi(s), u(s), x(s), z(s), F_{me}(s)]^T$.

**Parameters to pass to the ODE function:**
* $\kappa$ (Bending rigidity)
* $\sigma$ (Effective tension)
* $m$ (Spontaneous curvature)
* $\omega$ (Total arc length - dynamically updated during shooting)


## 4. Execution Loop (Energy Landscape)
To generate the energy landscape:
1.  Define an array of wrapping angles $\phi \in [0, \pi]$.
2.  Initialize guesses for $u_0$ and $\omega$.
3.  Loop over $\phi$:
    * Pass the current $\phi$ and guesses to the shooting solver (e.g., `scipy.optimize.root`).
    * Extract $F_{me}^{un} = Y_{final}[4]$.
    * Calculate $F_{me}^{bo}$ and $F_{ad}$ analytically.
    * Sum to find $F_{tot}(\phi)$.
    * **Continuation:** Use the solved `[u0, omega]` from the current step as the initial guess for the next $\phi$ step to speed up convergence.

## 5. Troubleshooting & Convergence Fallback

Because the boundary condition $\psi=0, u=0$ at a finite distance $\omega$ is a mathematical approximation of an asymptotic exponential decay, the Jacobian in the root-finder might become singular, or the solver may fail to converge.
