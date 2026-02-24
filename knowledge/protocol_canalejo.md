Dear Pietro,

The optimization of these systems is indeed very tricky. Your starting point for (Σ, ΔP, u0​, and s) usually needs to be quite close to the solution, otherwise it will not converge.

The strategy that I always used was to, once I find a solution for some value of the control parameters (reduced volume, spontaneous curvature, wrapping angle, particle-vesicle size ratio), follow it very closely. That is, modify a control parameter only by a very small amount, and use the solution for the previous value of the control parameters as the initial guess for the new value of the control parameters.

If you find two solutions with free parameters (Σ, ΔP, u0​, and s) that are different to each other it could be:
If the solutions are quite different to each other, they could be two different branches (e.g. prolate, oblate, stomatocyte). Note that if your spontaneous curvature is large compared to the inverse vesicle radius you start to have many many branches with different numbers and locations of buds.
If the solutions are very similar to each other, it could be that the problem is quite "flat" around that solution, i.e. changes in the free parameters only lead to small changes in the constraints that need to be satisfied. Most numerical solvers have "flags" that tell you about these potential issues.

In particular, in case it helps, I used MATLAB's fsolve() for the solution of the boundary problem, and ode45() within it for the solution of the ODEs. fsolve() would often give exitflags 2 and 3, which imply this kind of flat problem, where you don't quite get a converged solution. To be sure that what you get is a true solution you need to get exitflag 1. If I would get an exitflag 2 or 3, I would add very small random perturbations to the initial guess (Σ, ΔP, u0​, and s), over and over, until I would get an exitflag 1. This would often work quite well. A telltale sign of the solutions not converging properly for exitflags 2/3  was that, if I would plot e.g. energy as a function of wrapping angle (or some other control parameter), the points with exitflag 1 would form a smooth line, whereas the points with exitflag 2/3 would show small random deviations around this smooth line.

Also, note that convergence gets harder and harder as the neck gets smaller. The fully closed neck cannot be reached (it is a singular limit) but you can get as close as possible, and from an extrapolation one sees that the neck closes at a finite value of the control parameters.

I hope that helps!
