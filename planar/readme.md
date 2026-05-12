# Content
This project folder contains an implementation of the numerical integration of the Helfrich equation for a particle interacting with a flat membrane.
We use the Monge gauge approximation. We solve the ODE system using a numerical shooting approach.


# Workflow
## Generate solutions
Use `shape_solver.py` to generate solutions. The script has few euristic to find a good solution but few tips to be successfull:
- use small step for phi (like 2,5,10)
- if a certain angle is unstable, try to reach with a decreasing angle (always small step)
- angles around 85-95 are inherently problematic, 90 degree is almost impossible to get it right.


## Label the solutions
I haven't found a way to properly classify solutions so the easiest way is to use `label_monotonicity.py` and to classify the energy landscape. The label can be p: positive monotonous (energy increase with the angle), n negative monotonous (energy decrease with the angle), x non monotonous (intermediate minima, partial wrapping)

## plot the phase plot
With `plot_monotonicity.py` you can plot the phase plot
