# Content
This project folder contains an implementation of the numerical integration of the Helfrich equation for a particle interacting with a flat membrane.
We use the Monge gauge approximation. We solve the ODE system using a numerical shooting approach.



# Doubts
- Should I have a sigma surface tension area also for the energy of the bound particle part? Should the total tension term be constant right? if the area is constant in theory yes.
-


# TODO list
- check how W is adimensionalized
- rewrite the code to allow for different tension as parameter and save the results accordingly




# changing sigma
See in diff folder the diff.png or diff.csv
- lower sigma is less unbound bending energy (and bound bending energy too)
- higher sigma means more energy to bend the membrane
- i am not sure if it makes sense to compare the sigma obtained from the vesicle.


# energy scale
- also the W is very different now, probably I should scale the radius of the particle accordingly. probably a good value can be scaled by rescaled length, depending from k and sigma. lambda = k / sigma


- in vesicle I have used the omega to rescale the energy. but should I consider also the rescaling of the aprticle respect to the vesicle size.
