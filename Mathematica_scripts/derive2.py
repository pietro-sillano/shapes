import sympy as sp
from sympy import series

# Define symbols
s = sp.Symbol('s', positive=True)
Omega, k, Sigma = sp.symbols('Omega k Sigma', positive=True)
u0 = sp.Symbol('u0', positive=True)

# Define unknown Taylor series coefficients
x1 = Omega
x2, x3, x4 = sp.symbols('x2 x3 x4')
psi1 = Omega * u0
psi2, psi3, psi4 = sp.symbols('psi2 psi3 psi4')
u1, u2, u3, u4 = sp.symbols('u1 u2 u3 u4')
gamma1, gamma2, gamma3, gamma4 = sp.symbols('gamma1 gamma2 gamma3 gamma4')

x = x1*s + x2/2*s**2 + x3/6*s**3 + x4/24*s**4
psi = psi1*s + psi2/2*s**2 + psi3/6*s**3 + psi4/24*s**4
u = u0 + u1*s + u2/2*s**2 + u3/6*s**3 + u4/24*s**4
gamma = gamma1*s + gamma2/2*s**2 + gamma3/6*s**3 + gamma4/24*s**4

dx_ds = sp.diff(x, s)
dpsi_ds = sp.diff(psi, s)
du_ds = sp.diff(u, s)
dgamma_ds = sp.diff(gamma, s)

# Full equations without multiplying by x^2 to capture the O(s) terms properly
# We will use series expansions of the components.
# sin(psi)/x
sin_psi_over_x = series(sp.sin(psi)/x, s, 0, 4).removeO()
# cos(psi)
cos_psi = series(sp.cos(psi), s, 0, 4).removeO()

# eq3: u' = Omega * [-u/x*cos(psi) + sin(psi)*cos(psi)/x^2 + gamma*sin(psi)/(2*pi*k*x)]
term1 = series(-u/x * sp.cos(psi), s, 0, 4).removeO()
term2 = series(sp.sin(psi)*sp.cos(psi)/x**2, s, 0, 4).removeO()
term3 = series(gamma*sp.sin(psi)/(2*sp.pi*k*x), s, 0, 4).removeO()

eq3 = du_ds - Omega*(term1 + term2 + term3)
s_eq3 = series(eq3, s, 0, 3).removeO()

# eq4: gamma' = Omega*[pi*k*(u^2 - sin^2(psi)/x^2) + 2*pi*k*Sigma]
term4 = series(u**2 - (sp.sin(psi)/x)**2, s, 0, 4).removeO()
eq4 = dgamma_ds - Omega*(sp.pi*k*term4 + 2*sp.pi*k*Sigma)
s_eq4 = series(eq4, s, 0, 3).removeO()

# eq1: x' = Omega*cos(psi)
eq1 = dx_ds - Omega*sp.cos(psi)
s_eq1 = series(eq1, s, 0, 4).removeO()

# eq2: psi' = Omega*u
eq2 = dpsi_ds - Omega*u
s_eq2 = series(eq2, s, 0, 4).removeO()

subs_dict = {}

# eq2 deg 0 -> psi2
psi2_sol = sp.solve(s_eq2.coeff(s, 0), psi2)[0]
subs_dict[psi2] = psi2_sol

# eq1 deg 1 -> x2
x2_sol = sp.solve(s_eq1.coeff(s, 1).subs(subs_dict), x2)
subs_dict[x2] = x2_sol[0] if x2_sol else 0 # actually degree 1 is s^1, wait, coeff(s,1)

# eq2 deg 1 -> psi3
psi3_sol = sp.solve(s_eq2.coeff(s, 1).subs(subs_dict), psi3)[0]
subs_dict[psi3] = psi3_sol

# eq1 deg 2 -> x3
x3_sol = sp.solve(s_eq1.coeff(s, 2).subs(subs_dict), x3)[0]
subs_dict[x3] = x3_sol

# eq2 deg 2 -> psi4
psi4_sol = sp.solve(s_eq2.coeff(s, 2).subs(subs_dict), psi4)[0]
subs_dict[psi4] = psi4_sol

# eq1 deg 3 -> x4
x4_sol = sp.solve(s_eq1.coeff(s, 3).subs(subs_dict), x4)[0]
subs_dict[x4] = x4_sol

# eq3 deg 0 -> u1
eq3_0 = s_eq3.coeff(s, 0).subs(subs_dict)
if eq3_0 != 0:
    u1_sol = sp.solve(eq3_0, u1)[0]
    subs_dict[u1] = u1_sol
else:
    subs_dict[u1] = 0

# eq4 deg 0 -> gamma1
eq4_0 = s_eq4.coeff(s, 0).subs(subs_dict)
gamma1_sol = sp.solve(eq4_0, gamma1)[0]
subs_dict[gamma1] = gamma1_sol

# eq3 deg 1 -> u2
eq3_1 = s_eq3.coeff(s, 1).subs(subs_dict)
if eq3_1 != 0:
    u2_sol = sp.solve(eq3_1, u2)[0]
    subs_dict[u2] = u2_sol
else:
    subs_dict[u2] = 0
    
# eq4 deg 1 -> gamma2
eq4_1 = s_eq4.coeff(s, 1).subs(subs_dict)
if eq4_1 != 0:
    gamma2_sol = sp.solve(eq4_1, gamma2)[0]
    subs_dict[gamma2] = gamma2_sol
else:
    subs_dict[gamma2] = 0

# Re-evaluate dependent variables since u1, u2 changes psi3, psi4
subs_dict[psi2] = sp.simplify(psi2_sol.subs(subs_dict))
subs_dict[psi3] = sp.simplify(psi3_sol.subs(subs_dict))
subs_dict[psi4] = sp.simplify(psi4_sol.subs(subs_dict))
subs_dict[x3] = sp.simplify(x3_sol.subs(subs_dict))
subs_dict[x4] = sp.simplify(x4_sol.subs(subs_dict))


print("Correct coefficients:")
for k_sym, v in subs_dict.items():
    print(f"{k_sym} = {sp.simplify(v)}")
    
