import sympy as sp
from sympy import series

# Define symbols
s = sp.Symbol('s', positive=True)
Omega, k, Sigma = sp.symbols('Omega k Sigma', positive=True)
u0 = sp.Symbol('u0', positive=True)

# Define unknown Taylor series coefficients
x1 = Omega
x2 = 0
x3 = -Omega**3*u0**2
x4 = 0

psi1 = Omega * u0
psi2 = 0
psi3, psi4 = sp.symbols('psi3 psi4')

u1 = 0
u2, u3, u4 = sp.symbols('u2 u3 u4')

gamma1 = 2*sp.pi*Omega*Sigma*k
gamma2 = 0
gamma3, gamma4 = sp.symbols('gamma3 gamma4')

x = x1*s + x2/2*s**2 + x3/6*s**3 + x4/24*s**4
psi = psi1*s + psi2/2*s**2 + psi3/6*s**3 + psi4/24*s**4
u = u0 + u1*s + u2/2*s**2 + u3/6*s**3 + u4/24*s**4
gamma = gamma1*s + gamma2/2*s**2 + gamma3/6*s**3 + gamma4/24*s**4

dx_ds = sp.diff(x, s)
dpsi_ds = sp.diff(psi, s)
du_ds = sp.diff(u, s)
dgamma_ds = sp.diff(gamma, s)

# eq3: u' = Omega * [-u/x*cos(psi) + sin(psi)*cos(psi)/x^2 + gamma*sin(psi)/(2*pi*k*x)]
term1 = series(-u/x * sp.cos(psi), s, 0, 4).removeO()
term2 = series(sp.sin(psi)*sp.cos(psi)/x**2, s, 0, 4).removeO()
term3 = series(gamma*sp.sin(psi)/(2*sp.pi*k*x), s, 0, 4).removeO()

eq3 = du_ds - Omega*(term1 + term2 + term3)
s_eq3 = series(eq3, s, 0, 3).removeO()

subs_dict = {}

eq3_1 = s_eq3.coeff(s, 1).subs(subs_dict)
u2_sol = sp.solve(eq3_1, u2)[0]
print(f"u2 = {sp.simplify(u2_sol)}")

