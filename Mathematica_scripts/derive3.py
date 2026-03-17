import sympy as sp

s, Omega, Sigma, k, u0 = sp.symbols('s Omega Sigma k u0', real=True, positive=True)

psi_s = sp.Function('psi')(s)
u_s = sp.Function('u')(s)
gamma_s = sp.Function('gamma')(s)
x_s = sp.Function('x')(s)

# Equations
eq_x = sp.diff(x_s, s) - Omega * sp.cos(psi_s)
eq_psi = sp.diff(psi_s, s) - Omega * u_s
eq_u = x_s**2 * sp.diff(u_s, s) - Omega * (-u_s * x_s * sp.cos(psi_s) + sp.sin(psi_s)*sp.cos(psi_s) + gamma_s * x_s * sp.sin(psi_s) / (2*sp.pi*k))
eq_gamma = x_s**2 * sp.diff(gamma_s, s) - Omega * (sp.pi*k*(u_s**2 * x_s**2 - sp.sin(psi_s)**2) + 2*sp.pi*k*Sigma * x_s**2)

# Sub in series
x_1, x_2, x_3, x_4 = sp.symbols('x_1 x_2 x_3 x_4')
psi_1, psi_2, psi_3, psi_4 = sp.symbols('psi_1 psi_2 psi_3 psi_4')
u_1, u_2, u_3, u_4 = sp.symbols('u_1 u_2 u_3 u_4')
g_1, g_2, g_3, g_4 = sp.symbols('g_1 g_2 g_3 g_4')

x_ser = x_1*s + x_2/2*s**2 + x_3/6*s**3 + x_4/24*s**4
psi_ser = psi_1*s + psi_2/2*s**2 + psi_3/6*s**3 + psi_4/24*s**4
u_ser = u0 + u_1*s + u_2/2*s**2 + u_3/6*s**3 + u_4/24*s**4
g_ser = g_1*s + g_2/2*s**2 + g_3/6*s**3 + g_4/24*s**4

subs_vars = {x_s: x_ser, psi_s: psi_ser, u_s: u_ser, gamma_s: g_ser}

eq1_ser = sp.series(eq_x.subs(subs_vars).doit(), s, 0, 4).removeO()
eq2_ser = sp.series(eq_psi.subs(subs_vars).doit(), s, 0, 4).removeO()
eq3_ser = sp.series(eq_u.subs(subs_vars).doit(), s, 0, 5).removeO()
eq4_ser = sp.series(eq_gamma.subs(subs_vars).doit(), s, 0, 5).removeO()

# Base logic
knowns = {x_1: Omega, psi_1: Omega*u0}

# degree 0 of eq1, eq2 are 0 due to our choices of x_1, psi_1
# degree 1 of eq1, eq2:
eq1_1 = eq1_ser.coeff(s, 1).subs(knowns)
eq2_1 = eq2_ser.coeff(s, 1).subs(knowns)
knowns[x_2] = sp.solve(eq1_1, x_2)[0]
knowns[psi_2] = sp.solve(eq2_1, psi_2)[0]

# degree 2 of eq1, eq2
eq1_2 = eq1_ser.coeff(s, 2).subs(knowns)
eq2_2 = eq2_ser.coeff(s, 2).subs(knowns)
knowns[x_3] = sp.solve(eq1_2, x_3)[0]
knowns[psi_3] = sp.solve(eq2_2, psi_3)[0]

# degree 3 of eq1, eq2
eq1_3 = eq1_ser.coeff(s, 3).subs(knowns)
eq2_3 = eq2_ser.coeff(s, 3).subs(knowns)
knowns[x_4] = sp.solve(eq1_3, x_4)[0]
knowns[psi_4] = sp.solve(eq2_3, psi_4)[0]

# Now for eq 3 and eq 4 which are multiplied by x^2 (so start at degree 2)
eq3_2 = eq3_ser.coeff(s, 2).subs(knowns)
eq4_2 = eq4_ser.coeff(s, 2).subs(knowns)

knowns[u_1] = sp.solve(eq3_2, u_1)[0] if u_1 in eq3_2.free_symbols else 0
knowns[g_1] = sp.solve(eq4_2, g_1)[0]

# Update knowns since x_4, psi_3, psi_4 depend on u_1
for k_sym in list(knowns.keys()):
    knowns[k_sym] = knowns[k_sym].subs(knowns)

eq3_3 = eq3_ser.coeff(s, 3).subs(knowns)
eq4_3 = eq4_ser.coeff(s, 3).subs(knowns)

if u_2 in eq3_3.free_symbols:
    knowns[u_2] = sp.solve(eq3_3, u_2)[0]
if g_2 in eq4_3.free_symbols:
    knowns[g_2] = sp.solve(eq4_3, g_2)[0]

for k_sym in list(knowns.keys()):
    knowns[k_sym] = knowns[k_sym].subs(knowns)
    
eq3_4 = eq3_ser.coeff(s, 4).subs(knowns)
eq4_4 = eq4_ser.coeff(s, 4).subs(knowns)

if u_3 in eq3_4.free_symbols:
    knowns[u_3] = sp.solve(eq3_4, u_3)[0]
if g_3 in eq4_4.free_symbols:
    knowns[g_3] = sp.solve(eq4_4, g_3)[0]

for k_sym in list(knowns.keys()):
    knowns[k_sym] = sp.simplify(knowns[k_sym].subs(knowns))

print("Analytical Taylor Coefficients (up to O(s^3) for functions):")
for k_sym, v in knowns.items():
    print(f"{k_sym} = {v}")

