import sympy as sp
from sympy import series

# Define symbols
s = sp.Symbol('s', positive=True)
Omega, k, Sigma = sp.symbols('Omega k Sigma', positive=True)
u0 = sp.Symbol('u0', positive=True)

# Define unknown Taylor series coefficients
x2, x3, x4 = sp.symbols('x2 x3 x4')
psi2, psi3, psi4 = sp.symbols('psi2 psi3 psi4')
u1, u2, u3, u4 = sp.symbols('u1 u2 u3 u4')
gamma1, gamma2, gamma3, gamma4 = sp.symbols('gamma1 gamma2 gamma3 gamma4')

# Initial conditions and first terms
psi1 = Omega * u0
x1 = Omega

# Taylor series
x = x1*s + x2/2*s**2 + x3/6*s**3 + x4/24*s**4
psi = psi1*s + psi2/2*s**2 + psi3/6*s**3 + psi4/24*s**4
u = u0 + u1*s + u2/2*s**2 + u3/6*s**3 + u4/24*s**4
gamma = gamma1*s + gamma2/2*s**2 + gamma3/6*s**3 + gamma4/24*s**4

# Derivatives
dx_ds = sp.diff(x, s)
dpsi_ds = sp.diff(psi, s)
du_ds = sp.diff(u, s)
dgamma_ds = sp.diff(gamma, s)

# Equations (to be equal to 0)
eq1 = dx_ds - Omega*sp.cos(psi)
eq2 = dpsi_ds - Omega*u

# For eq3 and eq4, let's multiply by denominator to avoid 1/x divergences in series expansion
# u' = Omega * [-u/x*cos(psi) + sin(psi)*cos(psi)/x^2 + gamma*sin(psi)/(2*pi*k*x)]
# Multiplying by x^2: x^2*u' - Omega*[-u*x*cos(psi) + sin(psi)*cos(psi) + gamma*x*sin(psi)/(2*pi*k)] = 0
eq3 = x**2 * du_ds - Omega*(-u*x*sp.cos(psi) + sp.sin(psi)*sp.cos(psi) + gamma*x*sp.sin(psi)/(2*sp.pi*k))

# gamma' = Omega*[pi*k*(u^2 - sin^2(psi)/x^2) + 2*pi*k*Sigma]
# Multiplying by x^2: x^2*gamma' - Omega*[pi*k*(u^2*x^2 - sin^2(psi)) + 2*pi*k*Sigma*x^2] = 0
eq4 = x**2 * dgamma_ds - Omega*(sp.pi*k*(u**2*x**2 - sp.sin(psi)**2) + 2*sp.pi*k*Sigma*x**2)

# Get series up to s^4 (meaning O(s^5))
s_eq1 = series(eq1, s, 0, 5).removeO()
s_eq2 = series(eq2, s, 0, 5).removeO()
s_eq3 = series(eq3, s, 0, 5).removeO()
s_eq4 = series(eq4, s, 0, 5).removeO()

# Extract coefficients
coeffs = {}
for i in range(5):
    if s_eq1.coeff(s, i) != 0: coeffs[f'eq1_{i}'] = s_eq1.coeff(s, i)
    if s_eq2.coeff(s, i) != 0: coeffs[f'eq2_{i}'] = s_eq2.coeff(s, i)
    if s_eq3.coeff(s, i) != 0: coeffs[f'eq3_{i}'] = s_eq3.coeff(s, i)
    if s_eq4.coeff(s, i) != 0: coeffs[f'eq4_{i}'] = s_eq4.coeff(s, i)

# We can solve iteratively
sol = {}

# From O(1), O(s)...
print("Eq1 coeffs:", [sp.simplify(s_eq1.coeff(s, i)) for i in range(5)])
print("Eq2 coeffs:", [sp.simplify(s_eq2.coeff(s, i)) for i in range(5)])
print("Eq3 coeffs:", [sp.simplify(s_eq3.coeff(s, i)) for i in range(5)])
print("Eq4 coeffs:", [sp.simplify(s_eq4.coeff(s, i)) for i in range(5)])

# Solve systematically
subs_dict = {}

# Eq2 degree 1 -> psi2
eq2_1 = sp.simplify(s_eq2.coeff(s, 1))
psi2_sol = sp.solve(eq2_1, psi2)[0]
subs_dict[psi2] = psi2_sol
print("psi2 =", psi2_sol)

# Eq2 degree 2 -> psi3
eq2_2 = sp.simplify(s_eq2.coeff(s, 2).subs(subs_dict))
psi3_sol = sp.solve(eq2_2, psi3)[0]
subs_dict[psi3] = psi3_sol
print("psi3 =", psi3_sol)

# Eq2 degree 3 -> psi4
eq2_3 = sp.simplify(s_eq2.coeff(s, 3).subs(subs_dict))
psi4_sol = sp.solve(eq2_3, psi4)[0]
subs_dict[psi4] = psi4_sol
print("psi4 =", psi4_sol)

# Eq1 degree 1 -> x2
eq1_1 = sp.simplify(s_eq1.coeff(s, 1).subs(subs_dict))
x2_sol = sp.solve(eq1_1, x2)[0]
subs_dict[x2] = x2_sol
print("x2 =", x2_sol)

# Eq1 degree 2 -> x3
eq1_2 = sp.simplify(s_eq1.coeff(s, 2).subs(subs_dict))
x3_sol = sp.solve(eq1_2, x3)[0]
subs_dict[x3] = x3_sol
print("x3 =", x3_sol)

# Eq1 degree 3 -> x4
eq1_3 = sp.simplify(s_eq1.coeff(s, 3).subs(subs_dict))
x4_sol = sp.solve(eq1_3, x4)[0] # Might depend on u1, let's substitute later if needed
subs_dict[x4] = x4_sol
print("x4 =", x4_sol)

# Now we look at eq3, eq4
# eq3 has degree 2 as lowest non-zero?
print("Eq3 lowest non-zero coefficient at degree 2:")
eq3_2 = sp.simplify(s_eq3.coeff(s, 2).subs(subs_dict))
print(eq3_2) # Gives relation for u1? Actually eq3_2 might be identically 0 if we correctly substitute
if eq3_2 != 0:
    temp_sol = sp.solve(eq3_2, u1)
    if temp_sol:
        u1_sol = temp_sol[0]
        subs_dict[u1] = u1_sol
        print("u1 =", u1_sol)

# eq4 has degree 2 lowest non-zero
print("Eq4 highest non-zero at degree 2:")
eq4_2 = sp.simplify(s_eq4.coeff(s, 2).subs(subs_dict))
print(eq4_2)
if eq4_2 != 0:
    temp_sol = sp.solve(eq4_2, gamma1)
    if temp_sol:
        gamma1_sol = temp_sol[0]
        subs_dict[gamma1] = gamma1_sol
        print("gamma1 =", gamma1_sol)

# Update eq3_3
eq3_3 = sp.simplify(s_eq3.coeff(s, 3).subs(subs_dict))
print("Eq3 degree 3:", eq3_3)
if eq3_3 != 0:
    temp_sol = sp.solve(eq3_3, u2)
    if temp_sol:
        u2_sol = temp_sol[0]
        subs_dict[u2] = u2_sol
        print("u2 =", u2_sol)

# Update eq4_3
eq4_3 = sp.simplify(s_eq4.coeff(s, 3).subs(subs_dict))
print("Eq4 degree 3:", eq4_3)
if eq4_3 != 0:
    temp_sol = sp.solve(eq4_3, gamma2)
    if temp_sol:
        gamma2_sol = temp_sol[0]
        subs_dict[gamma2] = gamma2_sol
        print("gamma2 =", gamma2_sol)

# Update eq3_4
eq3_4 = sp.simplify(s_eq3.coeff(s, 4).subs(subs_dict))
print("Eq3 degree 4:", eq3_4)
if eq3_4 != 0:
    temp_sol = sp.solve(eq3_4, u3)
    if temp_sol:
        u3_sol = temp_sol[0]
        subs_dict[u3] = u3_sol
        print("u3 =", u3_sol)

# Update eq4_4
eq4_4 = sp.simplify(s_eq4.coeff(s, 4).subs(subs_dict))
print("Eq4 degree 4:", eq4_4)
if eq4_4 != 0:
    temp_sol = sp.solve(eq4_4, gamma3)
    if temp_sol:
        gamma3_sol = temp_sol[0]
        subs_dict[gamma3] = gamma3_sol
        print("gamma3 =", gamma3_sol)

print("Final Substitutions:")
for k_sym, v in subs_dict.items():
    print(f"{k_sym} = {v}")

