using DifferentialEquations
using BoundaryValueDiffEq
using Plots
using Optim


function ShapeIntegrator!(dy, y, params, t)
    # R,Z,psi,u,v
    omega,sigma,p = params

    R =   y[1]
    Z =   y[2]
    psi = y[3]
    u =   y[4] #psidot
    v =   y[5] #psidotdot

    dy[1] = omega*cos(psi) # R
    dy[2] = - omega*sin(psi) # Z
    dy[3] = omega*u
    dy[4] = omega*v

    v1 = (k * sigma * sin(psi)) / R
    v2 = -k * p
    v3 = k * sigma * u
    v4 = (3 * u^2 * sin(psi)) / (2 * R)
    v5 = (3 * u * cos(psi)^2) / (2 * R^2)
    v6 = -(2 * v * cos(psi)) / R
    v7 = -sin(psi) / (2 * R^3)
    v8 = -sin(psi) * cos(psi)^2 / (2 * R^3)
    v9 = -u / (2 * R^2)
    v10 = -0.5 * u^3
    dy[5] = omega*(v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9 + v10)
end

function bc1!(residual,u,p,t)
    # R,Z,psi,u,v
    residual[1] = u[1][1]    # R = 0
    residual[2] = u[end][1] - Rring
    residual[3] = u[end][1] - u0
    residual[4] = u[end][1] - v0
end



function bc2a!(resid_a, u_a, p, t)
    # for s = 0

    # R,Z,psi,u,v
    resid_a[1] = u_a[1] - 0.0  # R = 0 
end


function bc2b!(resid_b, u_b, p,t)
    # for s = sstar
    # R,Z,psi,u,v
    resid_b[1] = Rring
    resid_b[2] = Z0
    resid_b[3] = u0
    resid_b[4] = v0
end

function init_values(p)
    # R0 is the value at the beginning of the integration
    # L0 is the length scale (in mathematica is R0)

    # omega = sstar / L0

    Rring,omega,sigma,p,f,eps = p

    Z0 = 0
    psi0 = asin(2*f/(sigma*Rring)) - eps
    u0 = - omega*sin(psi0)/Rring #psidot

    # this is the correct adimensionalized version
    v0 = omega^2 * (-((f* (1 / cos(psi0))) / Rring) - k * π * p* Rring * (1 / cos(psi0)) + (cos(psi0) * sin(psi0)) / (2 * Rring^2) + 0.5 * k * sigma* tan(psi0) + tan(psi0) / (2 * Rring^2) - (cos(psi0) * u0) / Rring - 0.5 * tan(psi0) * u0^2)

    init = [Rring,Z0,psi0,u0,v0]
    return init
end

Rring = 20   # multiples of R0 = 20 nm
k = 1        # multiples of k0 = 40 pN nm
f = 0.8        # multiples of f0 = 12.6 pN
sigma = 1    # multiples of sigma0 = 0.05
p = 0.0

omega = 1.0
params = [omega,sigma,p]
eps = -0.01
init = init_values_adim([Rring,omega,sigma,p,f,eps])
Rring,Z0,psi0,u0,v0 = init
println("\n init:",init,"\n")



# General boundary problem

bvp1 = BVProblem(BVPFunction{true}(ShapeIntegrator!, bc1!; bcresid_prototype = zeros(4)),
    init, (0.0,1.0),params, nlls = Val(true))

solver = Shooting(Tsit5(), NewtonRaphson(),
            jac_alg = BVPJacobianAlgorithm(AutoForwardDiff(; chunksize = 2)))

sol1 = solve(bvp1,solver,dt = 0.001,maxiters=1e10)
# sol1 = solve(bvp1,Shooting(Vern7()),dt = 0.01,maxiters=1e10)

# plot(sol1)



# Two point problem definition
# bvp2 = TwoPointBVProblem(ShapeIntegrator!, (bc2a!,bc2b!),init,(0.0, 1.0),params, bcresid_prototype = (zeros(1), zeros(4)))

# sol1 = solve(bvp2,MIRK4(),dt = 0.02,maxiters=1e10)


