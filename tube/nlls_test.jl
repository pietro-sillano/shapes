using DifferentialEquations
using BoundaryValueDiffEq

f1(u, p, t) = [u[2], -u[1]]

function bc1(sol, p, t)
    t₁, t₂ = extrema(t)
    solₜ₁ = sol(t₁)
    solₜ₂ = sol(t₂)
    solₜ₃ = sol((t₁ + t₂) / 2)
    # We know that this overconstrained system has a solution
    return [solₜ₁[1], solₜ₂[1] - 1, solₜ₃[1] - 0.51735, solₜ₃[2] + 1.92533]
end

tspan = (0.0, 100.0)
u0 = [0.0, 1.0]

bvp1 = BVProblem(BVPFunction{false}(f1, bc1; bcresid_prototype = zeros(4)),
    u0, tspan; nlls = Val(true))

solver = Shooting(Tsit5(), NewtonRaphson(),
            jac_alg = BVPJacobianAlgorithm(AutoForwardDiff(; chunksize = 2)))

solver = Shooting(Tsit5())
solver = Shooting(Tsit5(), NewtonRaphson())

sol1 = solve(bvp1,solver,dt =0.001,verbose = true, abstol = 1e-6, reltol = 1e-6)


