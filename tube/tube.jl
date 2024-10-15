using DifferentialEquations
using Plots
using Optim
using LSODA


function ShapeIntegratorAdim!(dy, y, params, t)
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

function ShapeIntegrator!(dy, y, p, t)
    # R,Z,psi,u,v
    sigmabar,pbar = p

    R =   y[1]
    Z =   y[2]
    psi = y[3]
    u =   y[4] #psidot
    v =   y[5] #psidotdot

    dy[1] = cos(psi) # R
    dy[2] = sin(psi) # Z
    dy[3] = u
    dy[4] = v
    dy[5] = -0.5*u^3 - (2*cos(psi)/R)*v + 3*sin(psi)/(2*R)*u^2 + (3*cos(psi)^2-1)/(2*R^2)*v + sigmabar*v-(cos(psi)^2+1)/(2*R^3)*sin(psi) + sigmabar/R*sin(psi)-pbar
end

function init_values(p)
    Rring,sigma,p,f,eps = p

    R0 = Rring
    Z0 = 0
    psi0 = asin(f/(2*pi*sigma*Rring)) - eps
    u0 = - sin(psi0)/R0 #psidot
    
    # this is not adimensionalized
    v0 = 1/cos(psi0)*(-0.5*u0^2*sin(psi0) - cos(psi0)^2/R0*u0+(cos(psi0)^2+1)/(2*R0^2)*sin(psi0)+sigma/k*sin(psi0)-0.5*p/k*R0-f/(2*pi*R0)) #psidotdot, from eq 6
    init = [R0,Z0,psi0,u0,v0]
    return init
end

function init_values_adim(p)
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

function Residuales(free_params,params)
    omega,eps = free_params
    sigma,p = params

    init = init_values_adim([Rring,omega,sigma,p,f,eps])
    # println("\n init:",init,"\n")


    extended_params = [omega,sigma,p]
    prob = ODEProblem(ShapeIntegratorAdim!, init,(0.0, - 1.0),extended_params)

    sol = solve(prob, saveat = 0.005)
    z_fina_num = sol.u[end]
    # R,Z,psi,u,v

    R = z_fina_num[1]
    psi = z_fina_num[3]

    println(R,"  ", psi,"  ",omega,"  ",eps)

    return [R]
end


function ODEsolution(init,free_params)
    omega,eps = free_params

    init = init_values([Rring,sigma,p,f,eps])
    
    prob = ODEProblem(ShapeIntegratorAdim!, init,(0.0,-10.0),[omega,sigma,p])
    sol = solve(prob, saveat = 0.01)

    fig = plot(size=(500, 500))
    plot!(sol[2,:] ,sol[1,:],xlabel="Z", ylabel="R",)
    display(fig)
end


function solution_low_force(eps,f)


    # not non-dimensionalized
    k = 1
    sigma = 0.5
    p = 0

    Rring = 20
    R0 = Rring
    Z0 = 0
    psi0 = asin(f/(2*pi*sigma*Rring)) - eps
    u0 = - sin(psi0)/R0 #psidot

    v0 = 1/cos(psi0)*(-0.5*u0^2*sin(psi0) - cos(psi0)^2/R0*u0+(cos(psi0)^2+1)/(2*R0^2)*sin(psi0)+sigma/k*sin(psi0)-0.5*p/k*R0-f/(2*pi*R0)) #psidotdot, from eq 6
    init = [R0,Z0,psi0,u0,v0]

    prob = ODEProblem(ShapeIntegrator!, init,(0.1,-100.0),[sigma,p])
    # sol = solve(prob,AutoTsit5(Rosenbrock23()),abstol=1e-15, reltol=1e-15, saveat = 0.01)

    # sol = solve(prob,AutoTsit5(Rodas5()),abstol=1e-15, reltol=1e-15, saveat = 0.01)


    # sol = solve(prob,AutoVern9(Rodas5()),abstol=1e-15, reltol=1e-15, saveat = 0.01,maxiters=100000)

    sol = solve(prob,RadauIIA7(),abstol=1e-15, reltol=1e-15, saveat = 0.01,maxiters=1000000)

    
    # fig = plot(size=(500, 500))
    # plot!(sol.t ,sol[1,:],xlabel="s", label="R")
    # display(fig)
    # fig = plot(size=(500, 500))
    # # plot!(sol.t ,sol[3,:],xlabel="s", label="psi",)
    # plot!(sol[2,:] ,sol[1,:],xlabel="Z", ylabel="R",label="curve")
    # display(fig)

    return sol

end


fig = plot(size=(500, 500))

eps_list = []
f_list = [0.01,0.02,0.05,0.1,0.2,0.5,0.75]
eps_list = [0.0005]

for eps in eps_list
    for f in f_list
        sol = solution_low_force(eps,f)
        plot!(sol[2,:] ,sol[1,:],xlabel="Z", ylabel="R",label="f = $f eps = $eps")

        # plot!(sol.t ,sol[1,:],xlabel="s", ylabel="R",label="f = $f eps = $eps")

        # plot!(sol.t ,sol[3,:],xlabel="s", ylabel="psi",label="f = $f eps = $eps")

        display(fig)

    end
end


# Rring = 20   # multiples of R0 = 20 nm
# k = 1       # multiples of k0 = 40 pN nm
# f = 0.0001        # multiples of f0 = 12.6 pN
# sigma = 0.5    # multiples of sigma0 = 0.05
# p = 0.0

# omega = 50
# params = [omega,sigma,p]
# eps = 0.001
# init = init_values([Rring,sigma,p,f,eps])
# println("\n init:",init,"\n")

# # R,Z,psi,u,v
# ODEsolution(init,[omega,eps])


# # # Shooting algorithm and solver

# params = [omega,eps]
# result = optimize(x -> sum(abs2, Residuales(x, params)),[sigma,p])
# init = init_values([Rring,sigma,p,f,eps])

# omega,eps = result.minimizer
# ODEsolution(init,[omega,eps])

