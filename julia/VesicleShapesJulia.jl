###### same method as python script VesicleShapesPietro.py,solving the BVP using shooting method with a custom optimizer ####
###### look at https://docs.sciml.ai/DiffEqDocs/stable/types/ode_types/#SciMLBase.ODEProblem
##### Using 
using DifferentialEquations
using Plots
using Optim
using Interpolations
using Printf
using QuadGK
using BlackBoxOptim



function ZCoordinate(paras, psi, s)
    omega, u0, sigma = paras

    # Interpolation with linear (order 1) interpolation
    interpolant = LinearInterpolation(s, omega .* sin.(psi), extrapolation_bc = Line())

    # Initialize an array to store the results
    z = []

    # Compute the integral for each s_max in s
    # because dzds=sin(psi)
    for s_max in s
        integral_value = quadgk(x -> interpolant(x), s[1], s_max)[1]
        push!(z, integral_value)
    end

    return z
end

function PlotShapes(sol, best_parameters, rpa, deg; Savefig=true)
    # Calculate z coordinates
    z = ZCoordinate(best_parameters, sol[1,:], sol.t)

    # Create a plot
    fig = plot(size=(700, 700))
    # Plot the shapes
    plot!(sol[4,:], z, linecolor=:blue, label="")
    plot!(-sol[4,:], z, linecolor=:red, label="")
    # Calculate y_center based on degree
    if deg > 90
        y_center = z[end] - rpa * sin(deg2rad(deg - 90))
    else
        y_center = z[end] + rpa * sin(deg2rad(90 - deg))
    end

    # Add a circle to the plot
    # scatter!(0, y_center, markershape=:circle, markersize=10 * rpa, color=:black, legend=false)
    # Set axis limits and aspect ratio
    xlims!(-2.2, 2.2)
    ylims!(-0.2, 3)
    plot!(aspect_ratio=:equal)
    # Save the figure if requested
    if Savefig
        savefig(fig, "wrapped_rpa$(rpa)_phi$(deg).png")
    end
    return fig
end

function ShapeIntegrator!(dy, y, p, t)
    omega,sigma = p    
    k = 1.0
    # dy[1] = DPsi_DS(omega, u)
    dy[1] = omega * y[2]

    a1 = -y[2] / y[4] * cos(y[1])
    a2 = (sin(y[1]) * cos(y[1])) / y[4]^2
    a3 = (sin(y[1]) * y[3]) / (2π * y[4] * k)
    dy[2] = omega * (a1 + a2 + a3)

    b1 = π * k * (y[2]^2 - sin(y[1])^2 / y[4]^2)
    b2 = 2π * sigma * k
    dy[3] = omega * (b1 + b2)

    dy[4] = omega * cos(y[1])
end

# South Pole expansion
function South_X(s, omega, u0)
    x1 = omega
    x3 = -omega^3 * u0^2
    return x1 * s + 1/6 * x3 * s^3
end

function South_Psi(s, omega, u0, sigma)
    psi1 = omega * u0
    psi3 = omega^3 * (3 * sigma * u0 - 2 * u0^3)
    return psi1 * s + 1/6 * psi3 * s^3
end

function South_U(s, omega, u0, sigma)
    return u0 + 1/2 * omega^2 * 0.5 * (3 * u0 * sigma - 2 * u0^3) * s^2
end

function South_Gamma(s, omega, u0, sigma, k)
    gamma1 = omega * (2π * sigma * k)
    gamma3 = 4/3 * π * k * u0 * omega^3 * (3 * sigma * u0 - 2 * u0^3)
    return gamma1 * s + 1/6 * gamma3 * s^3
end
 
function InitialArcLength(p)
    # Find the minimimum length s_init to not have a divergence in x(s).
    omega = p[1]
    u0 = p[2]
    threshold = 0.035  # Changed from 0.01 for better stability
    n = 1
    delta_s = 0.0001
    while true
        if South_X(n * delta_s, omega, u0) > threshold
            break
        end
        n += 1
    end
    return n * delta_s
end

function InitialValues(p,s_init)
    omega = p[1]
    u0 = p[2]
    sigma = p[3]
    k = p[4]

    return [South_Psi(s_init, omega, u0, sigma),
            South_U(s_init, omega, u0, sigma),
            South_Gamma(s_init, omega, u0, sigma, k),
            South_X(s_init, omega, u0),
            ]
end

# Jacobian of the shape equations
function ShapeJacobian!(J,y,p,t)
    # psi, u, gamma, x = z
    k = 1.0
    omega,sigma=p
    # Pre-computed terms for readability
    sin_psi = sin(y[1])
    cos_psi = cos(y[1])
    omega_x = omega / y[4]
    omega_x2 = omega / (y[4]^2)
    omega_x3 = omega / (y[4]^3)
    gamma_kpi = y[3] / (2π * k * y[4])

    # Jacobian elements
    a11, a12, a13, a14, a15, a16 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    a21 = omega * (omega_x * sin_psi + cos_psi^2 / y[4]^2 - sin_psi^2 / y[4]^2 * gamma_kpi * cos_psi)
    a22 = -omega_x * cos_psi
    a23 = omega * sin_psi / (2π * k * y[4])
    a24 = omega * (omega_x * cos_psi - 2 * sin_psi * cos_psi / y[4]^2 - gamma_kpi * sin_psi / (2π * k * y[4]^2))
    a25, a26 = 0.0, 0.0

    a31 = -omega * π * k * cos_psi / y[4]^2
    a32 = omega * π * k * 2*y[2]
    a33, a34 = 0.0, 2 * omega * π * k * sin_psi / y[4]^3
    a35, a36 = 0.0, 0.0

    a41 = -omega * sin_psi
    a42, a43, a44, a45, a46 = 0.0, 0.0, 0.0, 0.0, 0.0

    a54 = omega

    a61 = 3/4 * omega * y[4]^2 * cos_psi
    a64 = 3/2 * omega * y[4] * sin_psi
    a62, a63, a65, a66 = 0.0, 0.0, 0.0, 0.0

    J = [
        [a11, a12, a13, a14],
        [a21, a22, a23, a24],
        [a31, a32, a33, a34],
        [a41, a42, a43, a44],
    ]
    nothing

end

function Onesolution(parameters)
    k = 1
    omega, sigma, u0 = parameters
    s_init = InitialArcLength((omega, u0))
    println("s_init: $s_init")
    z_init = InitialValues((omega, u0, sigma, k),s_init)
    println("z_init: $z_init")

    tspan = (s_init, 1.0)

    p = [omega,sigma]
    prob = ODEProblem(ShapeIntegrator!, z_init, tspan,p) 
    sol = solve(prob, saveat = 0.005)
    return sol
end

function hamiltonian(sol,sigma)
    psi = sol[1,:]
    u = sol[2,:]
    gamma = sol[3,:]
    x = sol[4,:]
    s = sol.t

    H = 0.5*u.^2 .* x + gamma.*cos.(psi)./(2*pi)-x.*sigma-sin.(psi).^2 ./ x
return H
end



function Residuales(parameters, boundary_conditions)
    k = 1
    omega, sigma, u0 = parameters
    # println("free params: $parameters")

    # Calculate initial arc length
    s_init = InitialArcLength((omega, u0))
    # println("free params: $s_init")

    # Calculate initial values
    z_init = InitialValues((omega, u0, sigma, k),s_init)
    # println("approx init values: $z_init")

    # Evaluation points for the solution
    s = range(s_init, stop=1, length=1000)

    # Solve the IVP using Radau method
    f = ODEFunction(ShapeIntegrator!,jac=ShapeJacobian!)
    prob = ODEProblem(f, z_init, (s_init, 1), (omega, sigma),save_everystep = false, abstol=1e-15, reltol=1e-15)
    sol = solve(prob, saveat=s)

    if sol.retcode != :Success
        println("integration failed")
        return [1e3, 1e3, 1e3]
    end

    z_fina_num = sol.u[end]
    psif, uf, xf,rpa,phi = boundary_conditions
    gammaf = -2 * pi * rpa * sigma * tan(phi)

    psi = z_fina_num[1] - psif
    u = z_fina_num[2] - uf
    gamma = z_fina_num[3] - gammaf
    x = z_fina_num[4] - xf
    print(psi,u,gamma,x,"\n")

    # res = norm([psi, u, x])
    # println("Omega: $omega  Sigma: $sigma  u0: $u0  Err: $(round(res, sigdigits=3))")

    return [psi, u, gamma, x]
end


function main(rpa,phi)
    # Free parameters initial values
    omega = 3.0
    sigma = 0.019
    u0 = 1.0

    # Boundary conditions
    psistar = π + phi
    # psistar = phi #no zero possbility that is just phi

    ustar = 1 / rpa
    xstar = rpa * sin(phi)

    # Check on xstar value
    if xstar < 0.035
        println("xstar: $xstar")
        error("xstar too small, can lead to divergences")
    else
        println("xstar: $xstar")
    end

    boundary_conditions = [psistar, ustar, xstar,rpa,phi]
    free_params_extended = [omega, sigma, u0]

    # Shooting algorithm and solver
    # result = optimize(x -> sum(abs2, Residuales(x, boundary_conditions)), free_params_extended, NelderMead())

    result = optimize(x -> sum(abs2, Residuales(x, boundary_conditions)), free_params_extended,)

    @printf "Err: %.5f\n" result.minimum

    best_parameters = result.minimizer
    print(best_parameters)
    sol = Onesolution(best_parameters)

    return sol,best_parameters



    # p = PlotShapes(sol, best_parameters, rpa, deg)
    # display(p)

    # save_best_params(best_parameters, rpa, deg)
    # save_coords_file(sol, rpa, deg)

    # hamiltonian(sol,sigma)

end


function global_min(rpa,phi)
    # Free parameters initial values
    omega = 3.0
    sigma = 0.05
    u0 = 1.0

    # Boundary conditions
    psistar = π + phi
    ustar = 1 / rpa
    xstar = rpa * sin(phi)

    # Check on xstar value
    if xstar < 0.035
        println("xstar: $xstar")
        error("xstar too small, can lead to divergences")
    else
        println("xstar: $xstar")
    end

    boundary_conditions = [psistar, ustar, xstar]
    free_params_extended = [omega, sigma, u0]

    # Shooting algorithm and solver
    # result = optimize(x -> sum(abs2, Residuales(x, boundary_conditions)), free_params_extended, NelderMead())

    res = bboptimize(x -> sum(abs2, Residuales(x, boundary_conditions)); Guess = SearchRange = [(0, pi), (0, 1), (0,1)], NumDimensions = 3)


    @printf "Err: %.5f\n" result.minimum

    best_parameters = result.minimizer
    print(best_parameters)
    sol = Onesolution(best_parameters)
    p = PlotShapes(sol, best_parameters, rpa, deg)
    display(p)

end



function main2(rpa,phi)
# BVP problem using numerical shooting
    omega = 3.0
    sigma = 0.019
    u0 = 1.0

    # Boundary conditions
    psistar = π + phi
    ustar = 1 / rpa
    xstar = rpa * sin(phi)

    # Check on xstar value
    if xstar < 0.035
        println("xstar: $xstar")
        error("xstar too small, can lead to divergences")
    else
        println("xstar: $xstar")
    end

    boundary_conditions = [psistar, ustar, xstar]
    free_params_extended = [omega, sigma, u0]

    # Shooting algorithm and solver
    # Shooting algorithm and solver
    result = optimize(x -> sum(abs2, Residuales(x, boundary_conditions)), free_params_extended, NelderMead())


    nl_prob = NonlinearProblem((res, u, p) -> bc1!(res, ode_prob, u, p), [init[2], omega0, sigma0], p)

    return solve(nl_prob, TrustRegion(; autodiff = AutoFiniteDiff()); show_trace = Val(true))

end

# global_min()

# Constitutive relations
Rparticle = 3
Rvesicle = 30
rpa = Rparticle / Rvesicle

# Wrapping angle
deg = 30
phi = deg2rad(deg)

sol,best_params = main(rpa,phi)

omega, sigma, u0 = best_params
H = hamiltonian(sol,sigma)

fig = plot(size=(500, 500))
plot!(sol.t,sol[1,:] , linecolor=:blue, label="psi")
plot!(sol.t,sol[2,:] , linecolor=:red, label="u")
plot!(sol.t,sol[3,:] , linecolor=:orange, label="gamma")
plot!(sol.t,sol[4,:] , linecolor=:green, label="x")
display(fig)

fig = plot(size=(500, 500))
plot!(sol.t, H, linecolor=:blue, label="")
# plot!(aspect_ratio=:equal)
display(fig)

p = PlotShapes(sol, best_params, 0.1, 60)
display(p)