using OrdinaryDiffEq
using NonlinearSolve
using Interpolations
using QuadGK
using Plots



function ZCoordinate(p, psi, s)
    (; omega, sigma, u0) = p
    # Interpolation with linear (order 1) interpolation
    interpolant = LinearInterpolation(s, omega .* sin.(psi), extrapolation_bc = Line())

    # Initialize an array to store the results
    z = []

    # Compute the integral for each s_max in s
    for s_max in s
        integral_value = quadgk(x -> interpolant(x), s[1], s_max)[1]
        push!(z, integral_value)
    end

    return z
end

function PlotShapes(sol, p)
    # Calculate z coordinates
    z = ZCoordinate(p, sol[1,:], sol.t)

    # Create a plot
    fig = plot(size=(700, 700))
    # Plot the shapes
    plot!(sol[4,:], z, linecolor=:blue, label="")
    plot!(-sol[4,:], z, linecolor=:red, label="")
    # Calculate y_center based on degree
    # if deg > 90
    #     y_center = z[end] - rpa * sin(deg2rad(deg - 90))
    # else
    #     y_center = z[end] + rpa * sin(deg2rad(90 - deg))
    # end

    # # Add a circle to the plot
    # # scatter!(0, y_center, markershape=:circle, markersize=10 * rpa, color=:black, legend=false)
    # # Set axis limits and aspect ratio
    # xlims!(-2.2, 2.2)
    # ylims!(-0.2, 3)
    plot!(aspect_ratio=:equal)
    # # Save the figure if requested
    # if Savefig
    #     savefig(fig, "wrapped_rpa$(rpa)_phi$(deg).png")
    # end
    return fig
end



function ShapeIntegrator!(dy, y, p, t)
    (; omega, sigma) = p  # use a named tuple to pass parameters

    dy[1] = omega * y[2]

    a1 = -y[2] / y[4] * cos(y[1])
    a2 = (sin(y[1]) * cos(y[1])) / y[4]^2
    a3 = (sin(y[1]) * y[3]) / (2π * y[4])
    dy[2] = omega * (a1 + a2 + a3)

    b1 = π * (y[2]^2 - sin(y[1])^2 / y[4]^2)
    b2 = 2π * sigma
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


function InitialArcLength(p)
    # Find the minimimum length s_init to not have a divergence in x(s).
    (; omega, u0) = p
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

function bc1!(residual, prob, u, p)
    (; psistar, ustar, xstar) = p

    s_init = InitialArcLength(p)
    k = 1
    init_cond = InitialValues((u[2], u[1], u[3], k),s_init)
    p′ = merge(p, (omega = u[2], sigma = u[3])) # update the params value 

    sol = solve(prob; u0 = init_cond, p = p′, save_everystep = false, abstol=1e-10, reltol=1e-10)

    residual[1] = sol[end][1] - psistar
    residual[2] = sol[end][2] - ustar
    residual[3] = sol[end][4] - xstar

    return residual
end

function Onesolution(p)
    (; omega, sigma, u0) = p
    k = 1
    s_init = InitialArcLength(p)
    println("s_init: $s_init")
    z_init = InitialValues((omega, u0, sigma, k),s_init)
    println("z_init: $z_init")
    tspan = (s_init, 1.0)
    prob = ODEProblem(ShapeIntegrator!, z_init, tspan,p) 
    sol = solve(prob, saveat = 0.005)
    return sol
end

function test()

    # constitutive relations
    Rparticle = 3
    Rvesicle = 30
    rpa = Rparticle / Rvesicle
    # wrapping angle
    phi = pi/3


    # Boundary conditions
    psistar = pi + phi
    ustar = 1/rpa
    xstar = rpa*sin(phi)


    # check on xstar value
    if xstar < 0.035
        print("xstar:",xstar,"\n")
        throw(DomainError())
    else
        print("xstar:",xstar,"\n")

    end

    # free params
    omega0 = 3.419653288280937
    sigma0 = 0.03890024246105399
    u0 = 0.8710355880503611

    omega0 = 1.0
    sigma0 = 1.0
    u0 = 1.0

    # init conditions
    k = 1
    p = (; omega = omega0, u0, sigma = sigma0, k, psistar, ustar, xstar)
    # init = [0.030671103390489153, 0.8706571792130107, 0.008576555548218276, 0.035216903281918656]

    init = [1.0,1.0,1.0,1.0]

    s_init = InitialArcLength(p)
    # s_init = 0.0103
    s_span = (s_init, 1.0)

# successful integration without any NaN values
    ode_prob = ODEProblem(ShapeIntegrator!, init, s_span, p)

# BVP problem using numerical shooting
    nl_prob = NonlinearProblem((res, u, p) -> bc1!(res, ode_prob, u, p), [init[2], omega0, sigma0], p)

    bvp2 = solve(nl_prob, TrustRegion(; autodiff = AutoFiniteDiff()); maxiters=100, show_trace = Val(true))

    best_parameters = merge(p, (omega = bvp2.u[2], sigma = bvp2.u[3], u0 = bvp2.u[1])) # update the params value 

    println(p)
    println(best_parameters)

    sol = Onesolution(best_parameters)
    # print(sol.u)
    p = PlotShapes(sol, best_parameters)
    display(p)
end

bvp2 = test()


