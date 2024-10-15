using OrdinaryDiffEq
using NonlinearSolve


function ZCoordinate(paras, psi, s)
    # Extract parameters
    omega, u0, sigma = paras
    # print(size(psi))
    # a = sin(psi)
    # print(a)

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
    # psistar, ustar, xstar = p
    (; psistar, ustar, xstar) = p

    init = [0.030671103390489153, u[1], 0.008576555548218276, 0.035216903281918656]  # update the initial conditions
    p′ = merge(p, (omega = u[2], sigma = u[3]))  # update the parameters

    sol = solve(prob; u0 = init, p = p′, save_everystep = false, abstol=1e-10, reltol=1e-10)

    residual[1] = sol[end][1] - psistar
    residual[2] = sol[end][2] - ustar
    residual[3] = sol[end][4] - xstar

    return residual
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

    # init conditions
    k = 1
    p = (; omega = omega0, u0, sigma = sigma0, k, psistar, ustar, xstar)
    init = [0.030671103390489153, 0.8706571792130107, 0.008576555548218276, 0.035216903281918656]

    s_init = InitialArcLength(p)
    s_init = 0.0103
    s_span = (s_init, 1.0)

# successful integration without any NaN values
    ode_prob = ODEProblem(ShapeIntegrator!, init, s_span,p)

# BVP problem using numerical shooting
    nl_prob = NonlinearProblem((res, u, p) -> bc1!(res, ode_prob, u, p), [init[2], omega0, sigma0], p)

    sol =  solve(nl_prob, TrustRegion(; autodiff = AutoFiniteDiff()); show_trace = Val(true))



end

test()