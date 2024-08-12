using DifferentialEquations
using Plots
using Interpolations

function ZCoordinate(paras, psi, s)
    # Extract parameters
    omega, u0, sigma = paras

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

function PlotShapes(sol, best_parameters, rpa, deg; savefig=true)
    # Calculate z coordinates
    z = ZCoordinate(best_parameters, sol.y[1, :], sol.t)

    # Create a plot
    fig = plot(size=(700, 700))
    # Plot the shapes
    plot!(sol.y[4, :], z, linecolor=:blue, label="")
    plot!(-sol.y[4, :], z, linecolor=:red, label="")
    # Calculate y_center based on degree
    if deg > 90
        y_center = z[end] - rpa * sin(deg2rad(deg - 90))
    else
        y_center = z[end] + rpa * sin(deg2rad(90 - deg))
    end

    # Add a circle to the plot
    scatter!(0, y_center, markershape=:circle, markersize=10 * rpa, color=:black, legend=false)
    # Set axis limits and aspect ratio
    xlims!(-2.2, 2.2)
    ylims!(-0.2, 5)
    plot!(aspect_ratio=:equal)
    # Save the figure if requested
    if savefig
        savefig(fig, "plot/wrapped_rpa$(rpa)_phi$(deg).png")
    end
    return nothing
end

function ShapeIntegrator!(dy, y, p, t)
    # psistar, ustar, xstar = p
    omega = y[5]
    sigma = y[6]

    dy[1] = omega * y[2]

    a1 = -y[2] / y[4] * cos(y[1])
    a2 = (sin(y[1]) * cos(y[1])) / y[4]^2
    a3 = (sin(y[1]) * y[3]) / (2π * y[4])
    dy[2] = omega * (a1 + a2 + a3)

    b1 = π * (y[2]^2 - sin(y[1])^2 / y[4]^2)
    b2 = 2π * sigma
    dy[3] = omega * (b1 + b2)

    dy[4] = omega * cos(y[1])
    dy[5] = 0 #omega
    dy[6] = 0 #sigma
end

function bc1!(residual, y, p, t)
    psistar, ustar, xstar = p
    residual[1] = y[end][1] - psistar
    residual[2] = y[end][2] - ustar
    residual[4] = y[end][4] - xstar
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


function InitialValuesExtended(p,s_init)
    omega = p[1]
    u0 = p[2]
    sigma = p[3]
    k = p[4]
    s_init = InitialArcLength((omega, u0))

    return [South_Psi(s_init, omega, u0, sigma),
            South_U(s_init, omega, u0, sigma),
            South_Gamma(s_init, omega, u0, sigma, k),
            South_X(s_init, omega, u0),
            omega,
            sigma
            ]
end

function test()

    # constitutive relations
    Rparticle = 3
    Rvesicle = 30
    rpa = Rparticle / Rvesicle
    # wrapping angle
    phi = pi/6


    # Boundary conditions
    psistar = pi + phi
    ustar = 1/rpa
    xstar = rpa*sin(phi)
    p = [psistar, ustar, xstar]


    # check on xstar value
    if xstar < 0.035
        print("xstar:",xstar)
        throw(DomainError())
    else
        print("xstar:",xstar)

    end

    # free params
    omega0 = 3.0
    sigma0 = 0.019
    u0 = 1.0

    k = 1
    s_init = InitialArcLength((omega0, u0))
    s_span = (s_init, 1.0)

    # init conditions
    p = (omega0, u0, sigma0, k)
    init = InitialValues(p,s_init)
    init = [init; [omega0, sigma0]]


    bvp2 = BVProblem(ShapeIntegrator!, bc1!, InitialValuesExtended(p,s_init), s_span, p,nlls=true)
end

bvp2 = test()
soln = solve(bvp2)