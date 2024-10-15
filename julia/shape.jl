using DifferentialEquations
using BoundaryValueDiffEq
using Plots
using Interpolations
using ProgressLogging
using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

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
    # omega = y[5]
    # sigma = y[6]

    omega, u0, sigma, k, psistar, ustar, xstar = p

    dy[1] = omega * y[2]

    a1 = -y[2] / y[4] * cos(y[1])
    a2 = (sin(y[1]) * cos(y[1])) / y[4]^2
    a3 = (sin(y[1]) * y[3]) / (2π * y[4])
    dy[2] = omega * (a1 + a2 + a3)

    b1 = π * (y[2]^2 - sin(y[1])^2 / y[4]^2)
    b2 = 2π * sigma
    dy[3] = omega * (b1 + b2)

    dy[4] = omega * cos(y[1])
    # dy[5] = 0 #omega
    # dy[6] = 0 #sigma
end

function ShapeIntegratorExtended!(dy, y, p, t)

    # omega, u0, sigma, k, psistar, ustar, xstar = p
    omega = y[5]
    sigma = y[6]
    # print(omega)


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
    # psistar, ustar, xstar = p
    omega, u0, sigma, k, psistar, ustar, xstar = p

    residual[1] = y[end][1] - psistar
    residual[2] = y[end][2] - ustar
    residual[3] = y[end][4] - xstar
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
    omega, u0, sigma, k, psistar, ustar, xstar = p
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
    omega, u0, sigma, k, psistar, ustar, xstar = p
    return [South_Psi(s_init, omega, u0, sigma),
            South_U(s_init, omega, u0, sigma),
            South_Gamma(s_init, omega, u0, sigma, k),
            South_X(s_init, omega, u0),
            ]
end


function InitialValuesExtended(p,s_init)
    omega, u0, sigma, k, psistar, ustar, xstar = p

    s_init = InitialArcLength(p)

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
    p = (omega0, u0, sigma0, k, psistar, ustar, xstar)
    init_cond = [0.030671103390489153, 0.8706571792130107, 0.008576555548218276, 0.035216903281918656]

    s_init = InitialArcLength(p)
    s_init = 0.0103
    s_span = (s_init, 1.0)

    prob = ODEProblem(ShapeIntegrator!, init_cond, s_span,p) 
    sol = solve(prob,dt = 1e-4,progress=true)
    println(sol.u)



    init_cond = [0.030671103390489153, u0, 0.008576555548218276, 0.035216903281918656,omega0,sigma0]

    bvp = BVProblem(BVPFunction{true}(ShapeIntegratorExtended!, bc1!; bcresid_prototype = zeros(3)),init_cond, s_span, p; nlls=Val(true))
    bvp


    # sol1 = solve(bvp,Tsit5(),maxiters=1e10)
    # sol2 = solve(bvp,TsitPap8(),maxiters=1e10)
    # sol3 = solve(bvp,Shooting(Tsit5()),maxiters=1e10)
    print("pre solve")

    
    # integrator = init(bvp, MIRK6(),dt = 0.2)
    # step!(integrator)

    # sol3 = solve(bvp, MIRK4(), dt = 0.2,progress=true)
    sol3 = solve(bvp, MIRK2(); dt = 0.001, adaptive=false,progress=true)

    print(" ")

    # sol1.u[end] - sol2.u[end]
    # omega, u0, sigma = 
    # p = PlotShapes(sol, best_parameters, rpa, deg)
    # display(p)

    # last_sol = 4.195559319788478, 10.060089267372492, 12.721976994624834, 0.09691778305895303, 3.419653288280937, 0.03890024246105399
end



bvp2 = test()
# soln = solve(bvp2,maxiters=1e8)