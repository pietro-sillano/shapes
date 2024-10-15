using DifferentialEquations
using ProgressLogging
using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

# Second order linear test
function f2!(du, u, p, t)
    du[1] = u[2]
    du[2] = -u[1]
end


function boundary!(residual, u, p, t)
    residual[1] = u[1][1] - 5
    residual[2] = u[end][1]
end

odef2! = ODEFunction(f2!,
    analytic = (u0, p, t) -> [
        5 * (cos(t) - cot(5) * sin(t)), 5 * (-cos(t) * cot(5) - sin(t))])

tspan = (0.0, 5.0)
u0 = [5.0, -3.5]

prob = BVProblem(odef2!, boundary!, u0, tspan, nlls = Val(false))
# sol = solve(prob, MIRK4(); dt = 0.2, progress=true)
sol = solve(prob, MIRK4(); dt = 0.001, adaptive=false,progress=true)

# println(sol)
