using Plots
using SpecialFunctions  # For the modified Bessel function of the second kind

# Define the constants


sigma = 1.0
p = 0.000001
Z0 = 6.0 
R0 = 1.0 
f = 1.0  
f0 = 1.0 
R_ves = 2*sigma/p

# Define the function Z_lin(R)
function Z_lin(R)
    term1 = Z0
    term2 = (2 * R0 * f / f0) * (log(R / (sqrt(2) * R0)) + besselk(0, R / (sqrt(2) * R0)))
    term3 = (R^2) / (2 * R_ves)
    return term1 - term2 - term3
end

R_values = range(0.1, stop=30, length=400)  # Avoid R = 0 to prevent log(0)
Z_values = Z_lin.(R_values)

# Plot the function
plot(Z_values,R_values, xlabel="Z", ylabel="R", title="R vs Zlin", grid=true)
plot!(aspect_ratio=:equal)

