########################################################################
# Pyhton code for the shape calculation of axissymettric vesicles      #
# written by Felix Frey, TU Delft, 2021-2022, following the paper by   #
# Christ, Simon, et al. "Active shape oscillations of giant vesicles   #
# with cyclic closure and opening of membrane necks."                  #
# Soft Matter 17.2 (2021): 319-330.                                    
########################################################################

from math import gamma
from re import X
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import least_squares


def deg_to_rad(deg):
    return np.pi*deg/180

# integrate 6 ODEs
# S50 from christ
# A4 from frey idema
# 1 pietro's calculation
def DPsi_DS(omega,u):
    return omega*u

# S51
# A5 from frey idema
# 2 from pietro (radius=x)
def DU_DS(omega,psi,u,x,k,gamma):
    a1= - u/x*np.cos(psi)
    a2=(np.sin(psi)*np.cos(psi))/x**2
    a3=(np.sin(psi)*gamma)/(2*np.pi*x*k)
    return omega*(a1+a2+a3)

# S53
# A6 
# 3 from Pietro
def DGamma_DS(omega,u,psi,x,k,sigma):
    a1=np.pi*k*(u**2-np.sin(psi)**2/x**2)
    a2=2*np.pi*sigma*k
    return omega*(a1+a2)

# S49
# A3 from frey idema
# 4 from Pietro
def DX_DS(omega,psi):
    return omega*np.cos(psi)


def South_X(s,omega,u0):    
    x1=omega
    x3=-omega**3*u0**2
    return x1*s+1/6*x3*s**3
# S68 
def South_Psi(s,omega,u0,sigma):
    psi1=omega*u0
    psi3=omega**3*(3*sigma*u0-2*u0**3)
    return psi1*s+1/6*psi3*s**3
# S72
def South_U(s,omega,u0,sigma):
    return u0+1/2*omega**2*0.5*(3*u0*sigma-2*u0**3)*s**2
# S73
def South_Gamma(s,omega,u0,sigma,k):
    gamma1=omega*(2 * np.pi * sigma * k)
    gamma3=4/3 * np.pi * k * u0 *omega**3*(3 * sigma * u0 - 2*u0**3)
    return gamma1*s+1/6*gamma3*s**3


def InitialArcLength(p):
    # this function finds the minimimum length s_init to not have a divergence in x(s).
    omega=p[0]
    u0=p[1]
    threshold=0.035#changed from 0.01 to make the code more stable (because you are more far from evaluating x in s=n*delta_s)
    # threshold = 0.1

    n=1
    delta_s=0.0001
    while(True):
        if South_X(n*delta_s,omega,u0)>threshold:
            break
        else:
            n+=1
    return n*delta_s        



# calculate the initial values at the south pole
def InitialValues(s_init,p):
    # also in felix example the U values is not that close to zero but more close to unity
    
    omega=p[0]
    u0=p[1]
    sigma=p[2]
    k=p[3]

    return [South_Psi(s_init,omega,u0,sigma),
            South_U(s_init,omega,u0,sigma),
            South_Gamma(s_init,omega,u0,sigma,k),
            South_X(s_init,omega,u0),
            ]



# integrate ODEs  
# this is the system of equation to integrate  
def ShapeIntegrator(s,z,omega,sigma):
    psi,u,gamma,x = z #indepedent variables
    k=1
    # return [DPsi_DS(omega,u),
    #        DU_DS(omega,psi,u,x,k,gamma),
    #        DGamma_DS(omega,u,psi,x,k,sigma),
    #        DX_DS(omega,psi),
    #        DA_DS(omega,x),
    #        DV_DS(omega,x,psi)]
    return [DPsi_DS(omega,u),
           DU_DS(omega,psi,u,x,k,gamma),
           DGamma_DS(omega,u,psi,x,k,sigma),
           DX_DS(omega,psi),
           ]

# Jacobian of the shape equations
def ShapeJacobian(s,z,omega,sigma):
    # x,psi,u,gamma,area=z #indepedent variables ma sei proprio un leso
    psi,u,gamma,x = z #indepedent variables

    k=1
    a12 = omega
    a11,a13,a14,a15,a16=0,0,0,0,0
    
    a21 = omega * (u / x * np.sin(psi) + 1 / x**2 * np.cos(psi)**2 - np.sin(psi)**2 / x**2 * gamma / (2 * np.pi * k * x) * np.cos(psi)) # dudpsi
    a22 = -omega/x*np.cos(psi) # dudu
    a23 = omega*np.sin(psi)/(2*k*np.pi*x) # dudgamma
    a24 = omega * (u / x**2 * np.cos(psi) - 2 * np.cos(psi) * np.sin(psi) / x**3 - gamma * np.sin(psi) / (2 * np.pi * k * x**2)) # dudx
    a25,a26=0,0
    
    a31 = omega*(-np.pi*k*np.cos(psi)/x**2) # dgammadpsi
    a32 = omega*np.pi*k*2*u # dgammadu
    a34 = 2*omega*np.pi*k*np.sin(psi)/x**3 # dgammadx
    a33,a35,a36 = 0,0,0
    
    a41 = -omega*np.sin(psi) # dxdpsi
    a42,a43,a44,a45,a46 = 0,0,0,0,0
    
    a54 = omega # dAdx
    a51,a52,a53,a55,a56 = 0,0,0,0,0
    
    a61=3/4*omega*x**2*np.cos(psi) # dVdpsi
    a64 =3/2*omega*x*np.sin(psi) #dVdx 
    a62,a63,a65,a66 = 0,0,0,0

    # return np.array([[a11,a12,a13,a14,a15,a16],
    #                 [a21,a22,a23,a24,a25,a26],
    #                 [a31,a32,a33,a34,a35,a36],
    #                 [a41,a42,a43,a44,a45,a46],
    #                 [a51,a52,a53,a54,a55,a56],
    #                 [a61,a62,a63,a64,a65,a66]])
    
    return np.array([[a11,a12,a13,a14],
                    [a21,a22,a23,a24],
                    [a31,a32,a33,a34],
                    [a41,a42,a43,a44],
                    ])

# calculate residuals
def Residuales(parameters,boundary_conditions):
    k=1
    omega,sigma,u0 = parameters 
    # print(f"free params:{parameters}")
    # calculate initial arc length
    s_init=InitialArcLength((omega,u0))
    
    # calculate initial values
    z_init=InitialValues(s_init,(omega,u0,sigma,k))
    # print(f"approx init values:{z_init}")
    
    # evaluation points for the solution
    s=np.linspace(s_init,omega,1000)

    sol=solve_ivp(ShapeIntegrator, t_span=[s_init,omega], y0=z_init,jac=ShapeJacobian,args=(omega,sigma),t_eval=s,method='Radau') 
    
    # sol=solve_ivp(ShapeIntegrator, t_span=[s_init,omega], y0=z_init,jac=ShapeJacobian,args=(omega,sigma),t_eval=s,method='RK23',max_step=0.001) 
    # print(sol)
    # print(sol.t.shape)
    
    # if sol.status==-1:
    # #     # raise ValueError('error in integration')
    # #     raise Warning('error in integration')
    #     return 'error in integration'
    
    z_fina_num=sol.y[:,-1]
    psif,uf,xf = boundary_conditions
     
    psi=z_fina_num[0]-boundary_conditions[0]
    u=z_fina_num[1]-boundary_conditions[1]  
    # gamma=z_fina_num[2]-boundary_conditions[1]  
    x=z_fina_num[3]-boundary_conditions[2]
    res = psi**2+u**2+x**2
    
    # print(f"Omega: {omega:.5f}  Sigma: {sigma:.5f}  u0: {u0:.5f}  Err:{res:.3g}")
    return [psi,u,x]


def ShapeCalculator(parameters):
    omega,sigma,u0 = parameters 
    k=1

    s_init=InitialArcLength((omega,u0))
    # print(f"s_init:{s_init}")

    z_init=InitialValues(s_init,(omega,u0,sigma,k))
    s=np.linspace(s_init,omega,1000)
    sol=solve_ivp(ShapeIntegrator, t_span=[s_init,omega], y0=z_init,jac=ShapeJacobian,args=(omega,sigma),t_eval=s,method='Radau') 
    return sol

def ZCoordinate(paras,psi,s):
    # to convert s to z
    omega,u0,sigma=paras

    f=InterpolatedUnivariateSpline(s, omega*np.sin(psi), k=1)  # k=1 gives linear interpolation
    z=np.array([])

    for s_max in s:
        z=np.append(z, f.integral(s[0], s_max))
        # print(z)
    return z

def PlotShapes(sol,best_parameters,rpa,deg):
    z=ZCoordinate(best_parameters,sol.y[0],sol.t)
    fig=plt.figure(figsize=(7,7))
    sub1=fig.add_subplot(111)
    sub1.plot(sol.y[3],z,'b-')
    sub1.plot(-sol.y[3],z,'r-')
    
    y_center = z[-1] # this works only for phi=90
    
    if deg > 90:
        y_center = z[-1] - rpa*np.sin(deg_to_rad(deg-90))
        
    else:
        y_center = z[-1] - rpa*np.sin(deg_to_rad(90-deg))


    circle1 = plt.Circle((0, y_center), rpa, color='k',facecolor=None)
    sub1.add_patch(circle1)
        
    # sub1.axis('off')
    plt.grid()
    sub1.set_aspect("equal")
    plt.savefig(f"plot/wrapped_rpa{rpa:.2f}_phi{deg:.2f}.png",dpi=200)
    # plt.show()

    return None

def save_best_params(result,rpa,deg):
    best_parameters = result.x
    with open("params.dat", "a+") as file:
        file.write(f"{rpa :.5f} {deg :.5f}  {best_parameters[0]:.5f} {best_parameters[1]:.5f} {best_parameters[2]:.5f}\n")
    return best_parameters
    
def read_best_params(filename='params.dat'):
    with open(filename, "a+") as file:
        params_dict = {}
        for lines in file:
            a = lines.readline()
            b = a.split(' ')
            params_dict[b[0],b[1]] = [b[2],b[3],b[4]]
    return params_dict
        

def save_coords_file(sol,rpa,deg):
    np.save(f"data/wrapped_rpa{rpa:.2f}_phi{deg:.2f}.npy",sol)


############################################################
############## main function ###############################
############################################################

def main():
    ###### constitutive relations
    Rparticle= 4
    Rvesicle = 30.0
    rpa = Rparticle / Rvesicle
   
    # wrapping angle
    # phi = np.pi/2 
    deg = 60
    phi = deg_to_rad(deg)
        
    ###### free parameters initial values
    omega = 2.41393
    sigma = 0.00987
    u0 = 0.51803

    ###### Boundary conditions
    psistar = np.pi + phi
    ustar = 1/rpa
    xstar=rpa*np.sin(phi)
    
    #### check on xstar value
    if xstar < 0.035:
        print(f"xstar:{xstar}") 
        raise ValueError('xstar too small, can lead to divergences')
    else:
        print(f"xstar:{xstar}") 
        
    boundary_conditions =  [psistar,ustar,xstar]
    # print(f"boundary_conditions:{boundary_conditions}")
    free_params_extended = [omega,sigma,u0]
   
    # shoting algorithm and solver
    result = least_squares(Residuales,free_params_extended,args=([boundary_conditions]),method='lm',verbose=1)
    print(f"Err: {result.cost}")
   
           
    best_parameters = result.x
    sol = ShapeCalculator(best_parameters)
    PlotShapes(sol,best_parameters,rpa,deg)
    
    save_best_params(result,rpa,deg)
    save_coords_file(sol,rpa,deg)
    
# Execute the main function only if the script is run directly
if __name__ == "__main__":
    main()
