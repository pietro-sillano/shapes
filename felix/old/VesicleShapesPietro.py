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
    a2=np.sin(psi)*np.cos(psi)/x**2
    a3=np.sin(psi)/(2*np.pi*x*k)*gamma
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

# S56
# 5 from Pietro
def DA_DS(omega,x):
    return omega*x

# S58
# 6 from Pietro
# def DV_DS(omega,x,psi):
    
#     return 3*omega/4*x**2*np.sin(psi)



# taylor expansion south pole because of the instability
# S66   
# A26 equation from frey idema
# all th quantities here are dashed (unitless quantities)

def South_X(s,omega,u0):    
    x1=omega
    x3=-omega**3*u0**2
    return x1*s+1/6*x3*s**3
# S68 
def South_Psi(s,omega,u0,sigma):
    psi1=omega*u0
    psi3=omega**3*(3*sigma*u0-2*u0)
    return psi1*s+1/6*psi3*s**3
# S72
def South_U(s,omega,u0,sigma):
    return u0+1/2*omega**2*0.5*(3*u0*sigma-2*u0**3)*s**2
# S73
def South_Gamma(s,omega,u0,sigma,k):
    gamma1=omega*(2 * np.pi * sigma * k)
    gamma3=4/3 * np.pi * k * u0 *omega**3*(3 * sigma * u0 - 2*u0**3)
    return gamma1*s+1/6*gamma3*s**3
# S77
def South_Area(s,omega,u0):
    A2 = (omega**2)/2
    A4 = -0.5*u0*omega**4
    return A2*s**2/2 + A4*s**4/24 
# # S77    
# def South_Volume(s,omega,u0):
#     return 9/2*u0**2*omega**4*s**4


def North_X(epsilon,omega,u1):
    r1=omega
    r3=-omega**3*u1**2
    return r1*epsilon+1/6*r3*epsilon**3
# S85
def North_Psi(epsilon,omega,u1,sigma):
    psi1=omega*u1
    psi3=omega**3*(3*sigma*u1-2*u1)
    return psi1*epsilon+1/6*psi3*epsilon**3
# S89
def North_U(epsilon,omega,u1,sigma):
    return u1+1/2*omega**2*0.5*(3*u1*sigma-2*u1**3)*epsilon**2

# S90
def North_Gamma(epsilon,omega,u1,sigma,k):    
    gamma1=omega*(2 * np.pi * sigma * k)
    gamma3=4/3 * np.pi * k * u1 *omega**3*(3 * sigma * u1 - 2*u1**3)
    return gamma1*epsilon+1/6*gamma3*epsilon**3

# S94
def North_Area(epsilon,omega,u1,sigma,k):    
    A2 = (omega**2)/2
    A4 = -0.5*u1*omega**4
    return A2*epsilon**2/2 + A4*epsilon**4/24 
    
# # S95
# def North_Volume(epsilon,omega,u1,nu):
#     return nu-1/16*(omega*epsilon)**4*u1

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

def FinalArcLength(p):
    # this function finds the maximum length s_fin to not have a divergence in x(s).
    omega=p[0]
    u1=p[1]
    threshold=0.035#changed from 0.01 to make the code more stable (because you are more far from evaluating x in s=n*delta_s)
    threshold = 0.1
    
    n=1
    delta_s=0.0001
    while(True):
        if North_X(omega - n*delta_s,omega,u1)>threshold:
            break
        else:
            n+=1
    print(omega,n)
    print(North_X(omega - n*delta_s,omega,u1))
    return omega - n*delta_s


# calculate the initial values at the south pole
def InitialValues(s_init,p):
    # also in felix example the U values is not that close to zero but more close to unity
    
    omega=p[0]
    u0=p[1]
    sigma=p[2]
    k=p[3]
    # print(f"k:{k}")
    
    # return [South_Psi(s_init,omega,u0,sigma),
    #         South_U(s_init,omega,u0,sigma),
    #         South_Gamma(s_init,omega,u0,sigma,k),
    #         South_X(s_init,omega,u0),
    #         South_Area(s_init,omega,u0),
    #         South_Volume(s_init,omega,u0)]
    return [South_Psi(s_init,omega,u0,sigma),
            South_U(s_init,omega,u0,sigma),
            South_Gamma(s_init,omega,u0,sigma,k),
            South_X(s_init,omega,u0),
            South_Area(s_init,omega,u0)]



# calculate the final values at the north pole
def FinalValues(s_fin,p):
    omega=p[0]
    u1=p[2]
    sigma=p[2]
    k=p[3]
    epsilon=omega - s_fin
    print(f"eps:{epsilon}") # it should be smaller and close to omega
    return [North_Psi(epsilon,omega,u1,sigma),
            North_U(epsilon,omega,u1,sigma),
            North_Gamma(epsilon,omega,u1,sigma,k),
            North_X(epsilon,omega,u1),
            North_Area(epsilon,omega,u1,sigma,k)]



# integrate ODEs  
# this is the system of equation to integrate  
def ShapeIntegrator(s,z,omega,sigma):
    x,psi,u,gamma,area=z #indepedent variables
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
           DA_DS(omega,x)]

# Jacobian of the shape equations
def ShapeJacobian(s,z,omega,sigma):
    x,psi,u,gamma,area=z #indepedent variables
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
    
    return np.array([[a11,a12,a13,a14,a15],
                    [a21,a22,a23,a24,a25],
                    [a31,a32,a33,a34,a35],
                    [a41,a42,a43,a44,a45],
                    [a51,a52,a53,a54,a55],
                    ])

# calculate residuals
def Residuales(parameters,boundary_conditions):
    k=1
    # parameters for ShapeIntegrator and ShapeJacobian
    # omega,sigma,u0,uf,gamma1 = parameters 
    # omega,sigma,u0,uf = parameters 
    omega,sigma,u0 = parameters 


    # print("omega:",omega,"sigma:",sigma,"u0:",u0,"uf:",uf,"gamma1:",gamma1)

    
    # calculate initial arc length
    s_init=InitialArcLength((omega,u0))
    print(f"s_init:{s_init}")
    
    print("southX:",South_X(s_init,omega,u0))
    
    # calculate final arc length
    # s_fin=FinalArcLength((omega,uf))
    # print(f"s_fin:{s_fin}")

    
    # calculate initial values
    z_init=InitialValues(s_init,(omega,u0,sigma,k))
    print(f"approx init values:{z_init}")
    
    # evaluation points for the solution
    s=np.linspace(s_init,omega,1000)
    # print("shape",s.shape)

    sol=solve_ivp(ShapeIntegrator, t_span=[s_init,omega], y0=z_init,jac=ShapeJacobian,args=(omega,sigma),t_eval=s,method='Radau') 
    # ,first_step=0.00001
    print(sol)
    print(sol.t.shape)
    
    # if sol.status==-1:
    # #     # raise ValueError('error in integration')
    # #     raise Warning('error in integration')
    #     return 'error in integration'
    
    # calculate expansion for xf,uf,gamma1
    z_fina_num=sol.y[0:5,-1]
    
    
    # psif,uf,gamma1,xf,Af=FinalValues(s_fin,(omega,uf,sigma,k))
    # print(f"approx final values:{psif,uf,gamma1,xf,Af}")
    print(f"{boundary_conditions}")
    psif,uf,gamma1,xf,Af = boundary_conditions
    
    # important psif and Af I'll get from the boundary and not from the expanded values(make sense?)
   
    # calculate residuals
    # these one from boundary conditions
    gamma_star = (np.sin(psif)**2+2*sigma*xf-uf**2*xf)/(2*np.cos(psif))
    
    print("omega:",omega,"sigma:",sigma,"u0:",u0,"uf:",uf,"gamma1:",gamma_star)
    
    u=z_fina_num[1]-boundary_conditions[1]
    # gamma=z_fina_num[2]-gamma_star
    
    psi=z_fina_num[0]-boundary_conditions[0]
    x=z_fina_num[3]-boundary_conditions[3]
    # A=z_fina_num[4]-boundary_conditions[4]

    # V=z_fina_num[5]-Vf
    res = psi**2+x**2+u**2
    
    
    # res = psi**2+u**2+gamma**2+x**2+A**2
    print(f"sum_res:{res:3.5g}")
    # print(f"single res:{psi},{u},{gamma},{x},{A}")
    # return [psi,u,gamma,x,A]
    return [psi,x,u]


def ShapeCalculator(parameters):
    omega,sigma,u0,uf,gamma1 = parameters 
    k=1
    
    s_init=InitialArcLength((omega,u0))
    print(f"s_init:{s_init}")

    z_init=InitialValues(s_init,(omega,u0,sigma,k))
    print(' omega',omega,s_init)
    s=np.linspace(s_init,omega,100)
    print("shape",s.shape)
    # args are the arguments for the ShapeIntegrator and ShapeJacobian function
     
    sol=solve_ivp(ShapeIntegrator, t_span=[s_init,omega], y0=z_init,jac=ShapeJacobian,args=(omega,sigma),t_eval=s,method='Radau') 
   
    return sol


def ZCoordinate(paras,psi,s):
    omega,u0,u1,delta_p,sigma=paras

    f=InterpolatedUnivariateSpline(s, omega*np.sin(psi), k=1)  # k=1 gives linear interpolation
    z=np.array([])

    for s_max in s:
        z=np.append(z, f.integral(s[0], s_max))
    return z


#def WriteCoordinatesToFile(nu,s,radius,z):
#    np.savetxt("Coordinates_Red_Vol="+str(nu)[:]+".csv", np.c_[s,radius,z],header="s-coordinate,r-coordinate,z-coordinate", delimiter=',')

def PlotShapes(result,shape_parameters):
    parameters_optimized=result.x
    print(parameters_optimized)
    sol=ShapeCalculator(parameters_optimized)
    radius=sol.y[3,:]
    psi=sol.y[0,:]
    s=sol.t[:]
    
    print('x,psi,s:',radius,psi,s)
   
    #Plot shapes 
    z=ZCoordinate(parameters_optimized,psi,s)
    fig=plt.figure(figsize=(7,7))
    sub1=fig.add_subplot(111)
    
    sub1.plot(radius,z,color="blue",linestyle="--",linewidth=7,label=r'$\nu$: "+str(nu)+", $\bar{H}_0$: "+str(m)+',alpha=0.5)
    sub1.plot(-radius,z,color="red",linestyle="--",linewidth=7,label="numeric",alpha=0.5)
    
    
    # sub1.axis('off')
    sub1.set_aspect("equal")
    # plt.savefig("Red_Vol="+str(nu)[:]+"Red_Pref_Curv="+str(m)[:]+".pdf", bbox_inches = 'tight',pad_inches = 0)
    #WriteCoordinatesToFile(nu,s,radius,z)
    plt.show()
    return None


############################################################
############## main function ###############################
############################################################

def main():
    ###### constitutive relations
    Rparticle= 5
    Rvesicle = 30.0
    rpa = Rparticle / Rvesicle
    A = 4 * np.pi * Rvesicle**2
    # V = 4/3 * np.pi * Rvesicle**3


    
    k = 1.0 # bending rigidity
    W = 0.1 # adhesion strength density J/m^2
    w = W*Rparticle**2/k
    
    
    ###### Independent parameter 
    # https://en.wikipedia.org/wiki/Spherical_cap
    phi = np.pi/6 # wrapping angle
    
    Abo = 2 * np.pi * Rparticle**2 * (1 - np.cos(phi)) #check
    # Vbo = np.pi/3*Rparticle**3*(2+np.cos(phi))*(1-np.cos(phi))**2
    
    # print("A_ve:",A)
    # print(f"A_pa:{Abo}, V_pa:{Vbo}")
    
    
    # protocol for computation
        
    # (0) choose  omega,u0,sigma   
    # (1) initial arc length
    # (2) initial values   
    # (3) integrate from initial arc length to 1-initial arc length   
    # (4) final values
    # (5) calculate residuals
    
    
    
    
    ###### free parameters initial values
    sigma = 1.0
    u0 = 1
    ustar = 1/rpa
    omega = 3

    ###### Boundary conditions
    psistar = np.pi + phi
    
    #### check on xstar value
    xstar=rpa*np.sin(phi)
    if xstar < 0.035:
        print(f"xstar:{xstar}") 
        raise ValueError('xstar too small, can lead to divergences')
    else:
        print(f"xstar:{xstar}") 
        
    gammastar = (np.sin(psistar)**2+2*sigma*xstar-ustar**2*xstar)/(2*np.cos(psistar))

    # print(North_X(omega,))
    Astar=(A-Abo)/(4*np.pi*Rvesicle**2)
    # Vstar =(V - Vbo)/(4/3*np.pi*Rvesicle**3)
    
    # boundary_conditions =  [psistar,ustar,gammastar,xstar,Astar,Vstar] # final values
    boundary_conditions =  [psistar,ustar,gammastar,xstar,Astar] # final values
     
    print(f"boundary_conditions:{boundary_conditions}")
    # free_params_extended = [omega,sigma,u0,ustar,gammastar]
    free_params_extended = [omega,sigma,u0]
   
    # shoting algorithm and solver
    result = least_squares(Residuales,free_params_extended,args=([boundary_conditions]),method='lm',verbose=1)


    # best_parameters = result.x
    # sol = ShapeCalculator(best_parameters)
    # z=ZCoordinate(best_parameters,sol.y[0],sol.t)
    # fig=plt.figure(figsize=(7,7))
    # sub1=fig.add_subplot(111)
    # sub1.plot(sol.y[3],z,'bo-')
    # sub1.plot(-sol.y[3],z,'ro-')
    # plt.show()
   
    # plt.plot(sol.t,sol.y[3],'-o')
    # plt.show()
   
   
# Execute the main function only if the script is run directly
if __name__ == "__main__":
    main()
