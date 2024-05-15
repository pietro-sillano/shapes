########################################################################
# Pyhton code for the shape calculation of axissymettric vesicles      #
# written by Felix Frey, TU Delft, 2021-2022, following the paper by   #
# Christ, Simon, et al. "Active shape oscillations of giant vesicles   #
# with cyclic closure and opening of membrane necks."                  #
# Soft Matter 17.2 (2021): 319-330.                                    #
########################################################################


import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import least_squares

# integrate 6 ODEs
# S49
# A3 from frey idema
def DRadius_DS(omega,psi):
    return omega*np.cos(psi)
# S50
# A4 from frey idema
def DPsi_DS(omega,u):
    return omega*u
# S51
# A5 from frey idema
def DU_DS(omega,psi,u,radius,delta_p,gamma):
    a1=np.sin(psi)*np.cos(psi)/radius**2
    a2=np.cos(psi)/radius*u
    a3=1/2*delta_p*radius*np.cos(psi)
    a4=np.sin(psi)/radius*gamma
    return omega*(a1-a2-a3+a4)
# S53
# A6 
def DGamma_DS(omega,u,m,psi,radius,delta_p,sigma):
    a1=1/2*(u-2*m)**2
    a2=np.sin(psi)**2/(2*radius**2)
    a3=delta_p*radius*np.sin(psi)
    return omega*(a1-a2-a3+sigma)
# S56
def DA_DS(omega,radius):
    return omega/2*radius
# S58
def DV_DS(omega,radius,psi):
    return 3*omega/4*radius**2*np.sin(psi)


# taylor expansion south pole because of the instability
# S66   
# A26 equation from frey idema
def South_Radius(s,omega,u0):    
    r1=omega
    r3=-omega**3*u0**2
    return r1*s+1/6*r3*s**3
# S68 
def South_Psi(s,omega,m,u0,delta_p,sigma):
    psi1=omega*u0
    psi3=3*omega**3/8*(4*m*u0*(m-u0)-delta_p+2*u0*sigma)
    return psi1*s+1/6*psi3*s**3
# S72
def South_U(s,omega,m,u0,delta_p,sigma):
    psi3=3*omega**3/8*(4*m*u0*(m-u0)-delta_p+2*u0*sigma)
    return u0+1/2*psi3/omega*s**2
# S73
def South_Gamma(s,omega,m,u0,delta_p,sigma):
    gamma1=omega*(2*m*(m-u0)+sigma)
    gamma3=omega**3*(m*u0*(m-u0)*(u0-3*m)-1/4*(9*u0-3*m)*delta_p+1/2*u0*(u0-3*m)*sigma)
    return gamma1*s+1/6*gamma3*s**3
# S77
def South_Area(s,omega):
    return (omega*s)**2/4-(omega*s)**4/48
# S77    
def South_Volume(s,omega):
    return 3*(omega*s)**4/16


# taylor expansion north pole
# S83
# A35 equation from frey idema

def North_Radius(epsilon,omega,u1):
    r1=omega
    r3=-omega**3*u1**2
    return r1*epsilon+1/6*r3*epsilon**3
# S85
def North_Psi(epsilon,omega,m,u1,delta_p,sigma):
    psi1=omega*u1
    psi3=3*omega**3/8*(4*m*u1*(m-u1)-delta_p+2*u1*sigma)
    return (psi1*epsilon+1/6*psi3*epsilon**3)
# S89
def North_U(epsilon,omega,m,u1,delta_p,sigma):
    psi3=3*omega**3/8*(4*m*u1*(m-u1)-delta_p+2*u1*sigma)
    return u1+1/2*psi3/omega*epsilon**2
# S90
def North_Gamma(epsilon,omega,m,u1,delta_p,sigma):    
    gamma1=omega*(2*m*(u1-m)-sigma)
    gamma3=omega**3*(m*u1*(m-u1)*(3*m-u1)+1/4*(9*u1-3*m)*delta_p+1/2*u1*(3*m-u1)*sigma)
    return gamma1*epsilon+1/6*gamma3*epsilon**3
# S94
def North_Area(epsilon,omega,u1):    
    return 1-1/4*(omega*epsilon)**2+(omega*epsilon)**4*u1**2/48
    
# S95
def North_Volume(epsilon,omega,u1,nu):
    return nu-1/16*(omega*epsilon)**4*u1

# find minimal radius
def InitialArcLength(p):
    omega=p[0]
    u0=p[1]
    threshold=0.035#changed from 0.01 to make the code more stable
    n=1
    delta_s=0.0001
    while(True):
        if South_Radius(n*delta_s,omega,u0)>threshold:
            break
        else:
            n+=1
    return n*delta_s        

# calculate the initial values at the south pole
def InitialValues(s_init,p):
    omega=p[0]
    u0=p[1]
    delta_p=p[3]
    sigma=p[4]
    m=p[5]
    
    return [South_Radius(s_init,omega,u0),
            South_Psi(s_init,omega,m,u0,delta_p,sigma),
            South_U(s_init,omega,m,u0,delta_p,sigma),
            South_Gamma(s_init,omega,m,u0,delta_p,sigma),
            South_Area(s_init,omega),
            South_Volume(s_init,omega)]
  
# integrate ODEs  
# this is the system of equation to integrate  
def ShapeIntegrator(s,z,omega,u0,u1,delta_p,sigma,m,nu):
    radius,psi,u,gamma,area,volume=z
    return [DRadius_DS(omega,psi),
           DPsi_DS(omega,u),
           DU_DS(omega,psi,u,radius,delta_p,gamma),
           DGamma_DS(omega,u,m,psi,radius,delta_p,sigma),
           DA_DS(omega,radius),
           DV_DS(omega,radius,psi)]

# Jacobian of the shape equations
def ShapeJacobian(s,z,omega,u0,u1,delta_p,sigma,m,nu):
    radius,psi,u,gamma,area,volume=z       
    a11=-omega*np.sin(radius)
    a12,a13,a14,a15,a16=0,0,0,0,0
    a21,a22,a23,a24,a25,a26=0,0,omega,0,0,0
    a31=omega*(u*np.cos(psi)/radius**2-1/2*np.cos(psi)*delta_p-gamma*np.sin(psi)/radius**2-2*np.cos(psi)*np.sin(psi)/radius**3)
    a32=omega*(gamma*np.cos(psi)/radius+np.cos(psi)**2/radius**2+u*np.sin(psi)/radius+1/2*radius*delta_p*np.sin(psi)-np.sin(psi)**2/radius**2)
    a33=-omega*np.cos(psi)/radius
    a34=omega*np.sin(psi)/radius
    a35,a36=0,0
    a41=omega*(-delta_p*np.sin(psi)+np.sin(psi)**2/radius**3)
    a42=omega*(-radius*np.cos(psi)*delta_p- np.cos(psi)*np.sin(psi)/radius**2)
    a43=omega*(-2*m+ u)
    a44,a45,a46=0,0,0
    a51,a52,a53,a54,a55,a56=omega/2,0,0,0,0,0
    a61,a62,a63,a64,a65,a66=3/2*omega*radius*np.sin(psi),3/4*omega*radius**2*np.cos(psi),0,0,0,0

    return np.array([[a11,a12,a13,a14,a15,a16],
                    [a21,a22,a23,a24,a25,a26],
                    [a31,a32,a33,a34,a35,a36],
                    [a41,a42,a43,a44,a45,a46],
                    [a51,a52,a53,a54,a55,a56],
                    [a61,a62,a63,a64,a65,a66]])

# calculate the final values at the north pole
def FinalValues(s_init,p):
    epsilon=1-(1-s_init)

    omega=p[0]
    u1=p[2]
    delta_p=p[3]
    sigma=p[4]
    m=p[5]
    nu=p[6]
    
    return [North_Radius(epsilon,omega,u1),
    np.pi-North_Psi(epsilon,omega,m,u1,delta_p,sigma),
    North_U(epsilon,omega,m,u1,delta_p,sigma),
    North_Gamma(epsilon,omega,m,u1,delta_p,sigma),
    North_Area(epsilon,omega,u1),
    North_Volume(epsilon,omega,u1,nu)]

# calculate residuals
def Residuales(paras,shape_parameters):
    omega,u0,u1,delta_p,sigma=paras
    m,nu=shape_parameters

    # parameters to be optimized
    parameters=[omega,u0,u1,delta_p,sigma,m,nu]
    # calculate initial arc length
    s_init=InitialArcLength(parameters)
    # calculate initial values
    z_init=InitialValues(s_init,parameters)
    print(f"approx init values:{z_init}")

    # integrate ODEs
    s=np.linspace(s_init,1-s_init,1000)
    sol=solve_ivp(ShapeIntegrator, [s_init,1-s_init], z_init,jac=ShapeJacobian,args=parameters,t_eval=s,method='Radau') 
    z_fina_num=sol.y[0:6,-1]
    # calculate final values
    z_fina=FinalValues(s_init,parameters)
    print(f"approx final values:{z_fina}")

    # calculate residuals
    r1=z_fina_num[0]-z_fina[0]
    r2=z_fina_num[1]-z_fina[1]
    r3=z_fina_num[2]-z_fina[2]
    r4=z_fina_num[3]-z_fina[3]
    r5=z_fina_num[4]-z_fina[4]
    r6=z_fina_num[5]-z_fina[5]
    
    return [r1,r2,r3,r4,r5,r6]

# calculate the final vesicle shape
def ShapeCalculator(paras,shape_parameters):
    omega,u0,u1,delta_p,sigma=paras
    m,nu=shape_parameters

    parameters=[omega,u0,u1,delta_p,sigma,m,nu]
    s_init=InitialArcLength(parameters)
    z_init=InitialValues(s_init,parameters)
    s=np.linspace(s_init,1-s_init,1000)
    sol=solve_ivp(ShapeIntegrator, [s_init,1-s_init], z_init,jac=ShapeJacobian,args=parameters,t_eval=s,method='Radau')    
    return sol

# calculate Z-coordinate
def ZCoordinate(paras,shape_parameters,psi,s):
    omega,u0,u1,delta_p,sigma=paras
    m,nu=shape_parameters

    f=InterpolatedUnivariateSpline(s, omega*np.sin(psi), k=1)  # k=1 gives linear interpolation
    z=np.array([])

    for s_max in s:
        
        z=np.append(z, f.integral(s[0], s_max))
    return z


#def WriteCoordinatesToFile(nu,s,radius,z):
#    np.savetxt("Coordinates_Red_Vol="+str(nu)[:]+".csv", np.c_[s,radius,z],header="s-coordinate,r-coordinate,z-coordinate", delimiter=',')



def PlotShapes(result,shape_parameters,gamma_list,nu_list):
    m=shape_parameters[0]
    nu=shape_parameters[1]
    print("reduced volume: " + str(nu))
    print("reduced preferred curvature: " + str(m))
    print("")
    parameters_optimized=result.x
    sol=ShapeCalculator(parameters_optimized,shape_parameters)
    radius=sol.y[0,:]
    psi=sol.y[1,]
    s=sol.t  
    print("shape",s.shape)
    print(s)
   
    #Plot shapes 
    z=ZCoordinate(parameters_optimized,shape_parameters,psi,s)
    fig=plt.figure(figsize=(7,7))
    sub1=fig.add_subplot(111)
    
    sub1.plot(radius,z,color="blue",linestyle="--",linewidth=7,label=r'$\nu$: "+str(nu)+", $\bar{H}_0$: "+str(m)+',alpha=0.5)
    sub1.plot(-radius,z,color="red",linestyle="--",linewidth=7,label="numeric",alpha=0.5)
    sub1.axis('off')
    sub1.set_aspect("equal")
    plt.savefig("Red_Vol="+str(nu)[:]+"Red_Pref_Curv="+str(m)[:]+".pdf", bbox_inches = 'tight',pad_inches = 0)
    #WriteCoordinatesToFile(nu,s,radius,z)
    return None





########################################################################
# main function
########################################################################

def main():
    # protocol for computation
        
    # (0) choose  omega,m,u0,delta_p,sigma   
    # (1) initial arc length
    # (2) initial values   
    # (3) integrate from initial arc length to 1-initial arc length   
    # (4) final values
    # (5) calculate residuals


    ###### my initial parameters: nu,omega,u0,u1,delta_p,sigma,m ######

    ###### sequential change sequence  ######
    sequence_1=np.array([
    [0.7,4.33,1.43,0.51,-0.66,-.22,1.41],
    [0.7,4.55,1.6,1.3,-1.3,0.15,1.],
    [0.7,4.4,1.79,1.35,-9.5,-2.8,0.5],
    [0.7,4.2,1.85,1.85,-17.8,-6.4,0],
    [0.75,4.1,1.8,1.8,-14.6,-5.4,0],
    [0.8,3.9,1.75,1.75,-12.7,-5.,0],
    [0.85,3.8,1.65,1.65,-12.,-5.05,0],
    [0.9,3.5,1.61,1.19,-11.26,-5.11,0],
    [0.95,3.35,1.4,1.4,-11.4,-5.5,0],
    [1,3.14,1,1,-2,-1,0]])


    ###### parallel change sequence  ######
    sequence_2=np.array([
    [0.7,4.33,1.43,0.51,-0.66,-.22,1.41],
    [0.7,4.4,1.52,1.46,-.4,0.23,1.3],
    [0.7,4.45,1.57,1.5,-1.3,0.,1.1],   
    [0.7,4.55,1.6,1.3,-1.3,0.15,1.],
    [0.725,4.3,1.65,1.27,-1.5,-0.1,.92],
    [0.75,4.,1.7,1.25,-1.8,-.6,0.83],
    [0.8,3.9,1.7,1.25,-5.2,-1.6,0.67],
    [0.85,3.8,1.7,1.22,-7.1,-2.4,0.5],
    [0.9,3.45,1.5,1.16,-9.2,-4.4,0.33],
    [0.95,3.38,1.42,1.13,-10.5,-4.8,0.17],
    [1,3.14,1,1,-2.1,-1.05,0]])





    # this line controls the output shapes:

    para_dict=sequence_2

    # sequence_1: sequential change sequence
    # sequence_2: parallel change sequence




    # shape calculation


    gamma_list=[]
    nu_list=[]

    for index in range(len(para_dict)):
        m=para_dict[index,6]
        nu=para_dict[index,0]
        parameters_init=[para_dict[index,1],para_dict[index,2],
                        para_dict[index,3],para_dict[index,4],
                        para_dict[index,5]]
        
        
        shape_parameters=[m,nu]
        # shoting algorithm and solver
        result = least_squares(Residuales,parameters_init,args=([shape_parameters]),method='lm')
        PlotShapes(result,shape_parameters,gamma_list,nu_list)

        
# main()