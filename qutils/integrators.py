from math import exp
# from numba import njit
from numpy import zeros, full, copy, trapz
# from mpmath import polylog, nstr
from scipy.integrate import solve_ivp


try:
    profile  # Check if the decorator is already defined (when running with memory_profiler)
except NameError:
    def profile(func):  # Define a no-op `profile` decorator if it's not defined
        return func

def myRK4(func,y0,tSpan,paramaters):
    '''This function provides a sovler for any first order system using RK4 fixed time step algorithm.

    Inputs: func - ode function point, y0 - inital conditions in the form of a numpy array (Nx1), tSpan - time span of integration (Mx1 vector of start time and end time), 
    parameters - any parameters passed to the ode function

    Outputs: ode solution in the form of a MxN matrix

    Created: 10/12/21
    Author: Hunter Quebedeaux'''
    numTimeSteps = tSpan.size
    h = tSpan[1] - tSpan[0]
    y = zeros((numTimeSteps,y0.size))
    y[0] = y0
    
    for i in range(1,numTimeSteps):
        k1 = h*func(tSpan[i-1],y[i-1],paramaters)
        k2 = h*func(tSpan[i-1]+0.5*h,y[i-1]+0.5*k1,paramaters)
        k3 = h*func((tSpan[i-1]+0.5*h),(y[i-1]+0.5*k2),paramaters)
        k4 = h*func((tSpan[i-1]+h),(y[i-1]+k3),paramaters)
        y[i]= y[i-1] + (k1+2*k2+2*k3+k4)/6
    return y

def myRK4Py(func,y0,tSpan,paramaters):
    '''This function provides a sovler for any first order system using RK4 fixed time step algorithm using the python frontend.

    Inputs: func - ode function point, y0 - inital conditions in the form of a numpy array (Nx1), tSpan - time span of integration (Mx1 vector of start time and end time), 
    parameters - any parameters passed to the ode function

    Outputs: ode solution in the form of a MxN matrix

    Created: 9/28/22
    Author: Hunter Quebedeaux'''
    numTimeSteps = tSpan.size
    h = tSpan[1] - tSpan[0]
    y = zeros((numTimeSteps,y0.size))
    y[0] = y0
    
    for i in range(1,numTimeSteps):
        k1 = h*func(tSpan[i-1],y[i-1],paramaters)
        k2 = h*func(tSpan[i-1]+0.5*h,y[i-1]+0.5*k1,paramaters)
        k3 = h*func((tSpan[i-1]+0.5*h),(y[i-1]+0.5*k2),paramaters)
        k4 = h*func((tSpan[i-1]+h),(y[i-1]+k3),paramaters)
        y[i]= y[i-1] + (k1+2*k2+2*k3+k4)/6
    return y



def myRK4Step(func,y0,t,h,paramaters):
    '''This function provides a sovler for any first order system using RK4 fixed time step algorithm.

    Inputs: func - ode function point, y0 - inital conditions in the form of a numpy array (Nx1), tSpan - time span of integration (Mx1 vector of start time and end time), 
    parameters - any parameters passed to the ode function

    Outputs: ode solution in the form of a MxN matrix

    Created: 10/12/21
    Author: Hunter Quebedeaux'''
    y = zeros((2,y0.size))
    k1 = h*func(t,y[0],paramaters)
    k2 = h*func(t+0.5*h,y[0]+0.5*k1,paramaters)
    k3 = h*func((t+0.5*h),(y[0]+0.5*k2),paramaters)
    k4 = h*func((t+h),(y[0]+k3),paramaters)
    y[1]= y[0] + (k1+2*k2+2*k3+k4)/6
    return y

def myRK8(func,y0,tSpan,paramaters):
    '''
    This function provides a sovler for any first order system using an 8th order Runge-Kutta solver.

    Inputs: func - ode function point, y0 - inital conditions in the form of a numpy array (Nx1), tSpan - time span of integration (Mx1 vector of start time and end time), 
    parameters - any parameters passed to the ode function

    Outputs: ode solution in the form of a MxN matrix

    Source: https://www.mathworks.com/matlabcentral/fileexchange/55431-runge-kutta-8th-order-integration
    \nCredit to Meysam Mahooti for developing this 8th order solver in MATLAB

    Created: 6/22/22
    Author: Hunter Quebedeaux
    '''
    numTimeSteps = tSpan.size
    h = tSpan[1] - tSpan[0]
    y = zeros((numTimeSteps,y0.size))
    y[0] = y0

    for i in range(1,numTimeSteps):
        k_1 = func(tSpan[i-1]         ,y[i-1]                                                                           ,paramaters)
        k_2 = func(tSpan[i-1]+h*(4/27),y[i-1]+(h*4/27)*k_1                                                              ,paramaters)
        k_3 = func(tSpan[i-1]+h*(2/9) ,y[i-1]+  (h/18)*(k_1+3*k_2)                                                      ,paramaters)
        k_4 = func(tSpan[i-1]+h*(1/3) ,y[i-1]+  (h/12)*(k_1+3*k_3)                                                      ,paramaters)
        k_5 = func(tSpan[i-1]+h*(1/2) ,y[i-1]+   (h/8)*(k_1+3*k_4)                                                      ,paramaters)
        k_6 = func(tSpan[i-1]+h*(2/3) ,y[i-1]+  (h/54)*(13*k_1-27*k_3+42*k_4+8*k_5)                                     ,paramaters)
        k_7 = func(tSpan[i-1]+h*(1/6) ,y[i-1]+(h/4320)*(389*k_1-54*k_3+966*k_4-824*k_5+243*k_6)                         ,paramaters)
        k_8 = func(tSpan[i-1]+h       ,y[i-1]+  (h/20)*(-234*k_1+81*k_3-1164*k_4+656*k_5-122*k_6+800*k_7)               ,paramaters)
        k_9 = func(tSpan[i-1]+h*(5/6) ,y[i-1]+ (h/288)*(-127*k_1+18*k_3-678*k_4+456*k_5-9*k_6+576*k_7+4*k_8)            ,paramaters)
        k_10= func(tSpan[i-1]+h       ,y[i-1]+(h/820)*(1481*k_1-81*k_3+7104*k_4-3376*k_5+72*k_6-5040*k_7-60*k_8+720*k_9),paramaters)
        y[i] = y[i-1] + h/840*(41*k_1+27*k_4+272*k_5+27*k_6+216*k_7+216*k_9+41*k_10)
    return y

@profile
def ode45(fun,tspan,y0,t_eval=None,rtol = 1e-8,atol = 1e-8):
    '''
    wrapper for scipy rk45 function
    input same as solve_ivp but has higher default tolerances
    outputs in the same format as the matlab function, a tuple of t and y reshapes to be more like matlab function
    '''
    solution = solve_ivp(fun,tspan,y0,t_eval=t_eval,rtol = rtol,atol = atol)

    t, y = solution.t.reshape(-1,1), solution.y.T

    return t, y

@profile
def ode85(fun,tspan,y0,t_eval=None,rtol = 1e-8,atol = 1e-8):
    '''
    wrapper for scipy dop853 function - an 8th order solver with 5th order error and adaptive step size
    input same as solve_ivp but has higher default tolerances
    outputs in the same format as the matlab function, a tuple of t and y reshapes to be more like matlab function
    '''
    solution = solve_ivp(fun,tspan,y0,t_eval=t_eval,rtol = rtol,atol = atol,method="DOP853")

    t, y = solution.t.reshape(-1,1), solution.y.T

    return t, y

# import desolver as de

# @profile
# def ode1412(fun,tspan,y0,t_eval=None,rtol=1e-15, atol=1e-15):
#     '''
#     NOT WORKING AS INTENDED -- dense output is not working as intended. does not interpolate between steps even though its goes through the interpolation functions!
#     I suspect that its not working because of how the step size calc happens, instead of doing the interpolation between the steps at the end?
#     cant say for sure...

#     wrapper for desolver integration function - an 14th order solver with 12th order error and adaptive step size
#     outputs in the same format as the matlab function, a tuple of t and y reshapes to be more like matlab function
#     '''
#     if t_eval.any():
#         dt = t_eval[1] - t_eval[0]
#     else:
#         dt = None
#     system = de.OdeSystem(fun,y0=y0,dense_output = True,t=tspan,dt=dt,atol=atol,rtol=rtol)
#     system.method = "RK1412"
#     system.integrate()
#     return system.t, system.y
