from math import exp
# from numba import njit
from numpy import zeros, full, copy, trapz
# from mpmath import polylog, nstr

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




def myRK5(func,y0,tSpan,paramaters):
    '''This function provides a sovler for any first order system using a 5th order Runge-Kutta-Fehlberg solver. Currently not working

    Inputs: func - ode function point, y0 - inital conditions in the form of a numpy array (Nx1), tSpan - time span of integration (Mx1 vector of start time and end time), 
    parameters - any parameters passed to the ode function

    Outputs: ode solution in the form of a MxN matrix

    Source: Applied Numerical Methods for Engineers and Scientists by S. Rao

    Created: 5/14/22
    Author: Hunter Quebedeaux'''

    numTimeSteps = tSpan.size
    h = tSpan[1] - tSpan[0]
    y = zeros((numTimeSteps,y0.size))
    y[0] = y0

    w1 = (16/135)        
    w2 = (6656/12825)   
    w3 = (28561/56430)     
    w4 = (9/50)    
    w5 = (2/55)  

    for i in range(1,numTimeSteps):
        k1 = h*func(tSpan[i-1],y[i-1],paramaters)
        k2 = h*func(tSpan[i-1]+0.25*h,y[i-1]+0.25*h*k1,paramaters)
        k3 = h*func((tSpan[i-1]+(3/8)*h),(y[i-1]+(3/32)*h*k1+(9/32)*h*k2),paramaters)
        k4 = h*func((tSpan[i-1]+(12/13)*h),(y[i-1]+(1932/2197)*h*k1-(7200/2197)*h*k2+(7296/2197)*h*k3),paramaters)
        k5 = h*func((tSpan[i-1]+h),(y[i-1]+(439/216)*h*k1-(8)*h*k2+(3680/513)*h*k3-(845/4104)*h*k4),paramaters)
        k6 = h*func((tSpan[i-1]+0.5*h),(y[i-1]-(8/27)*h*k1+(2)*h*k2+(3544/2565)*h*k3+(1859/4104)*h*k4-(11/40)*h*k5),paramaters)
        y[i]= y[i-1] + ((w1 * k1 )+ (w2 * k3) + (w3 * k4) - (w4 * k5) + (w5 * k6))
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

def picard(EOM,IC,IG,eps = 1e-4):
    # system functions from EOM class
    sysfuncptr = EOM.funcptr
    parameters = EOM.parameters
    t = EOM.tSpanExplicit

    # initial conditions
    initialConditions = IC
    initialGuess      = IG
    problemDim = len(initialConditions)

    # time values
    nSamples = int(len(t))
    delT = EOM.tEnd / len(t)

    # initialization of matrices
    last_K = zeros((problemDim,1))
    current_K = full((nSamples,problemDim),initialGuess)
    next_K = zeros((nSamples,problemDim))

    n = 0
    # picard iteration loop
    while any(abs(current_K[-1,:] - last_K[-1,:]) > eps): # if x_k - x_k-1 > eps, keep going
        for i in range(nSamples): # loop through the number of function sampling
            forcing = sysfuncptr(t,current_K[0:i+1,:],parameters)
            next_K[i] = initialConditions + trapz(forcing,dx=delT)
        # x_k = x(0) + integral from 0 to t_k of f(t,x_k-1) dt                    
        last_K = copy(current_K)
        current_K = copy(next_K)
        n = n + 1
        print(end='\rPicard Iterations: {}'.format(n))
    print()
    return current_K