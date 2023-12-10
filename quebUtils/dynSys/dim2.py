import numpy as np
from numba import njit

#SIMPLE HARMONIC OSCILLATOR
def simpleHarmonicODE(t,y,p):
    if y.ndim > 1: # vectorized input 
        dydt1 = y[:,1]
        dydt2 = -(p[0]/p[1]) * y[:,0]
    else: # single input 
        dydt1 = y[1]
        dydt2 = -(p[0]/p[1]) * y[0]

    return np.array((dydt1,dydt2))

@njit(nogil=True)
def simpleHarmonicODEnjit(t,y,p):
    dydt1 = y[1]
    dydt2 = -(p[0]/p[1]) * y[0]

    return np.array((dydt1,dydt2))

def simpleHarmonicAnalyticalSolution(parameters, t, initialConditions):
    '''This function provides the analytical solution of the simple
    spring-mass system (i.e. a simple harmonic oscillator).

    Inputs: Spring stiffness (N/m), attached mass (kg), time of interest, 
    initial conditions(2x1 vector) 

    Outputs: position and velocity 

    Created: 9/30/21
    Author: Hunter Quebedeaux'''
    stiffness = parameters[0]
    mass = parameters[1]

    omega = np.sqrt(stiffness/mass)

    x = initialConditions[0] * np.cos(omega * t) + (initialConditions[1]/omega) * np.sin(omega * t)
    xdot = -omega * initialConditions[0] * np.sin(omega * t) + (initialConditions[1]) * np.cos(omega * t)

    return np.array((x,xdot))



#DUFFING OSCILLATOR
def duffingOscillatorODE(t,y,p):
    if y.ndim > 1: # vectorized input 
        dydt1 = y[:,1]
        dydt2 = -(p[0]*y[:,1] + p[1]*y[:,0] + p[2]*y[:,0]**3 + p[3]*np.sin(p[4] * t + p[5]))
    else: # single input 
        dydt1 = y[1]
        dydt2 = -(p[0]*y[1] + p[1]*y[0] + p[2]*y[0]**3 + p[3]*np.sin(p[4] * t + p[5]))

    return np.array((dydt1,dydt2))

@njit(nogil=True)
def duffingOscillatorODEnjit(t,y,p):
    dydt1 = y[1]
    dydt2 = -(p[0]*y[1] + p[1]*y[0] + p[2]*y[0]**3 + p[3]*np.sin(p[4] * t + p[5]))

    return np.array((dydt1,dydt2))

@njit(nogil=True)
def duffingOscillatorProbeODE(t,y,p):
    # only 3 parameters
    if y.ndim > 1: # vectorized input 
        dydt1 = y[:,1]
        dydt2 = -(p[0]*y[:,1] + (p[1] * p[1]) * y[:,0] - (p[2] * p[2]) * y[:,0]**3)
    else: # single input 
        dydt1 = y[1]
        dydt2 = -(p[0]*y[1] + (p[1] * p[1]) * y[0] - (p[2] * p[2]) * y[0]**3)

    return np.array((dydt1,dydt2))

@njit(nogil=True)
def duffingOscillatorProbeODEnjit(t,y,p):
    # only 3 parameters
    dydt1 = y[1]
    dydt2 = -(p[0]*y[1] + (p[1] * p[1]) * y[0] - (p[2] * p[2]) * y[0]**3)

    return np.array((dydt1,dydt2))



# NONLINEAR PENDULUM
def nonlinearPendulumODE(t,y,p):
    if y.ndim > 1: # vectorized input 
        dydt1 = y[:,1]
        dydt2 = -(p[0]/p[1]) * np.sin(y[:,0])
    else:
        dydt1 = y[1]
        dydt2 = -(p[0]/p[1]) * np.sin(y[0])

    return np.array([dydt1,dydt2])

@njit(nogil=True)
def nonlinearPendulumODEnjit(t,y,p):
    # p = [gravity, length]
    dydt1 = y[1]
    dydt2 = -(p[0]/p[1]) * np.sin(y[0])

    return np.array((dydt1,dydt2))



if __name__ == "__main__":
    from quebutils.integrators import myRK4 
    import time
    import matplotlib.pyplot as plt

    parameters = np.array([0.05,4,0.2,-0.5,10,np.pi/2])

    initialConditions = np.array([2,0],dtype=np.float64)
    
    tStart = 0
    tEnd = 300
    dt = 0.01
    tSpan = np.array([tStart,tEnd])
    tSpanExplicit = np.arange(tStart,tEnd,dt)


    start = time.time()
    solNonlinear = myRK4(duffingOscillatorODEnjit,initialConditions,tSpanExplicit,parameters)
    end = time.time()
    print("solution time, myRK4",(end - start))
    
    # print(sol.t.shape)

    # plt.plot(tSpanExplicit,solLSODA[:,0])
    # plt.plot(tSpanExplicit,solLSODA[:,1])
    # plt.plot(soly[:,1])

    plt.figure()
    plt.plot(tSpanExplicit,solNonlinear[:,0])

    plt.figure()
    plt.plot(solNonlinear[:,0],solNonlinear[:,1])

    # plt.plot(sol.t,sol.y[0,:])
    plt.show()
