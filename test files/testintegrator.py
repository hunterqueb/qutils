import numpy as np
import matplotlib.pyplot as plt
import math
import random

from quebutils.integrators import ode45


# data size set that define amount of data sets we will generate to train the network
DATA_SET_SIZE = 1
TIME_STEP = 0.01

# ------------------------------------------------------------------------
## NUMERICAL SOLUTION

def linPendulumODE(t,theta,p=None):
    m = 1
    k = 1
    dtheta1 = theta[1]
    dtheta2 = -k/m*(theta[0])
    return np.array([dtheta1, dtheta2])


L = 10
g = 9.81


def pendulumODE(t,theta,p=None):
    dtheta1 = theta[1]
    dtheta2 = -g/L*math.sin(theta[0])
    return np.array([dtheta1, dtheta2])


b = 0.1


def pendulumODEFriction(t,theta,p=None):
    m = 1
    dtheta1 = theta[1]
    dtheta2 = -b/m*theta[1]-g/L*math.sin(theta[0])
    return np.array([dtheta1, dtheta2])


c = 0.05
w = 1.4
k = 0.7

# strange values from my discussion with pugal
c = 0.2
w = 1.3
k = 2


def duffingOscillatorODE(t,y, p=[c, w**2, k**2]):
    dydt1 = y[1]
    dydt2 = -(p[0]*y[1] + p[1]*y[0] + p[2]*y[0]**3)

    return np.array([dydt1, dydt2])


sysfuncptr = duffingOscillatorODE
# sim time
t0, tf = 0, 20

t = np.arange(t0, tf, TIME_STEP)
degreesOfFreedom = 2
# initilize the arrays used to store the info from the numerical solution
theta = np.zeros((degreesOfFreedom,DATA_SET_SIZE))
output_seq = np.zeros((len(t),degreesOfFreedom))

# generate random data set of input thetas and output thetas and theta dots over a time series
theta = np.array([random.uniform(0.84, 0.86), 0])
# numericResult[i] = integrate.solve_ivp(pendulumODEFriction, (t0, tf), theta, "LSODA")

t, y = ode45(sysfuncptr,[t0,tf],theta)

print(t.shape)
plt.plot(y[:,0],y[:,1])
plt.show()