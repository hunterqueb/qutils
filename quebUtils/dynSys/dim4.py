import numpy as np
from numba import njit

def planarOrbitODE(t,y,p):
    #y = [x xdot y ydot]
    mu = p
    if y.ndim > 1:
        r = np.sqrt(np.multiply(y[:,0], y[:,0]) + np.multiply(y[:,2], y[:,2]))
        dydt1 = y[:,1]
        dydt2 = np.multiply(-(mu/(np.power(r,3))), y[:,0])
        dydt3 = y[:,3]
        dydt4 = np.multiply(-(mu/(np.power(r,3))), y[:,2])
    else:
        r = np.sqrt(y[0] * y[0] + y[2] * y[2])
        dydt1 = y[1]
        dydt2 = -(mu/(r ** 3)) * y[0]
        dydt3 = y[3]
        dydt4 = -(mu/(r ** 3)) * y[2]
    return np.array([dydt1,dydt2,dydt3,dydt4])

@njit(nogil=True)
def planarOrbitODEnjit(t,y,p):
    #y = [x y xdot ydot]
    mu = p[0]
    r = np.sqrt(y[0] * y[0] + y[2] * y[2])
    # if r < 6371:
    #     print('collides with earth')
    dydt1 = y[2]
    dydt2 = y[3]
    dydt3 = -(mu/(r ** 3)) * y[0]
    dydt4 = -(mu/(r ** 3)) * y[1]

    return np.array((dydt1,dydt2,dydt3,dydt4))

def nondim_cr3bp(t, Y,mu=7.348E22/(5.974E24 + 7.348E22)):
    """Solve the CR3BP in nondimensional coordinates. Defaults to Earth moon system.
    
    The state vector is Y, with the first three components as the
    position of $m$, and the second three components its velocity.
    
    The solution is parameterized on $\\pi_2$, the mass ratio.
    """
    # Get the position and velocity from the solution vector
    x, y = Y[:2]
    xdot, ydot = Y[2:]

    # Define the derivative vector

    dydt1 = xdot
    dydt2 = ydot
    sigma = np.sqrt(np.sum(np.square([x + mu, y])))
    psi = np.sqrt(np.sum(np.square([x - 1 + mu, y])))
    dydt3 = 2 * ydot + x - (1 - mu) * (x + mu) / sigma**3 - mu * (x - 1 + mu) / psi**3
    dydt4 = -2 * xdot + y - (1 - mu) * y / sigma**3 - mu * y / psi**3
    return np.array([dydt1, dydt2,dydt3,dydt4])


