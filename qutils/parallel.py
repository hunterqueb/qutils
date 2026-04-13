from numba import njit, prange
import numpy as np
@njit(parallel=True)
def visVivaEnergy(r,v,mu):
    '''
    Calculates the energy of a body in orbit using the vis viva equation

    Inputs: r - position vector (x,2/3), v - velocity vector (x,2/3), mu - gravitational parameter of body being orbited
    Outputs: E - energy vector (x,1)

    Created: 6/23/22
    Author: Hunter Quebedeaux
    '''
    E = np.zeros((len(r[0,:]),1))
    for i in prange(len(r[0,:])):
        E[i] = (np.linalg.norm(v[:,i]) ** 2 ) / 2 - (mu / np.linalg.norm(r[:,i]))
    return E
