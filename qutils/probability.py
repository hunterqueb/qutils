import numpy as np
from itertools import permutations

def generateSigmaBounds(xN,sigmaCov,sigmaBoundValue,num_pts):
    '''
    This function gives the initial Gaussian distribtuion of points based on the initial covariance and mean.
    \nInputs: xN - mean, sigmaCov - intitial covariance matrix, sigmaBoundValue - desired sigma bound, num_pts - number of points in sigma bound
    \nOutputs: numpy array of shape (num_pts,dim_num) of the generated sigma bounds, not equallaly spaced  

    Created: 6/17/2022
    Author: Hunter Quebedeaux

    Coverted from original MATLAB function written by Junkins cohort
    '''
    dim_num = len(xN)
    X0 = np.kron(np.ones((1,num_pts)), xN[:, None])
    plusMinus = 2 * (np.round(np.random.rand(num_pts,dim_num))-0.5)

    def perms(x):
        """Python equivalent of MATLAB perms."""
        return np.vstack(list(permutations(x)))[::-1]

    probOrd = perms(range(dim_num))

    # adsVec = np.zeros((dim_num,num_pts))

    probDelt = np.zeros((dim_num,1))
    comp = np.zeros((dim_num,1))

    for ii in range(0,num_pts):

        probInd = (ii % dim_num) # need to check with MATLAB where this is used, and 0/1 indexing

        probDelt[probOrd[probInd,0]] = np.random.rand() * sigmaBoundValue

        if dim_num > 2:
            for jj in range(1,dim_num-1):
                probDelt[probOrd[probInd,jj]] = np.random.rand() * np.sqrt(sigmaBoundValue ** 2 - np.sum(np.square(probDelt[probOrd[probInd,0:jj]])))

        probDelt[probOrd[probInd,dim_num-1]] = np.sqrt(sigmaBoundValue ** 2 - np.sum(np.square(probDelt[probOrd[probInd,0:dim_num-1]])))

        for jj in range(dim_num):
            comp[probOrd[probInd,jj]] = plusMinus[ii-1,probOrd[probInd,jj]] * probDelt[probOrd[probInd,jj]]

        # adsVec[:,ii] = comp

        X0[0:dim_num,ii] = X0[0:dim_num,ii] + np.matmul(np.sqrt(sigmaCov), comp).T

    bounds_0 = np.transpose(X0)

    return bounds_0
