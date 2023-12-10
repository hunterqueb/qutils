# FILE IMPORTED 12/9/23

# Computational imports
import numpy as np
from numba import njit, jit, prange
import matplotlib.pyplot as plt
import gc
from tempfile import mkdtemp
import os
from ctypes import *



def kroneckMult(Q,X):
    '''
    Calculates the vector multiplation of X to Q where Q is a list of matrices to be kronecker producted
    Inputs: Q - tuple of matrices to be kroneckered together
            X - vector to be multiplied to the kronecker product
    Outputs: X - vector that represents the product of the kroneckered Q and X
    Source: https://www.mathworks.com/matlabcentral/fileexchange/23606-fast-and-efficient-kronecker-multiplication
    
    Author: Hunter Quebedeaux
    Created: 7/25/23

    '''
    N = len(Q)
    n = np.zeros((N,),dtype=int)
    nright = 1
    nleft = 1
    for i in range(N-1): 
        n[i] = Q[i].shape[0]
        nleft = nleft*n[i]
    n[N-1] = Q[N-1].shape[0]
    for i in range(N-1,-1,-1):
        base = 0
        jump = n[i]*nright
        for k in range(nleft):
            for j in range(nright):
                index1 = base+j
                index2 = base+j+nright*(n[i]-1) + 1
                X[index1:index2:nright] = Q[i] @ X[index1:index2:nright] 
            base = base+jump
        nleft = int(nleft/n[max((i-1,1))])
        nright = nright*n[i]

    return X

def __scale_up(z,x_min,x_max):
    """
    Scales up z \in [-1,1] to x \in [x_min,x_max]
    where z = (2 * (x - x_min) / (x_max - x_min)) - 1
    """
    
    return x_min + (z + 1) * (x_max - x_min) / 2
# def generateMeshGrid(minMax,nChebPts):
#     """
#     Generates a nd grid for plotting
#     Inputs: minMax - 2nx1 matrix in the format of [min(x),max(x),min(y),max(y),...] to generate chebyshev nodes in the function domain
#             evalPts - int that determines the number of chebyshev nodes generated
#     Outputs: xg - meshgrid x coordinate for graphing
#              yg - meshgrid y coordinate for graphing
    
#     Author: Hunter Quebedeaux
#     Created: 11/22/21
#     """
#     # order of approximating polynomial
#     x_min = minMax[0]
#     x_max = minMax[1]

#     y_min = minMax[2]
#     y_max = minMax[3]


#     # generate chebyshev nodes (the roots of Chebyshev polynomials, a Chebyshev polynomial of degree m-1 has m roots)
#     r_x = -np.cos((2*np.arange(1,nChebPts+1) - 1) * np.pi / (2*nChebPts))
#     r_y = -np.cos((2*np.arange(1,nChebPts+1) - 1) * np.pi / (2*nChebPts))

#     # scale up nodes to function domain
#     x = __scale_up(r_x,x_min,x_max)
#     y = __scale_up(r_y,y_min,y_max)
#     xg, yg = np.meshgrid(x, y)
#     return xg, yg

@njit
def evaluateCheb(minMax,alpha,evalPts):
    """
    Returns the value of a multivariate Chebyshev polynomial based on a given coefficent matrix and evaluation points.

    Inputs: minMax - 2*problemDimx1 matrix in the format of [min(x),max(x),min(y),max(y),...] to scale the chebyshev polynomials to an arbitrary range
            alpha - coefficent matrix reshaped to a nChebPts^problemDim of the chebyshev polynomial
            evalPts - problemDimx1 tuple that you want to approximate the pdf at
    Outputs: approx - value at (x,y,...) for the pdf

    References: Chebyshev Polynomials by JC Mason
        Sections 5.3.3 (bivariate polynomials) and 1.3.2 (polynomials on a general range)
    Author: Hunter Quebedeaux
    Created: 6/19/23

    relies on equal number of chebpts in the all dimensions, but can easily be swapped

    """

    problemDim = len(evalPts)

    nChebPts = int(np.power(len(alpha),1/problemDim))

    n = [0 for _ in range(problemDim)]

    for i in range(problemDim):
        n[i] = nChebPts

    approx = 0

    x_min = minMax[0]
    x_max = minMax[1]
    
    x = evalPts[0]
    s1 = (2*x - (x_min + x_max))/(x_max-x_min)

    # generate the chebyshev polynomials, these will be needed no matter the dimension

    # x
    Tx = np.zeros(n[0])
    Tx[0] = 1;Tx[1] = s1
    for k in range(2,n[0]):
        Tx[k] = 2*s1*Tx[k-1] - Tx[k-2]

    # i could fix this horrible looking code with a recursive function...
    # im not gonna do that right now but it would reduce runtime


    # check problem dimension to create the chebyshev polynomials if larger dimensions exist

    if problemDim > 1:
        y_min = minMax[2]
        y_max = minMax[3]

        y = evalPts[1]
        s2 = (2*y - (y_min + y_max))/(y_max-y_min)
        
        # y
        Ty = np.zeros(n[1])
        Ty[0] = 1;Ty[1] = s2
        for k in range(2,n[1]):
            Ty[k] = 2*s2*Ty[k-1] - Ty[k-2]


    if problemDim > 2:
        z_min = minMax[4]
        z_max = minMax[5]

        z = evalPts[2]
        s3 = (2*z - (z_min + z_max))/(z_max-z_min)

        Tz = np.zeros(n[2])
        Tz[0] = 1;Tz[1] = s3
        for k in range(2,n[2]):
            Tz[k] = 2*s3*Tz[k-1] - Tz[k-2]

    if problemDim > 3:
        vx_min = minMax[6]
        vx_max = minMax[7]

        vx = evalPts[3]
        s4 = (2*vx - (vx_min + vx_max))/(vx_max-vx_min)

        Tvx = np.zeros(n[3])
        Tvx[0] = 1;Tvx[1] = s4
        for k in range(2,n[3]):
            Tvx[k] = 2*s4*Tvx[k-1] - Tvx[k-2]

    if problemDim > 4:
        vy_min = minMax[8]
        vy_max = minMax[9]

        vy = evalPts[4]
        s5 = (2*vy - (vy_min + vy_max))/(vy_max-vy_min)

        Tvy = np.zeros(n[4])
        Tvy[0] = 1;Tvy[1] = s5
        for k in range(2,n[4]):
            Tvy[k] = 2*s5*Tvy[k-1] - Tvy[k-2]

    if problemDim > 5:
        vz_min = minMax[10]
        vz_max = minMax[11]

        vz = evalPts[5]
        s6 = (2*vz - (vz_min + vz_max))/(vz_max-vz_min)

        Tvz = np.zeros(n[5])
        Tvz[0] = 1;Tvz[1] = s6
        for k in range(2,n[5]):
            Tvz[k] = 2*s6*Tvz[k-1] - Tvz[k-2]


    # evaluating chebyshev poly
    if problemDim == 1:
        for i in range(n[0]):
            approx = approx + alpha[i] * Tx[i]

    if problemDim == 2:
        for i in range(n[0]):
            for j in range(n[1]):
                approx = approx + alpha[j*nChebPts+i]*Tx[i]*Ty[j]
                # approx = approx + alpha[i*nChebPts+j]*Tx[i]*Ty[j]

    elif problemDim == 3:
        for i in range(n[0]):
            for j in range(n[1]):
                for k in range(n[2]):
                    approx = approx + alpha[i*nChebPts*nChebPts+j*nChebPts + k]*Tx[i]*Ty[j]*Tz[k]

    elif problemDim == 4:
        for i in range(n[0]):
            for j in range(n[1]):
                for k in range(n[2]):
                    for ii in range(n[3]):
                        # approx = approx + alpha[i*nChebPts*nChebPts*nChebPts+j*nChebPts*nChebPts + k*nChebPts + ii]*Tx[i]*Ty[j]*Tz[k]*Tvx[ii]
                        approx = approx + alpha[ii*nChebPts*nChebPts*nChebPts+k*nChebPts*nChebPts + j*nChebPts + i]*Tx[i]*Ty[j]*Tz[k]*Tvx[ii]
                        # approx = approx + alpha[i,j,k,ii]*Tx[i]*Ty[j]*Tz[k]*Tvx[ii]


    elif problemDim == 5:
        for i in range(n[0]):
            for j in range(n[1]):
                for k in range(n[2]):
                    for ii in range(n[3]):
                        for jj in range(n[4]):
                            approx = approx + alpha[i*nChebPts*nChebPts*nChebPts*nChebPts+j*nChebPts*nChebPts*nChebPts + k*nChebPts*nChebPts + ii*nChebPts + jj]*Tx[i]*Ty[j]*Tz[k]*Tvx[ii]*Tvy[jj]

    elif problemDim == 6:
        for i in range(n[0]):
            for j in range(n[1]):
                for k in range(n[2]):
                    for ii in range(n[3]):
                        for jj in range(n[4]):
                            for kk in range(n[5]):
                                approx = approx + alpha[i*nChebPts*nChebPts*nChebPts*nChebPts*nChebPts+j*nChebPts*nChebPts*nChebPts*nChebPts + k*nChebPts*nChebPts*nChebPts + ii*nChebPts*nChebPts + jj*nChebPts + kk]*Tx[i]*Ty[j]*Tz[k]*Tvx[ii]*Tvy[jj]*Tvz[kk]


    
    # if the array is reshaped using fortran indexing
    # access 2d
    # fgrid[x+y*width]

    # access 3d
    # x + width*y + width*height*z;

    # access 6d
    # x + x_length * y + x_length * y_length * z +
    # x_length * y_length * z_length * vx + x_length * y_length * z_length * vx_length * vy + x_length * y_length * z_length * vx_length * vy_length * vz   


    # if the array is reshaped using C indexing
    # access 2d
    # fgrid[x*width+y]

    # access 3d
    # arr3d[x*3*3 + y*3 + z]

    # access 6d
    # [x*3*3*3*3*3 + y*3*3*3*3 + z*3*3*3 + vx*3*3 + vy*3 + vz]

    return approx

def evaluateChebC(d,alpha,evalPts,nChebPts,lib_cheb):
    
    lib_cheb.evaluateCheb.restype = c_double 

    problemDim = len(evalPts)

    problemDim = c_int(problemDim)
    nChebPts = c_int(nChebPts)

    d = np.ctypeslib.as_ctypes(d)
    alpha = np.ctypeslib.as_ctypes(alpha)
    evalPts = np.ctypeslib.as_ctypes(np.array(evalPts))

    approx = lib_cheb.evaluateCheb(d,alpha,evalPts,problemDim,nChebPts)

    return approx

def generateChebCoeff(minMax,G,n,dataType = np.float32):
    '''
    Returns the nd Coefficent Matrix for any discrete nd function where the number of evaluation pts equals the number of approximation of the function
    Inputs: minMax - 2Nx1 vector of the min and max values of the domain of each dimension,
            G - (evalPts^n,1) vector that is the discrete nd function at chebyshev nodes
            n - integer corresponding to the dimensions of the problem.
    Outputs: a - coefficant matrix of the size (len(G),1)

    Author: Hunter Quebedeaux
    Created: 7/25/2023
    Source: Orthogonal polynomial approximation in higher dimensions: Applications in astrodynamics; AH Bani Younes - 2013 

    relies on equal number of chebpts in the all dimensions
    '''

    evalPts = int(np.power(len(G),1/n))

    n_x = evalPts

    m_x = n_x - 1

    zeta = np.polynomial.chebyshev.chebpts2(evalPts) # only returns an array of pts, i need to assign the range myself

    # kronecker product calculation
    # doing this reduces the memory allocation required by python because the W and Wx matrices are diagonal.
    # because of their diagonallity, performing these space operations reduces the space required exponentially!
    # has the added benefit of being faster than performing kronecker operations and allocating large matrices constantly
    Wx = np.identity(evalPts)
    Wx[0][0] = 1/2 ; Wx[-1][-1] = 1/2

    # a kroncker product is a giant distribution of one matrix element to another full matrix 
    W = np.diag(Wx)
    newList = np.array([])

    for _ in range(n-1):
        newList = np.array([])
        for i in range(len(W)):
            newList = np.append(newList,np.diag(W[i]*Wx))
        W = newList

    W.setflags(write=1)
    for i in range(len(W)):
        W[i] = np.sqrt(W[i])
    # W = fractional_matrix_power(W, 0.5)

    WG = np.multiply(W , G)

    # using only del W does not free the memory asap as opposed to W = None, del only clears the reference to the variable. 
    # we first set the matrix to a null pointer, then delete the variable to free the memory asap
    # Source:
    # https://stackoverflow.com/questions/35316728/does-setting-numpy-arrays-to-none-free-memory

    # x
    Tx = np.zeros((n_x,n_x))
    for i in range(n_x):
        Tx[0][i] = 1;Tx[1][i] = zeta[i]
    for k in range(2,n_x):
        for j in range(n_x):
            Tx[k][j] = 2*zeta[j]*Tx[k-1][j] - Tx[k-2][j]
    
    Cxbar = Tx * (1 / (m_x));         

    del W; del Wx; del Tx; del zeta; del newList
    gc.collect()

    for j in range(n_x):
        for i in range(n_x):
            if (j == 0 and i == 0) or (j == 0 and i == m_x) or (j == m_x and i == 0) or (j == m_x and i == m_x):
                Cxbar[j][i] = np.sqrt(2)*Cxbar[j][i]
            elif (not i == 0 and not i == m_x):
                Cxbar[j][i] = 2*Cxbar[j][i]
    for i in range(n_x):
        Cxbar[0][i] = Cxbar[0][i]/2
        Cxbar[-1][i] = Cxbar[-1][i]/2

    Cxbar = Cxbar.astype(dataType)
    WG = WG.astype(dataType)

    # all the magic in this calc happens here
    # the giant kronecker matrix is not stored in memory
    a = kroneckMult([Cxbar for _ in range(n)],WG)

    # how do i reshape this??? - dont do it!
    # a = np.reshape(a,(evalPts,evalPts))

    if dataType == np.float16:
        saveType = np.float32
    else:
        saveType = dataType
    return a.astype(saveType)


def generateChebCoeffMap(minMax,G,n,dataType = np.float32):
    '''
    Returns the nd Coefficent Matrix for any discrete nd function where the number of evaluation pts equals the number of approximation of the function using memory maps
    Inputs: minMax - 2Nx1 vector of the min and max values of the domain of each dimension,
            G - (evalPts^n,1) vector that is the discrete nd function at chebyshev nodes
            n - integer corresponding to the dimensions of the problem.
    Outputs: a - coefficant matrix of the size (len(sqrt(G)),len(sqrt(G)),...)

    Author: Hunter Quebedeaux
    Created: 6/16/2023
    Source: Orthogonal polynomial approximation in higher dimensions: Applications in astrodynamics; AH Bani Younes - 2013 
    
    relies on equal number of chebpts in the all dimensions
    '''

    evalPts = int(np.power(len(G),1/n))

    n_x = evalPts

    m_x = n_x - 1

    zeta = np.polynomial.chebyshev.chebpts2(evalPts) # only returns an array of pts, i need to assign the range myself

    # kronecker product calculation
    # doing this reduces the memory allocation required by python because the W and Wx matrices are diagonal.
    # because of their diagonallity, performing these space operations reduces the space required exponentially!
    # has the added benefit of being faster than performing kronecker operations and allocating large matrices constantly
    Wx = np.identity(evalPts)
    Wx[0][0] = 1/2 ; Wx[-1][-1] = 1/2

    # a kroncker product is a giant distribution of one matrix element to another full matrix 
    W = np.diag(Wx)
    newList = np.array([])

    for _ in range(n-1):
        newList = np.array([])
        for i in range(len(W)):
            newList = np.append(newList,np.diag(W[i]*Wx))
        W = newList

    W.setflags(write=1)
    for i in range(len(W)):
        W[i] = np.sqrt(W[i])
    # W = fractional_matrix_power(W, 0.5)

    WG = np.multiply(W , G)

    # using only del W does not free the memory asap as opposed to W = None, del only clears the reference to the variable. 
    # we first set the matrix to a null pointer, then delete the variable to free the memory asap
    # Source:
    # https://stackoverflow.com/questions/35316728/does-setting-numpy-arrays-to-none-free-memory

    # x
    Tx = np.zeros((n_x,n_x))
    for i in range(n_x):
        Tx[0][i] = 1;Tx[1][i] = zeta[i]
    for k in range(2,n_x):
        for j in range(n_x):
            Tx[k][j] = 2*zeta[j]*Tx[k-1][j] - Tx[k-2][j]
    
    Cxbar = Tx * (1 / (m_x));         

    del W; del Wx; del Tx; del zeta; del newList
    gc.collect()

    for j in range(n_x):
        for i in range(n_x):
            if (j == 0 and i == 0) or (j == 0 and i == m_x) or (j == m_x and i == 0) or (j == m_x and i == m_x):
                Cxbar[j][i] = np.sqrt(2)*Cxbar[j][i]
            elif (not i == 0 and not i == m_x):
                Cxbar[j][i] = 2*Cxbar[j][i]
    for i in range(n_x):
        Cxbar[0][i] = Cxbar[0][i]/2
        Cxbar[-1][i] = Cxbar[-1][i]/2

    @njit(parallel=False)
    def memmapKronecker(A,B,memMap,iter):
        n = evalPts**(1+iter)
        for i in prange(n):
            for j in prange(n):
                memMap[evalPts*i:evalPts*i+evalPts,evalPts*j:evalPts*j+evalPts] = (A[i][j] * B)[:]
                # I suspect a large amount of runtime is used in access the hard disk file, but
                # this representation slowness is exasturbated by the array slicing, and
                # doing this upsets me on a deep level...
                # however, I think this is the only option if i want to run at higher approximation rates
                # IF memMap arrays can be converted to a C type, I could offload this caclulation to C.
                # not sure what my other options are.
    Cxbar = Cxbar.astype(dataType)
    WG = WG.astype(dataType)

    tempDir = mkdtemp()
    print("Memmap temp directory located at " + tempDir)
    tempFilename = os.path.join(tempDir, 'CbarTemp.npy')
    filename = os.path.join(tempDir, 'Cbar.npy')
    
    # create temp file for saving Cbar -- me "touch-ing" the file
    Cbar = np.memmap(filename, dtype=dataType, mode='w+',shape=(evalPts,evalPts))
    CbarTemp = np.memmap(tempFilename, dtype=dataType, mode='w+',shape=(1,1))

    Cbar[:] = Cxbar[:]
    del Cbar
    del CbarTemp

    for i in range(n-1):
        CbarTemp = np.memmap(tempFilename, dtype=dataType, mode='w+',shape=(evalPts**(1+i),evalPts**(1+i)))
        Cbar = np.memmap(filename, dtype=dataType, mode='r+',shape=(evalPts**(1+i),evalPts**(1+i)))
        CbarTemp[:] = Cbar[:]
        del Cbar
        Cbar = np.memmap(filename, dtype=dataType, mode='r+',shape=(evalPts**(2+i),evalPts**(2+i)))
        memmapKronecker(CbarTemp,Cxbar,Cbar,i)
        del CbarTemp
        del Cbar

    Cbar = np.memmap(filename, dtype=dataType, mode='r',shape=(evalPts**(n),evalPts**(n)))
    a = np.matmul(Cbar, WG).astype(np.float32)

    del Cbar
    os.remove(filename)
    os.remove(tempFilename)
    os.rmdir(tempDir)
    # how do i reshape this??? - dont do it!
    # a = np.reshape(a,(evalPts,evalPts))

    if dataType == np.float16:
        saveType = np.float32
    else:
        saveType = dataType
    return a.astype(saveType)

def __quadwts2(n):
    '''
    Converted from MATLAB package chebfun, function quadwts2
    '''
    if ( n == 0 ):                      # Special case (no points!)
        w = []
    elif ( n == 1 ):                  # Special case (single point)
        w = 2
    else:                               # General case
        temp = np.arange(2,n-1,2)
        c = 2/np.insert(1-np.power(temp,2),0,1)  # Exact integrals of T_k (even)
        index = np.arange(np.floor(n/2),1,-1)
        for i in range(len(index)):
            c = np.append(c,c[int(index[i])-1])      # Mirror for DCT via FFT
        w = np.real(np.fft.ifft(c))                   # Interior weights
        w[0] = w[0]/2
        w = np.append(w,w[0]/2)             # Boundary weights
    return w

def __scaleWeights(w,dom):
    #SCALEWEIGHTS   Scale the Chebyshev weights W from [-1,1] to DOM.
    if ( dom[0] == -1 and dom[1] == 1 ):
        # Nodes are already on [-1, 1];
        return w
    # Scale the weights:
    w = (dom[1]-dom[0])/2 * w
    return w

def iClenshawCurtis(dom,PDFnd):
    '''
    Integration of chebyshev approximated pdf that returns the total area under surface. Usually used for understand CDF 
    Inputs: dom - domain of chebyshev approximation, PDFnd - 1d array of values that correspond to PDF values at chebyshev nodes of the size n^problemDim x 1. n MUST BE EVEN
    compatiable array output from self.evaluateCheb. if unwrapping a multi-dim array yourself, ensure that it is accessed via C indexing, not fortran indexing 
    Outputs: returns single float value of the total area under surface
    
    note: there has to be a better way in doing these calculations, im just lazy and i know that we wont go over 6 dimensions for a while.
    # https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/23972/versions/22/previews/chebfun/examples/quad/html/GaussClenCurt.html?access_key=
    
    Created: 6/19/23
    Author: Hunter Quebedeaux
    '''

    problemDim = int(len(dom)/2)

    nChebPts = int(np.power(len(PDFnd),1/problemDim))

    n = [nChebPts for _ in range(problemDim)]

    iclenshawcurtis = 0

    w1 = __scaleWeights(__quadwts2(n[0]),(dom[0],dom[1])) # weights for dim 1 
    w2 = __scaleWeights(__quadwts2(n[1]),(dom[2],dom[3])) # wieghts for dim 2

    if problemDim > 2:
        w3 = __scaleWeights(__quadwts2(n[2]),(dom[4],dom[5]))

    if problemDim > 3:
        w4 = __scaleWeights(__quadwts2(n[3]),(dom[6],dom[7]))

    if problemDim > 4:
        w5 = __scaleWeights(__quadwts2(n[4]),(dom[8],dom[9]))

    if problemDim > 5:
        w6 = __scaleWeights(__quadwts2(n[5]),(dom[10],dom[11]))

    if problemDim == 2:
        for i in range(n[0]):
            for j in range(n[1]):
                iclenshawcurtis = iclenshawcurtis + w1[i] * w2[j] * PDFnd[i*n[0] + j]

    elif problemDim == 3:
        for i in range(n[0]):
            for j in range(n[1]):
                for k in range(n[2]):
                    iclenshawcurtis = iclenshawcurtis + w1[i] * w2[j] * w3[k] * PDFnd[i*n[0]*n[1] + j*n[1] + k]

    elif problemDim == 4:
        for i in range(n[0]):
            for j in range(n[1]):
                for k in range(n[2]):
                    for ii in range(n[3]):
                        iclenshawcurtis = iclenshawcurtis + w1[i] * w2[j] * w3[k] * w4[ii] * PDFnd[i*n[0]*n[1]*n[2] + j*n[1]*n[2] + k*n[2] + ii]

    elif problemDim == 5:
        for i in range(n[0]):
            for j in range(n[1]):
                for k in range(n[2]):
                    for ii in range(n[3]):
                        for jj in range(n[4]):
                            iclenshawcurtis = iclenshawcurtis + w1[i] * w2[j] * w3[k] * w4[ii] * w5[jj] * PDFnd[i*n[0]*n[1]*n[2]*n[3] + j*n[1]*n[2]*n[3] + k*n[2]*n[3] + ii*n[3] + jj]

    elif problemDim == 6:
        for i in range(n[0]):
            for j in range(n[1]):
                for k in range(n[2]):
                    for ii in range(n[3]):
                        for jj in range(n[4]):
                            for kk in range(n[5]):
                                iclenshawcurtis = iclenshawcurtis + w1[i] * w2[j] * w3[k] * w4[ii] * w5[jj] * w6[kk] * PDFnd[i*n[0]*n[1]*n[2]*n[3]*n[4] + j*n[1]*n[2]*n[3]*n[4] + k*n[2]*n[3]*n[4] + ii*n[3]*n[4] + jj*n[4] + kk]

    return iclenshawcurtis

def iReduceToMarginal(dom,PDFnd,axis):
    '''
    Reduces an nd PDF by one order using the trapzoidal rule. reduction is performed on the axis dimension.
    calling this function twice for a 2d pdf first can reduce to the final CDF value, but this is not as accurate 
    as the clenshaw curtis quad routine.
    Inputs: dom - domain of the PDF  
            PDF2d - a 2d array that corresponds to the approximated pdf values at chebyshev nodes
            axis - axis that you want the reduction to happen in. ie: 0 - x, 1 - y, ...
    Outputs: a reduced dimension probability

    Created: 6/29/23
    Author: Hunter Quebedeaux

    relies on equal number of chebpts in the all dimensions
    '''

    n = len(PDFnd)
    pts = np.polynomial.chebyshev.chebpts2(n)
    yDom = np.interp(pts, (pts.min(), pts.max()), (dom[axis*2], dom[axis*2 + 1]))

    return np.trapz(PDFnd,x = yDom,axis = axis)

def iReduceToMarginalM(dom,PDFnd,M):
    '''
    Reduces an nd PDF by order of M using the trapzoidal rule starting from the last axis moving forward
    For example, reducing by an order of 3 of a 6d pdf will result in collapsing the pdf to only describe the position variables 
    Inputs: dom - domain of the PDF  
            PDF2d - a 2d array that corresponds to the approximated pdf values at chebyshev nodes
            M - number of axes to collapse starting from the back
    Outputs: a reduced dimension probability

    Created: 6/29/23
    Author: Hunter Quebedeaux
    
    relies on equal number of chebpts in the all dimensions
    '''
    problemDim = int(len(dom)/2)

    n = len(PDFnd)
    pts = np.polynomial.chebyshev.chebpts2(n)
    for i in range(problemDim-1,M-1,-1):
        yDom = np.interp(pts, (pts.min(), pts.max()), (dom[i*2], dom[i*2 + 1]))
        PDFnd = np.trapz(PDFnd,x = yDom,axis = i)
    return PDFnd

def generateCDF(dom,PDFnd,X=None):
    '''
    Currently not working

    relies on equal number of chebpts in the all dimensions
    '''
    nChebPts = len(PDFnd)
    
    # 1 - reduce to 1d marginal pdf
    margPDF = iReduceToMarginalM(dom,PDFnd,1)

    # 2 - approximate new coefficents from 1d pdf
    alpha1d = generateChebCoeff((dom[0],dom[1]),margPDF,1)
    
    # this is the maginal pdf
    eval = np.zeros_like(X)
    if X is not None:
        for i in range(len(X)):
            eval[i] = evaluateCheb(dom,alpha1d,(X[i],))

    # 3 - use new coefficents with integrated chebyshev polynomials
    n_x = nChebPts

    zeta = np.polynomial.chebyshev.chebpts2(nChebPts+1) # only returns an array of pts, i need to assign the range myself

    x_min = dom[0]
    x_max = dom[1]
    
    x = X
    s1 = (2*x - (x_min + x_max))/(x_max-x_min)
    
    
    # Tx = np.zeros(n_x)
    # Tx[0] = 1;Tx[1] = s1
    # for k in range(2,n[0]):
    #     Tx[k] = 2*s1*Tx[k-1] - Tx[k-2]

    Tx = np.zeros((n_x,n_x))
    for i in range(n_x):
        Tx[0][i] = 1;Tx[1][i] = s1[i]
    for k in range(2,n_x):
        for j in range(n_x):
            Tx[k][j] = 2*s1[j]*Tx[k-1][j] - Tx[k-2][j]
    iTx = np.zeros((n_x,n_x))
    for i in range(n_x):
        iTx[0][i] = s1[i]; iTx[1][i] = s1[i] ** 2 / 2
    for k in range(2,n_x):
        for j in range(n_x):
            iTx[k][j] = Tx[k+1][j]/(2*(n_x+1)) - Tx[k-1][j]/(2*(n_x-1))

    CDFValues = np.matmul(iTx.T,alpha1d)

    return CDFValues
    # return eval
if __name__ == '__main__':
    # test import
    import scipy.io as spio
    # mat = spio.loadmat('coefficents.mat', squeeze_me=True)
    # a = mat['test']
    # d = mat['d']
    # dPy = np.array([0.45479094, 0.91387777, 0.24011789, 0.76839796])
    # print(evaluateCheb(d,a,(0.6,0.5)),'Test Sample at x = 0.6, xdot = 0.5')
    # print(evaluateCheb(dPy,a,(0.6,0.5)),'Test Sample w/ python region at x = 0.6, xdot = 0.5')
    # print(4.391685244177514,'Test Sample at x = 0.6, xdot = 0.5 from MATLAB')

    # w = __quadwts2(50)
    # wWieght = __scaleWeights(w,[-2,2])
    pass