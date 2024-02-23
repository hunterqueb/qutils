
import numpy as np
from numba import njit, prange

def dim2NonDim4(array,DU = 6378.1 ,TU = ((6378.1)**3 / 3.96800e14)**0.5):

    array = array / DU
    array[:,2:4] = array[:,2:4] * TU

    return array


def nonDim2Dim4(array,DU = 6378.1 ,TU = 806.80415):
    array = array * DU
    array[:,2:4] = array[:,2:4] / TU

    return array

def genTimestep4EquiTrueAnom(numPoints,numPeriods,e,T):
    '''
    Generates a 1d vector of the time points corresponding to a linear spacing in the true anomaly.
    Only tested for 2d planar orbits
    Input: numPoints - number of equispaced true anomaly points
           numPeriods - number of periods to generate
           e - eccentricty of orbit
           T - period of orbit (can be dimensional or nondimensional)
    Output: tEval - 1d numpy array of time points
    

    '''
    trueAnom = np.linspace(0,2*np.pi,numPoints)
    # linspace f 0 to 360 

    # find E from f
    # 3.13b curtis
    E = 2*np.arctan(np.sqrt((1-e)/(1+e))*np.tan(trueAnom/2))

    # find Me from E and e
    # 3.14 curtis
    Me = E - np.multiply(e,np.sin(E))

    # ensure the angles lies on the the correct range from 0 to 2pi
    Me[int(numPoints/2):] = Me[int(numPoints/2):] + 2*np.pi

    # find t from Me and period
    # 3.15 from curtis
    tEval = Me/(2*np.pi) * T

    # construct time eval matrix
    tEvalM = np.zeros((numPoints,numPeriods))
    tEvalM[:,0] = tEval

    # compute for arbitrary periods
    for i in range(numPeriods-1):
        tEvalM[:,i+1] = tEvalM[:,i] + T
        tEval = np.concatenate((tEval,tEvalM[1:,i+1]))
    
    return tEval

def SolveKeplerEq(M,e,eps=1e-6 ,N=5):

    # Inputs :-
    # M:   Mean Anomally
    # E0:  Initial Guess for Eccentric Anomally
    # eps: Tolerance
    # N:   Number of iterations

    E = M
    err = 1
    k = 0

    while((err>eps) and (k<=N)):
        
        k = k + 1
        
        f = M - E + e*np.sin(E)
        
        fd = -1 + e*np.cos(E)
        
        DelE = -f/fd
        
        E = E + DelE
        
        f = M - E + e*np.sin(E)
        
        err = np.abs(f)
        
    Feval = err

    return E,Feval

def ECI2OE(r0,v0,mu = 398600):

    # modified to output elements as [Omega i omega a e M0 P] to align with
    # lagrange planetary equations
    R0 = np.linalg.norm(r0)
    V0 = np.linalg.norm(v0)

    h = np.cross(r0,v0)
    H = np.linalg.norm(h)

    a = (2/R0 - V0**2/mu)**-1

    p = H**2/mu

    c = np.cross(v0, h) - (mu/R0) * r0

    e = np.linalg.norm(c)/mu

    C = np.zeros((3,3))
    i_h = h/H
    i_e = c/(mu*e)
    i_m = np.cross(i_h,i_e)

    C[2,:] = i_h
    C[0,:] = i_e
    C[1,:] = i_m

    i = np.arccos(C[2,2])
    Omega = np.arctan2(C[2,0],-C[2,1])
    omega = np.arctan2(C[0,2],C[1,2])

    sigma0 = np.dot(r0,v0) / np.sqrt(mu)

    E0 = np.arctan2(sigma0/np.sqrt(a), 1-(R0/a))

    M0 = E0 - e*np.sin(E0)

    n = np.sqrt(mu)/a**1.5

    P = 2*np.pi/n

    return np.array((a, e, i, Omega, omega, M0, P))

def OE2ECI(OE,t, mu = 398600):
 
    a = OE[0]
    e = OE[1]
    i = OE[2]
    Omega = OE[3]
    omega = OE[4]
    M0 = OE[5]

    COmega_3 = np.array([[np.cos(Omega), np.sin(Omega), 0],
                         [-np.sin(Omega), np.cos(Omega), 0], [0, 0, 1]])
    Ci_1 = np.array([[1, 0, 0], [0, np.cos(i), np.sin(i)],
                     [0, -np.sin(i), np.cos(i)]])
    Comega_3 = np.array([[np.cos(omega), np.sin(omega), 0],
                         [-np.sin(omega), np.cos(omega), 0], [0, 0, 1]])

    C = np.matmul(np.matmul(Comega_3, Ci_1), COmega_3)
    
    
    n = np.sqrt(mu/a**3)

    M = M0 + n*t

    E,Feval = SolveKeplerEq(M, e)

    x = a*(np.cos(E) - e)
    y = a*np.sqrt(1-e**2)*np.sin(E)

    R = a*(1-e*np.cos(E))

    xd = -np.sqrt(mu*a)/R*np.sin(E)
    yd = np.sqrt(mu*a*(1-e**2))/R*np.cos(E)

    r = np.matmul(C.T, np.array([x, y, 0]))
    v = np.matmul(C.T, np.array([xd, yd, 0]))

    return np.concatenate((r, v))

def _getOrbitElements(r0,rdot0,mu):
    '''
    Takes a cartesian state and converts it to the classical orbital elements
    
    Inputs: r0 - 2 OR 3 x 1 list/numpy array of the current state position, rdot0 - 2 OR 3 x 1 list/numpy array of the current state velocity, mu - gravitational parameter of body being orbited
    Output: returns a numpy array of the 6 orbital elements - [semi major axis, eccentricity, inclination, RAAN, long of periapsis, mean anomaly, semi-latus rectum]

    Created: 6/22/22
    Author: Hunter Quebedeaux
    '''
    if len(r0) == 2:
        r0 = np.append(r0,0)
        rdot0 = np.append(rdot0,0)
    R0 = np.linalg.norm(r0)
    Rdot0 = np.linalg.norm(rdot0)
    h = np.cross(r0,rdot0)
    H = np.linalg.norm(h)

    a = (2/R0 - (Rdot0 ** 2)/mu) ** -1

    p = H ** 2 / mu

    c = np.cross(rdot0,h) - (mu * r0) / R0

    e = np.linalg.norm(c)/mu

    C = np.zeros((3,3))
    i_h = h/H
    i_e = c / (mu*e)
    i_m = np.cross(i_h,i_e)

    C[2,:] = i_h
    C[0,:] = i_e
    C[1,:] = i_m


    i = np.arccos(C[2,2])
    Omega = np.arctan2(C[2,0],-C[2,1])
    omega = np.arctan2(C[0,2],C[1,2])

    sigma0 = (r0[0] * rdot0[0] + r0[1] * rdot0[1] + r0[2] * rdot0[2]) / np.sqrt(mu)

    E0 = np.arctan2(sigma0/np.sqrt(a),1-(R0/a))
    M0 = E0 - e*np.sin(E0)

    n = np.sqrt(mu)/(a ** 1.5)

    P = 2*np.pi/n

    OE = np.array([a,e,i,Omega,omega,M0,P])
    return OE

def classicOrbitProp(t,OE,mu,eps=1e-10,Nitr=5):
    '''
    Propagates an orbit using the classic orbital elements and keplers equation using mean anomoly
    Inputs: t - explicit time span series, OE - array of orbital elements, get from getOrbitalElements, mu - gravitational parameter of the body being orbited, eps - error passed to solve keplers equation, Nitr - number of iterations to solve keplers equation
    Outputs: returns the cartesian space position and velocity of the orbiting body

    Created: 6/23/22
    Author: Hunter Quebedeaux
    '''
    T = t
    a = OE[0]; e = OE[1]; i = OE[2]; Omega = OE[3]; omega = OE[4]; M0 = OE[5]
    COmega_3 = np.array([[np.cos(Omega),np.sin(Omega),0], [-np.sin(Omega),np.cos(Omega),0], [0,0,1]])
    Ci_1 = np.array([[1,0,0],[0,np.cos(i),np.sin(i)],[0,-np.sin(i),np.cos(i)]])
    Comega_3 = np.array([[np.cos(omega),np.sin(omega),0],[-np.sin(omega),np.cos(omega),0], [0,0,1]])   

    C = np.matmul(Comega_3,Ci_1); C = np.matmul(C,COmega_3) # COmega_3 * Ci_1 * COmega_3

    N = len(t)
    n = np.sqrt(mu/(a ** 3))

    X = np.zeros((N,6))

    for j in range(N):
        M = M0 + n*t[j]

        E,kepErr = SolveKeplerEq(M,e,eps,Nitr)

        x = a*(np.cos(E) - e)
        y = a*np.sqrt(1-(e ** 2))*np.sin(E)

        R = a*(1-e*np.cos(E))

        xd = -np.sqrt(mu*a)/R*np.sin(E)
        yd = np.sqrt(mu*a*(1-(e**2)))/R*np.cos(E)

        X[j,0:3] = np.transpose((np.matmul(np.transpose(C),np.array([[x],[y],[0]]))))
        X[j,3:6] = np.transpose((np.matmul(np.transpose(C),np.array([[xd],[yd],[0]]))))

    return T, X


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

def jacobiConstant(Y):
    m_1 = 5.974E24  # kg
    m_2 = 7.348E22 # kg
    mu = m_2/(m_1 + m_2)

    x = Y[:,0].reshape(-1,1)
    y = Y[:,1].reshape(-1,1)
    xdot = Y[:,2].reshape(-1,1)
    ydot = Y[:,3].reshape(-1,1)

    vSquared = (xdot**2 + ydot**2)
    xn1 = -mu
    xn2 = 1-mu
    rho1 = np.sqrt((x-xn1)**2+y**2)
    rho2 = np.sqrt((x-xn2)**2+y**2)

    C = (x**2 + y**2) + 2*(1-mu)/rho1 + 2*mu/rho2 - vSquared

    return C


if __name__ == "__main__":
    from scipy.integrate import solve_ivp
    import matplotlib.pyplot as plt
    from qutils.dynSys.dim6 import lagrangePlanEq
    r0 = np.array([6780,100,200]); #km
    rdot0 = np.array([-0.1,8.1,-0.2]) #km/s
    mu = 398600

    OEIC = ECI2OE(r0,rdot0,mu)
    XIC = OE2ECI(OEIC,0)
    print(OEIC)
    print(XIC)

    delT = 5
    nSamples = int(np.ceil(OEIC[-1] / delT))
    t = np.linspace(0,OEIC[-1],nSamples)

    rk45sol = solve_ivp(lagrangePlanEq, (0, OEIC[-1]), OEIC[:-1], method='RK45', rtol=1e-8, atol=1e-10)

    tElements = rk45sol.t
    odeElements = rk45sol.y.T

    
    plt.figure()
    plt.plot(tElements, odeElements[:, 0])
    plt.title('RAAN (W)')
    plt.figure()
    plt.plot(tElements, odeElements[:, 1])
    plt.title('Inclination (i)')
    plt.figure()
    plt.plot(tElements, odeElements[:, 2])
    plt.title('Argument of Periapsis (w)')
    plt.figure()
    plt.plot(tElements, odeElements[:, 3])
    plt.title('Semimajor Axis (a)')
    plt.figure()
    plt.plot(tElements, odeElements[:, 4])
    plt.title('Eccentricity (e)')
    plt.figure()
    plt.plot(tElements, odeElements[:, 5])
    plt.title('Mean Anomaly (M0)')
    plt.show()
