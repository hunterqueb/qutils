import numpy as np

from qutils.orbital import SolveKeplerEq

def lagrangePlanEq(t, elements, mu=398600):
# elements = [OMEGA i omega a e M0]
    delements = np.zeros(elements.shape)
    J2 = 1082.63*1e-6
    req = 6378
    OMEGA = elements[0]
    i = elements[1]
    omega = elements[2]
    a = elements[3]
    e = elements[4]
    M0 = elements[5]
    n = np.sqrt(mu/a**3)
    M = M0 + n*t
    E,Feval = SolveKeplerEq(M,e)
    f = 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2))
    theta = f + omega
    b = a*np.sqrt(1-e**2)
    p = a*(1-e**2)
    r = p/(1+e*np.cos(f))
    nue = np.sqrt(1-e**2)
    
    
    delements[0] = -3*J2*n*(a**2/(b*r))*((req/r)**2)*(np.sin(theta)**2)*np.cos(i)
    delements[1] = -(3/4)*J2*n*(a**2/(b*r))*((req/r)**2)*np.sin(2*theta)*np.sin(2*i)
    delements[2] = (3/2)*J2*n*(p/(r**2*e*nue**3))*((req/r)**2)*(2*r*e*(np.cos(i)**2)*(np.sin(theta)**2)-(p+r)*np.sin(f)*(np.sin(i)**2)*np.sin(2*theta)+p*np.cos(f)*(1-(3*np.sin(i)**2)*(np.sin(theta)**2)))
    delements[3] = -3*J2*n*(a**4/(b*r**2))*((req/r)**2)*(e*np.sin(f)*(1-3*(np.sin(theta)**2)*(np.sin(i))**2) + (p/r)*np.sin(2*theta)*(np.sin(i)**2))
    delements[4] = -(3/2)*J2*n*((a**2)/(b*r))*((req/r)**2)*((p/r)*np.sin(f)*(1-3*(np.sin(theta)**2)*(np.sin(i)**2)+(e+np.cos(f)*(2+e*np.cos(f))*np.sin(2*theta)*(np.sin(i)**2))))
    delements[5] = (3/2)*J2*n*(p/(e*(r*nue)**2)*((req/r)**2) * ((p+r)*np.sin(f)*(np.sin(i)**2)*np.sin(2*theta)+((2*r*e-p*np.cos(f))*(1-3*(np.sin(i)**2)*(np.sin(theta)**2)))))

    return delements


def three_body_prob(t, r,m=np.array(5.967e23,7.35e22,3.675e22)):

    G = 6.6743015e-11
    m1 = m[0]
    m2 = m[1]
    m3 = m[2]
    
    r1 = r[0:2]
    r2 = r[2:4]
    r3 = r[4:6]
    v1 = r[6:8]
    v2 = r[8:10]
    v3 = r[10:12]
    
    r12 = r2 - r1
    r23 = r3 - r2
    r13 = r3 - r1
    r21 = -r12
    r32 = -r23
    r31 = -r13
    
    drdt = np.zeros(12)
    drdt[0:6] = r[6:12]
    drdt[6:8] = G*((m2/(np.linalg.norm(r12)**3))*r12 + (m3/(np.linalg.norm(r13)**3))*r13)
    drdt[8:10] = G*((m1/(np.linalg.norm(r21)**3))*r21 + (m3/(np.linalg.norm(r23)**3))*r23)
    drdt[10:12] = G*((m1/(np.linalg.norm(r31)**3))*r31 + (m2/(np.linalg.norm(r32)**3))*r32)
    
    return drdt

def nondim_cr3bp(t, Y,mu):
    """Solve the CR3BP in nondimensional coordinates.
    
    The state vector is Y, with the first three components as the
    position of $m$, and the second three components its velocity.
    
    The solution is parameterized on $\\pi_2$, the mass ratio.
    """
    # Get the position and velocity from the solution vector
    x, y, z = Y[:3]
    xdot, ydot, zdot = Y[3:]

    # Define the derivative vector
    Ydot = np.zeros_like(Y)
    Ydot[:3] = Y[3:]

    sigma = np.sqrt(np.sum(np.square([x + mu, y, z])))
    psi = np.sqrt(np.sum(np.square([x - 1 + mu, y, z])))
    Ydot[3] = 2 * ydot + x - (1 - mu) * (x + mu) / sigma**3 - mu * (x - 1 + mu) / psi**3
    Ydot[4] = -2 * xdot + y - (1 - mu) * y / sigma**3 - mu * y / psi**3
    Ydot[5] = -(1 - mu)/sigma**3 * z - mu/psi**3 * z
    return Ydot

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp
    from qutils.dynSys.dim6 import *
    m = np.zeros(3)
    m[0] = 5.967e23
    m[1] = 7.35e22
    m[2] = 3.675e22

    # Equilateral Triangle Case
    G = 6.6743015e-11
    # meters
    rho = 1e9

    # m/s
    rdotMag = np.zeros(3)
    rdotMag[0] = 29.8659
    rdotMag[1] = 189.181
    rdotMag[2] = 195.552 

    M = np.sum(m)

    # m
    r12 = rho*np.array([1, 0])
    r23 = rho*np.array([-np.cos(np.pi/3), np.sin(np.pi/3)])
    r13 = rho*np.array([np.cos(np.pi/3), np.sin(np.pi/3)])

    # m
    r10 = (-m[1]*r12 - m[2]*r13)/M
    r20 = (m[0]*r12 - m[2]*r23)/M
    r30 = (m[0]*r13 + m[1]*r23)/M

    # frame vectors er and etheta for each mass.
    er1 = r10/np.linalg.norm(r10)
    et1 = np.array([-er1[1], er1[0]])
    er2 = r20/np.linalg.norm(r20)
    et2 = np.array([-er2[1], er2[0]])
    er3 = r30/np.linalg.norm(r30)
    et3 = np.array([-er3[1], er3[0]])

    r1dot0 = rdotMag[0]*(-np.cos(np.deg2rad(40))*er1+np.sin(np.deg2rad(40))*et1)
    r2dot0 = rdotMag[1]*(-np.cos(np.deg2rad(40))*er2+np.sin(np.deg2rad(40))*et2)
    r3dot0 = rdotMag[2]*(-np.cos(np.deg2rad(40))*er3+np.sin(np.deg2rad(40))*et3)

    eps = 1e-12
    opts = {'rtol': eps, 'atol': eps}

    initialConditions = np.concatenate((r10, r20, r30, r1dot0, r2dot0, r3dot0))
    tSpan = np.linspace(0, 3e7, 300)

    triSol = solve_ivp(three_body_prob,(0,3e7),initialConditions, **opts)
    T = triSol.t
    triSol = np.transpose(triSol.y)
    # plot
    plt.figure()
    plt.plot(triSol[:,0],triSol[:,1],label='Orbit of m1')
    plt.plot(triSol[:,2],triSol[:,3],label='Orbit of m2')
    plt.plot(triSol[:,4],triSol[:,5],label='Orbit of m3')
    # plotting equilateral triangles
    # plt.plot([triSol[0,0], triSol[0,2], triSol[0,4], triSol[0,0]],[triSol[0,1],triSol[0,3],triSol[0,5],triSol[0,1]], 'k-*',label='Equilateral Triangle at Initial Time')
    plt.plot([triSol[100,0], triSol[100,2], triSol[100,4], triSol[100,0]],[triSol[100,1],triSol[100,3],triSol[100,5],triSol[100,1]], 'k:*',label='Equilateral Triangle at an Arbitrary Time')

    # plot texts
    plt.title('Three Body Problem: Equilateral Triangle Solution')
    plt.legend()
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.grid()
    plt.axis('equal')
    plt.tight_layout()

    # collinear solution

    quinticRoots = np.roots([m[0]+m[1], 3*m[0]+2*m[1], 3*m[0]+m[1], -m[1]-3*m[2], -2*m[1]-3*m[2], -m[1]-m[2]])
    chi = np.real(quinticRoots[2])
    x12 = 1e9  # m
    x13 = (1 + chi) * x12
    x23 = x12 * chi

    # Calculate the initial positions of each body
    r10 = np.array([(-m[1]*x12 - m[2]*x13)/M, 0])
    r20 = np.array([(m[0]*x12 - m[2]*x23)/M, 0])
    r30 = np.array([(m[0]*x13 + m[1]*x23)/M, 0])

    r1dotMag = 32.9901  # m/s
    r2dotMag = 150.903  # m/s
    r3dotMag = 233.844  # m/s

    vc10 = r1dotMag * np.array([np.cos(np.deg2rad(40)), -np.sin(np.deg2rad(40))])

    f0 = 1

    er1 = r10 / np.linalg.norm(r10)
    er2 = r20 / np.linalg.norm(r20)
    er3 = r30 / np.linalg.norm(r30)
    etheta1 = np.array([0, -1])
    etheta2 = np.array([0, 1])
    etheta3 = np.array([0, 1])

    fdot0 = np.dot((vc10 / np.linalg.norm(r10)), er1)
    omega = np.dot((vc10 / np.linalg.norm(r10)), etheta1)

    v10 = (np.linalg.norm(r10) * fdot0) * er1 + (np.linalg.norm(r10) * f0 * omega) * etheta1
    v20 = (np.linalg.norm(r20) * fdot0) * er2 + (np.linalg.norm(r20) * f0 * omega) * etheta2
    v30 = (np.linalg.norm(r30) * fdot0) * er3 + (np.linalg.norm(r30) * f0 * omega) * etheta3

    initialConditions = np.concatenate((r10, r20, r30, v10, v20, v30))
    tSpan = np.linspace(0, 6e7, 601)
    eps = 1e-10
    opts = {'rtol': eps, 'atol': eps}

    colSol = solve_ivp(three_body_prob,(0,6e7), initialConditions, **opts)
    T = colSol.t
    colSol = np.transpose(colSol.y)

    # plot
    plt.figure()
    plt.plot(colSol[:,0], colSol[:,1],label='Orbit of m1')
    plt.plot(colSol[:,2], colSol[:,3],label='Orbit of m2')
    plt.plot(colSol[:,4], colSol[:,5],label='Orbit of m3')
    # plotting equilateral triangles
    # plt.plot([colSol[0,0], colSol[0,2], colSol[0,4]], [colSol[0,1],colSol[0,3],colSol[0,5]], 'k-',label='Collinear Configuration at Initial Time')
    plt.plot([colSol[100,0], colSol[100,2], colSol[100,4]], [colSol[100,1], colSol[100,3], colSol[100,5]], 'k:',label="Collinear Configuration at an Arbitrary Time")
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.grid()

    #CR3BP

    m_1 = 5.974E24  # kg
    m_2 = 7.348E22 # kg
    mu = m_2/(m_1 + m_2)

    x_0 = 1 - mu
    y_0 = .0455
    z_0 = 0
    vx_0 = 0.5
    vy_0 = -0.5
    vz_0 = 0

    mu = 0.012277471
    x_0 = 0.994
    y_0 = 0
    z_0 = 0
    vx_0 = 0
    vy_0 = -2.0317326295573368357302057924
    vz_0 = 0


    # Then stack everything together into the state vector
    r_0 = np.array((x_0, y_0, z_0))
    v_0 = np.array((vx_0, vy_0, vz_0))
    Y_0 = np.hstack((r_0, v_0))

    t_0 = 0  # nondimensional time
    t_f = 20  # nondimensional time
    t_points = np.linspace(t_0, t_f, 1000)

    eps = 1e-7
    opts = {'rtol': eps, 'atol': eps}


    sol = solve_ivp(nondim_cr3bp, [t_0, t_f], Y_0, t_eval=t_points,args=(mu,), **opts)

    Y = sol.y.T
    r = Y[:, :3]  # nondimensional distance
    v = Y[:, 3:]  # nondimensional velocity

    fig, ax = plt.subplots(figsize=(5,5), dpi=96)
    # Plot the orbits
    ax.plot(r[:, 0], r[:, 1], 'r', label="Trajectory")
    # ax.plot(np.hstack((x_2, x_2[::-1])), np.hstack((y_2, -y_2[::-1])))
    ax.plot(-mu, 0, 'bo', label="$Body 1$")
    ax.plot(1 - mu, 0, 'go', label="$Body 2$",markersize = 3)
    ax.plot(x_0, y_0, 'ro',label='$Body 3$',markersize = 2)
    ax.set_aspect("equal")
    ax.legend()
    ax.grid()
    plt.show()
