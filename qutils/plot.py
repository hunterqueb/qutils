import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot3dCR3BPPredictions(yTruth, yTest, epoch=None, t=None, L=4, earth=True, moon=True):
    if L == False or L == None:
        L = 0
    if L == 1:
        L = [0.8369154703225321, 0, 0]
    if L == 2:
        L = [1.1556818961296604, 0, 0]
    if L == 3:
        L = [-1.0050626166357435, 0, 0]
    if L == 4:
        L = [0.48784941, 0.86602540, 0]
    if L == 5:
        L = [0.48784941, -0.86602540, 0]

    m_1 = 5.974E24  # Mass of Earth in kg
    m_2 = 7.348E22  # Mass of Moon in kg
    mu = m_2 / (m_1 + m_2)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(yTruth[:, 0], yTruth[:, 1], yTruth[:, 2], label='Truth')
    ax.plot(yTest[:, 0], yTest[:, 1], yTest[:, 2], label='NN')

    if earth:
        ax.plot(-mu, 0, 0, 'ko', label='Earth')
    if moon:
        ax.plot(1 - mu, 0, 0, 'go', label='Moon')

    if L is not 0:
        ax.plot([L[0]], [L[1]], [L[2]], 'd', color='grey', label='Lagrange Point')

    ax.set_title('Cislunar CR3BP')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
    ax.set_box_aspect(np.ptp(limits, axis=1))
    ax.legend()
    ax.grid(True)

    if epoch is not None:
        plt.savefig(f'predict/predict{epoch}.png')
        plt.close()


def plotOrbitPhasePredictions(yTruth,legend=None):
    plt.plot(yTruth[:,0], yTruth[:,1],label=legend)
    plt.plot(0,0,'ko')
    plt.title('Orbit Solution')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('square')
    plt.tight_layout()
    plt.legend()
    plt.grid()

def plotCR3BPPhasePredictions(yTruth,yTest,epoch=None,t=None,L = 4,earth=True,moon=True,plane = 'xy'):
    if L == False or L == None:
        L = 0
    if L == 1:
        L = [0.8369154703225321,0,0]
    if L == 2:
        L = [1.1556818961296604,0,0]
    if L == 3:
        L = [-1.0050626166357435,0,0]
    if L == 4:
        L = [0.48784941,0.86602540,0]
    if L == 5:
        L = [0.48784941,-0.86602540,0]

    if plane == 'xy':
        x_idx, y_idx = 0, 1
        title = 'XY Plane'
    elif plane == 'xz':
        x_idx, y_idx = 0, 2
        title = 'XZ Plane'
    elif plane == 'yz':
        x_idx, y_idx = 1, 2
        title = 'YZ Plane'
    else:
        raise ValueError("Invalid plane selection. Choose 'xy', 'xz', or 'yz'.")

    m_1 = 5.974E24  # Mass of Earth in kg
    m_2 = 7.348E22  # Mass of Moon in kg
    mu = m_2 / (m_1 + m_2)

    plt.figure()
    plt.plot(yTruth[:, x_idx], yTruth[:, y_idx], label='Truth')
    plt.plot(yTest[:, x_idx], yTest[:, y_idx], label='NN')

    if earth:
        plt.plot(-mu if x_idx == 0 else 0, 0 if y_idx in [1, 2] else -mu, 'ko', label='Earth')
    if moon:
        plt.plot((1 - mu) if x_idx == 0 else 0, 0 if y_idx in [1, 2] else (1 - mu), 'go', label='Moon')
    if L is not 0:
        plt.plot(L[x_idx], L[y_idx], 'd', color='grey', label='Lagrange Point')

    plt.title(f'Planar CR3BP: {title}')
    plt.xlabel(['x', 'z'][x_idx // 2])
    plt.ylabel(['y', 'z'][y_idx % 2])
    plt.axis('square')
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
    plt.tight_layout()
    plt.legend()
    plt.grid(True)

    if epoch is not None:
        plt.savefig(f'predict/predict{epoch}.png')
        plt.close()

def plotOrbitPredictions(yTruth,yTest,epoch=None,t=None):
    fig, axes = plt.subplots(nrows=2, ncols=2,layout='constrained')

    # Plot the data in each subplot
    axes[0, 0].plot(t, yTruth[:,0],label = 'Truth')
    axes[0, 1].plot(t, yTruth[:,1],label = 'Truth')
    axes[1, 0].plot(t, yTruth[:,2],label = 'Truth')
    axes[1, 1].plot(t, yTruth[:,3],label = 'Truth')

    axes[0, 0].plot(t, yTest[:,0],label = 'NN')
    axes[0, 1].plot(t, yTest[:,1],label = 'NN')
    axes[1, 0].plot(t, yTest[:,2],label = 'NN')
    axes[1, 1].plot(t, yTest[:,3],label = 'NN')

    axes[0, 0].set_ylabel('x')
    axes[0, 1].set_ylabel('y')
    axes[1, 0].set_ylabel('xdot')
    axes[1, 1].set_ylabel('ydot')


    # Set legends for each subplot
    axes[0, 0].legend()
    axes[0, 1].legend()
    axes[1, 0].legend()
    axes[1, 1].legend()

    # Set legends for each subplot
    axes[0, 0].grid()
    axes[0, 1].grid()
    axes[1, 0].grid()
    axes[1, 1].grid()

    # Set titles for each subplot
    axes[0, 0].set_title('State 1')
    axes[0, 1].set_title('State 2')
    axes[1, 0].set_title('State 3')
    axes[1, 1].set_title('State 4')

    # Adjust spacing between subplots to avoid overlap
    fig.tight_layout()

    # Show the plot
    if epoch == None:
        return
    else:
        plt.savefig('predict/predict%d.png' % epoch)
        plt.close()
        
def plotSolutionErrors(yTruth, yTest, t, idxLen=None, units='km',states = ('x', 'y', 'z')):
    error = (yTruth - yTest)
    num_cols = error.shape[1]
    num_rows = int(num_cols / 2)
    
    fig, axes = plt.subplots(2, num_rows, figsize=(12, 6),layout='constrained')  # Change the subplots dimensions
    axes = axes.ravel()
    
    # handle the units labeling automatically - if DU, then use TU, if not append string with '/s'
    posLabel = units
    if units == 'DU':
        velLabel = 'TU'
    else:
        velLabel = units + '/s'
    
    # automatically generate the state title labels for titling the plots
    velStates = []
    for i in range(len(states)):
        velStates.append('\dot{'+states[i]+'}')
    state_labels = [states,velStates]

    numPos = 0
    numVel = 0
    for i, ax in enumerate(axes[:num_cols]):
        ax.plot(t, error[:, i])
        ax.set_xlabel('t')
        if i < num_rows:
            ax.set_ylabel('Error ['+ posLabel +']')
            ax.set_title(fr'$\mathrm{{Solution\ Error\ }} ({state_labels[0][numPos]})$', fontsize=10)
            numPos = numPos + 1
        else:
            ax.set_ylabel('Error ['+ velLabel +']')
            ax.set_title(fr'$\mathrm{{Solution\ Error\ }} ({state_labels[1][numVel]})$', fontsize=10)
            numVel = numVel + 1
        ax.grid()

def plotEnergy(yTruth,yTest,t,energyFunc,xLabel = 'Time (TU)',yLabel = 'Energy'):
    ETruth = energyFunc(yTruth)
    ETest = energyFunc(yTest)

    plt.figure()
    plt.plot(t,ETruth,label='Truth')
    plt.plot(t,ETest,label='NN')
    plt.title('Conserved Quantity')
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.legend()
    plt.grid()