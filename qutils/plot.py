import matplotlib.pyplot as plt

def plot3dOrbit(yTruth,yTest,earth=True,moon=True):
    return


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
    fig, axes = plt.subplots(nrows=2, ncols=2)

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
        
def plotSolutionErrors(yTruth,yTest,t,idxLen=None):
    error = (yTruth-yTest)
    num_cols = error.shape[1]
    num_rows = int(num_cols / 2)

    fig, axes = plt.subplots(num_rows, 2, figsize=(12, 6))
    axes = axes.ravel()

    for i, ax in enumerate(axes[:num_cols]):
        ax.plot(t, error[:, i])
        ax.set_title(f'Solution Error (State {i+1})')
        ax.set_xlabel('t')
        if i < num_rows:
            ax.set_ylabel('Error [km]')
        else:
            ax.set_ylabel('Error [km/s]')
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