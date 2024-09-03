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

    if L != 0:
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

def plot3dOrbitPredictions(yTruth, yTest, epoch=None, t=None, earth=True,title="Two-Body Problem"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(yTruth[:, 0], yTruth[:, 1], yTruth[:, 2], label='Truth')
    ax.plot(yTest[:, 0], yTest[:, 1], yTest[:, 2], label='NN')

    if earth:
        ax.plot(0, 0, 0, 'ko', label='Earth')

    ax.set_title(title)
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


def plotOrbitPhasePredictions(yTruth,yTest,epoch=None,t=None,earth=True,plane = 'xy'):
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

    plt.figure()
    plt.plot(yTruth[:, x_idx], yTruth[:, y_idx], label='Truth')
    plt.plot(yTest[:, x_idx], yTest[:, y_idx], label='NN')

    if earth:
        plt.plot(0,0, 'ko', label='Earth')

    plt.title(f'Two-Body Problem: {title}')
    plt.xlabel(['x', 'z'][x_idx // 2])
    plt.ylabel(['y', 'z'][y_idx // 2])
    plt.axis('square')
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
    plt.tight_layout()
    plt.legend()
    plt.grid(True)

    if epoch is not None:
        plt.savefig(f'predict/predict{epoch}.png')
        plt.close()

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
    plt.ylabel(['y', 'z'][y_idx // 2])
    plt.axis('square')
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
    plt.tight_layout()
    plt.legend()
    plt.grid(True)

    if epoch is not None:
        plt.savefig(f'predict/predict{epoch}.png')
        plt.close()

def plotOrbitPredictions(yTruth,yTest,epoch=None,t=None):
    fig, axes = plt.subplots(nrows=2, ncols=2,constrained_layout=True)

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
        
def plotSolutionErrors(yTruth, yTest, t, units='km',states = ('x', 'y', 'z')):
    error = (yTruth - yTest)
    num_cols = error.shape[1]
    num_rows = int(num_cols / 2)
    
    fig, axes = plt.subplots(2, num_rows, figsize=(12, 6),constrained_layout=True)  # Change the subplots dimensions
    axes = axes.ravel()

    # automatically generate the state title labels for titling the plots
    if len(states) < 4:
        velStates = []
        for i in range(len(states)):
            velStates.append('\dot{'+states[i]+'}')
        state_labels = [states,velStates]
        
        posLabel = [units] * len(states)
        if units == 'DU':
            velLabel = 'TU'
        else:
            velLabel = [item + r'\s' for item in posLabel]


    if len(states) == 6:
        state_labels = np.reshape(states,(2,3))
        posLabel = units[0:3]
        velLabel = units[3:6]
    # handle the units labeling automatically - if DU, then use TU, if not append string with '/s'
    
    numPos = 0
    numVel = 0
    for i, ax in enumerate(axes[:num_cols]):
        ax.plot(t, error[:, i])
        ax.set_xlabel('t')
        if i < num_rows:
            ax.set_ylabel('Error ['+ posLabel[numPos] +']')
            ax.set_title(fr'$\mathrm{{Solution\ Error\ }} ({state_labels[0][numPos]})$', fontsize=10)
            numPos = numPos + 1
        else:
            ax.set_ylabel('Error ['+ velLabel[numVel] +']')
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


def plotStatePredictions(model,t,truth,train_in,test_in,train_size,lookback = 1, states = None,units=None,timeLabel = 'sec',DU = None, TU = None):
    from torch import no_grad
    with no_grad():
        # shift train predictions for plotting
        train_plot = np.ones_like(truth) * np.nan
        y_pred = model(train_in)
        y_pred = y_pred[:, -1, :]
        train_plot[lookback:train_size] = model(train_in)[:, -1, :].cpu()
        # shift test predictions for plotting
        test_plot = np.ones_like(truth) * np.nan
        test_plot[train_size+lookback:len(truth)] = model(test_in)[:, -1, :].cpu()

    problemDim = train_plot.shape[1]

    if states is not None:
        def nonDim2Dim(*args):
            return args[0] if len(args) == 1 else args

    if states == None and problemDim == 4:
        states = ['x', 'y', 'xdot', 'ydot']
        unitsDefault = ['km', 'km', 'km/s','km/s']
        from qutils.orbital import nonDim2Dim4 as nonDim2Dim

    elif states == None and problemDim == 6:
        states = ['x', 'y', 'z', 'xdot', 'ydot', 'zdot']
        unitsDefault = ['km', 'km','km', 'km/s', 'km/s','km/s']
        from qutils.orbital import nonDim2Dim6 as nonDim2Dim

    if units is None:
        units = unitsDefault

    paired_labels = [f'{label} ({unit})' for label, unit in zip(states, units)]

    if DU == None and TU == None:
        train_plot = nonDim2Dim(train_plot)
        test_plot = nonDim2Dim(test_plot)
    elif DU is not None and TU is not None:
        train_plot = nonDim2Dim(train_plot,DU,TU)
        test_plot = nonDim2Dim(test_plot,DU,TU)


    fig, axes = plt.subplots(2,problemDim // 2)

    for i, ax in enumerate(axes.flat):
        ax.plot(t, truth[:, i], c='b', label='True Motion')
        ax.plot(t, train_plot[:, i], c='r', label='Training Region')
        ax.plot(t, test_plot[:, i], c='g', label='Prediction')
        ax.set_xlabel('time ('+timeLabel+')')
        ax.set_ylabel(paired_labels[i])
        ax.grid()


    plt.legend(loc='upper left', bbox_to_anchor=(1,0.5))
    plt.tight_layout()
    
    from qutils.mlExtras import generateTrajectoryPrediction

    return generateTrajectoryPrediction(train_plot,test_plot)
