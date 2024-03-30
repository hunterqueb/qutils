import matplotlib.pyplot as plt

def plotOrbitPhasePredictions(yTruth,legend=None):
    plt.figure()
    plt.plot(yTruth[:,0], yTruth[:,1],label=legend)
    plt.plot(0,0,'ko')
    plt.title('Orbit Solution')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('square')
    plt.tight_layout()
    plt.legend()
    plt.grid()

def plotCR3BPPhasePredictions(yTruth,yTest,epoch=None,t=None,L = 4,earth=True,moon=True):
    if L == False or L == None:
        L = 0
    if L == 4:
        L = [0.48784941,0.86602540]
    if L == 5:
        L = [0.48784941,-0.86602540]
    m_1 = 5.974E24  # kg
    m_2 = 7.348E22 # kg
    mu = m_2/(m_1 + m_2)
    plt.figure()
    plt.plot(yTruth[:,0], yTruth[:,1], label='Truth')
    plt.plot(yTest[:,0], yTest[:,1], label='NN')
    if earth:
        plt.plot(-mu,0,'ko',label='Earth')
    if moon:
        plt.plot(1-mu,0,'go',label='Moon')
    if L != 0:
        plt.plot(L[0],L[1],'d',color='grey',label='Lagrange Point')
    plt.title('Planar CR3BP')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('square')

    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)

    plt.tight_layout()
    plt.legend()
    plt.grid()
    if epoch == None:
        return
    else:
        plt.savefig('predict/predict%d.png' % epoch)
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
        
def plotSolutionErrors(yTruth,yTest,t,idxLen):
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
