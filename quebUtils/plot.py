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
