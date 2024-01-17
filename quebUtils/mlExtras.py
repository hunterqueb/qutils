# need dataloader, 

import numpy as np

def findDecimalAccuracy(testingDataOutput,y_pred):
    '''
    Function to find the decimal accuracy of a prediction from a NN compared to the testing data output using a modified Llyod Heuristic
    Input: testingDataOutput - numpy array that defines the true output that the NN should be estimating
           y_pred - numpy array that defines the estimated output that the NN generates
    Output: prints to screen the average of both forward and backward integrations decimal accuracies
            returns the average forward and backward integrations decimal accuracies in a tuple

    Created: 6/15/2022
    Author: Hunter Quebedeaux
    '''
    # get some constants to find problem dimension and the number of testing sets
    problemDim = len(y_pred[0])
    numTestingSets = len(y_pred[:])
    # initialize the array that will store decimal accuracy
    decAcc = np.zeros_like(y_pred)
    for i in range(numTestingSets):
        for j in range(problemDim):
            # finds the decimal accuracy based on the LLyod Heuristic
            # av = np.log10(np.abs(testingDataOutput[i][j]/(testingDataOutput[i][j]-y_pred[i][j])))
            av = np.abs(np.log10(np.abs(1/(testingDataOutput[i][j]-y_pred[i][j]))))
            # if infinity, assign the highest accuracy possible for float64s
            # value of infinity is due to testingDataOutput - y_pred = 0 == log(someNumber/0) == infinity
            if av == np.inf:
                av = 15
            # if i == 0:
            #     av = 15 #enforce BC on each state - not needed generally
            decAcc[i][j] = av
    # initialize the fwd and bwd average lists
    avg = []

    # for a time agnostic NN, the estimations alternate between forward and backward in the list, so seperate averages based on this
    mse = (np.square(y_pred - testingDataOutput)).mean(axis=0)
    for i in range(problemDim):
            sAvg = np.average(decAcc[:,i])
            avg.append(sAvg)
            print('State {} Decimal Accuracy Avg: {}'.format(i+1,sAvg))

    for i in range(problemDim):
        print('MSE for State {}: {}'.format(i+1,mse[i]))
    avg = np.asarray(avg)
    # return values

    return decAcc, avg

