# need dataloader, 

import numpy as np
import torch

def findDecAcc(testingDataOutput,y_pred,printOut=True):
    '''
    Function to find the decimal accuracy of a prediction from a NN compared to the testing data output using a modified Llyod Heuristic

    Support for LSTM networks using a lookback window, or standard NN output that im used to
    
    Input: testingDataOutput - tensor that defines the true output that the NN should be estimating
           y_pred - tensor that defines the estimated output that the NN generates
    Output: prints to screen the average of decimal accuracy states and returns the array

    Created: 1/19/2024
    Author: Hunter Quebedeaux
    '''

    # check if the input tensors are in LSTM lookback format
    if testingDataOutput.dim() == 3: 
        testingDataOutput = testingDataOutput[:,-1,:]   
        y_pred = y_pred[:,-1,:]
    else:
        pass

    # get some constants to find problem dimension and the number of testing sets
    # initialize the array that will store decimal accuracy
            # finds the decimal accuracy based on the LLyod Heuristic
            # av = np.log10(np.abs(testingDataOutput[i][j]/(testingDataOutput[i][j]-y_pred[i][j])))
    decAcc = torch.abs(torch.log10(torch.abs(1/(testingDataOutput-y_pred))))
            # if infinity, assign the highest accuracy possible for float64s
            # value of infinity is due to testingDataOutput - y_pred = 0 == log(someNumber/0) == infinity
    inf_mask = torch.isinf(decAcc)
    decAcc[inf_mask] = 15

    # initialize the fwd and bwd average lists
    avg = decAcc.mean(axis=0).cpu().numpy()

    if printOut == True:
        problemDim = len(avg)
        for i in range(problemDim):
                print('State {} Decimal Accuracy Avg: {}'.format(i+1,avg[i]))
    
    # return values
    return avg, (testingDataOutput-y_pred).cpu().numpy()

def generateTrajectoryPrediction(train_plot,test_plot):
    '''
    takes matrices of two equal lengths and compares the values element by element. 
    if a number occupys one matrix but not the other return a new matrix with the nonzero value.
    if a number occupies both matrics then the value is prefered from the testing output / prediction
    if both matrices have nan, a new matrix is returned with the nan value.
    '''
    trajPredition = np.zeros_like(train_plot)

    for i in range(test_plot.shape[0]):
        for j in range(test_plot.shape[1]):
            # Check if either of the matrices has a non-nan value at the current position
            if not np.isnan(test_plot[i, j]) or not np.isnan(train_plot[i, j]):
                # Choose the non-nan value if one exists, otherwise default to test value
                trajPredition[i, j] = test_plot[i, j] if not np.isnan(test_plot[i, j]) else train_plot[i, j]
            else:
                # If both are nan, set traj element to nan
                trajPredition[i, j] = np.nan

    return trajPredition


def __findDecimalAccuracyOLD(testingDataOutput,y_pred):
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

