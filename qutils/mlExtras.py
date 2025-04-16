# need dataloader, 

import numpy as np
import torch
import matplotlib.pyplot as plt
from qutils.mamba import Mamba

def findDecAcc(testingDataOutput,y_pred,printOut=True):
    '''
    Function to find the decimal accuracy of a prediction from a NN compared to the testing data output using a modified Llyod Heuristic

    Support for LSTM networks using a lookback window, or standard NN output that im used to
    
    Input: testingDataOutput - tensor or numpy array that defines the true output that the NN should be estimating
           y_pred - tensor or numpy array that defines the estimated output that the NN generates
           both inputs must be of the same type
    Output: prints to screen the average of decimal accuracy states and returns the array

    Created: 1/19/2024
    Author: Hunter Quebedeaux
    '''

    if isinstance(y_pred, np.ndarray):
        arrayType = 'numpy'
    elif isinstance(y_pred, torch.Tensor):
        arrayType = 'torch'
    
    # check if the input tensors are in LSTM lookback format
    if len(testingDataOutput.shape) == 3: 
        testingDataOutput = testingDataOutput[:,-1,:]   
        y_pred = y_pred[:,-1,:]
    else:
        pass

    # get some constants to find problem dimension and the number of testing sets
    # initialize the array that will store decimal accuracy
            # finds the decimal accuracy based on the LLyod Heuristic
            # av = np.log10(np.abs(testingDataOutput[i][j]/(testingDataOutput[i][j]-y_pred[i][j])))
    if arrayType == 'torch':
        decAcc = torch.abs(torch.log10(torch.abs(1/(testingDataOutput-y_pred))))
        inf_mask = torch.isinf(decAcc)

    if arrayType == 'numpy':
        decAcc = np.abs(np.log10(np.abs(1/(testingDataOutput-y_pred))))
        inf_mask = np.isinf(decAcc)
            # if infinity, assign the highest accuracy possible for float64s
            # value of infinity is due to testingDataOutput - y_pred = 0 == log(someNumber/0) == infinity
    decAcc[inf_mask] = 15

    # initialize the fwd and bwd average lists
    if arrayType == 'torch':
        avg = decAcc.mean(axis=0).cpu().numpy()
        error = (testingDataOutput-y_pred).cpu().numpy()
    if arrayType == 'numpy':
        avg = decAcc.mean(axis=0)
        error = (testingDataOutput-y_pred)

    if printOut == True:
        problemDim = len(avg)
        for i in range(problemDim):
                print('State {} Decimal Accuracy Avg: {}'.format(i+1,avg[i]))
    
    # return values
    return avg, error

def mse(y_truth, y_pred,output='full'):
    mse = np.nanmean((y_truth - y_pred)**2,axis=0)
    
    if output == 'single':
        mseVal = np.mean(mse)
        mse = np.array([mseVal])
    elif output == 'dynamicalSystem':
        mid_idx = len(mse) // 2 
        pos = np.mean(mse[:mid_idx])
        vel = np.mean(mse[mid_idx:])
        mse = np.array((pos,vel))
    print("\nMean Square Error")
    for i, avg in enumerate(mse, 1):
        print(f"Dimension {i}: {avg}")

    return mse

def rmse(y_truth, y_pred,output='full',percentRMSE=False):
    if percentRMSE:
        error = (y_truth - y_pred)/y_truth
    else:
        error = y_truth - y_pred
    mse = np.sqrt(np.nanmean((error)**2,axis=0))
    
    if output == 'single':
        mseVal = np.mean(mse)
        mse = np.array([mseVal])
    elif output == 'dynamicalSystem':
        mid_idx = len(mse) // 2 
        pos = np.mean(mse[:mid_idx])
        vel = np.mean(mse[mid_idx:])
        mse = np.array((pos,vel))
    print("\nRoot Mean Square Error")
    for i, avg in enumerate(mse, 1):
        print(f"Dimension {i}: {avg}")

    return mse


def generateTrajectoryPrediction(train_plot,test_plot,outputToc = False):
    '''
    takes matrices of two equal lengths and compares the values element by element. 
    if a number occupys one matrix but not the other return a new matrix with the nonzero value.
    if a number occupies both matrics then the value is prefered from the testing output / prediction
    if both matrices have nan, a new matrix is returned with the nan value.
    '''
    trajPredition = np.zeros_like(train_plot)
    from qutils.tictoc import timer

    solTime = timer()
    for i in range(test_plot.shape[0]):
        for j in range(test_plot.shape[1]):
            # Check if either of the matrices has a non-nan value at the current position
            if not np.isnan(test_plot[i, j]) or not np.isnan(train_plot[i, j]):
                # Choose the non-nan value if one exists, otherwise default to test value
                trajPredition[i, j] = test_plot[i, j] if not np.isnan(test_plot[i, j]) else train_plot[i, j]
            else:
                # If both are nan, set traj element to nan
                trajPredition[i, j] = np.nan

    elapsedTime = solTime.tocVal()
    if outputToc:
        print("Network Solution Generation Time: ",elapsedTime)
        return trajPredition, elapsedTime
    else:
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

def printoutMaxLayerWeight(model):
    if isinstance(model,Mamba):
        for param in model.named_parameters():
            print("Weight Tensor Name: ", param[0])  # Name of the parameter
            # Flatten the tensor to find the maximum index
            flat_tensor = param[1].flatten()
            flat_abs_tensor = flat_tensor.abs()
            flat_index = flat_abs_tensor.argmax()   
            max_abs_value = flat_tensor[flat_index]
            print("Maximum Weight Value: ", max_abs_value.item())
            # Calculate the original multi-dimensional index manually
            original_shape_index = (param[1]==torch.max(torch.abs(param[1]))).nonzero()
            print("Maximum Weight Index (original shape): ", original_shape_index)
            print()
    else:
        print("Model is not a mamba model. Returning...")
        return

def getSuperWeight(model):
    if isinstance(model,Mamba):
        highest = [None,0,None]
        for param in model.named_parameters():
            flat_tensor = param[1].flatten()
            flat_abs_tensor = flat_tensor.abs()
            flat_index = flat_abs_tensor.argmax()   
            max_abs_value = flat_tensor[flat_index]
            if abs(highest[1]) < abs(max_abs_value.item()):
                highest[0] = param[0]
                highest[1] = max_abs_value.item()
                highest[2] = (param[1]==torch.max(torch.abs(param[1]))).nonzero()
                # Calculate the original multi-dimensional index manually
        print("Layer with Superweight",highest)
        return highest[1]
    else:
        print("Model is not a mamba model. Returning...")
        return

def plotSuperWeight(model,newPlot=True):
    if newPlot:
        plt.figure()
    i = 1
    maxVal = []
    t = []
    for param in model.named_parameters():
        flat_tensor = param[1].flatten()
        flat_abs_tensor = flat_tensor.abs()
        flat_index = flat_abs_tensor.argmax()   
        max_abs_value = flat_tensor[flat_index].item()

        maxVal.append(max_abs_value)
        t.append(i)
        i += 1
    plt.plot(t,maxVal)
    plt.xlabel("Learnable Tensor")
    plt.ylabel("Maximum Numerical Value")
    plt.tight_layout()
    plt.grid()

def plotMinWeight(model,newPlot=True):
    if newPlot:
        plt.figure()
    i = 1
    minVal = []
    t = []
    for param in model.named_parameters():
        flat_tensor = param[1].flatten()
        flat_abs_tensor = flat_tensor.abs()
        flat_index = flat_abs_tensor.argmin()   
        min_abs_value = flat_tensor[flat_index].item()

        minVal.append(min_abs_value)
        t.append(i)
        i += 1
    plt.plot(t,minVal)
    plt.xlabel("Learnable Tensor")
    plt.ylabel("Minimum Numerical Value")
    plt.tight_layout()
    plt.grid()
    return

def getQ2NormMatrix(y_true, y_pred):
    y_mean = np.mean(y_true)
    # Numerator: Sum of squared prediction errors
    SS_res = np.sum((y_true - y_pred)**2)
    # Denominator: Sum of squared deviations from the mean
    SS_tot = np.sum((y_true - y_mean)**2)
    # Q^2-norm calculation
    Q2 = 1 - (SS_res / SS_tot)
    return Q2

def getQ2Norm(y_true,y_pred,dimension=0):
    #dimension = 0 == row
    #dimension = 1 == column
    Q2_vect = -np.inf * np.ones((y_true.shape[dimension],1))
    if dimension == 0:
        for i in range(len(Q2_vect)):
            Q2_vect[i] = getQ2NormMatrix(y_true[i,:],y_pred[i,:])
    elif dimension == 1:
        for i in range(len(Q2_vect)):
            Q2_vect[i] = getQ2NormMatrix(y_true[:,i],y_pred[:,i])
    return Q2_vect