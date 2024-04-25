from torch import nn, cat, exp, autograd,stack, sin, cos, is_tensor,normal
import numpy as np
from qutils.tictoc import timer

def genTimeDomain(numSegments,pts,ptsPerSeg,tStart,tEnd):
    if int(pts) != int(np.sum(ptsPerSeg)):
        print("Warning! The Points are not equally divided!")
    numSegmentDiv = numSegments + 1
    segPts = np.linspace(tStart,tEnd,numSegmentDiv)
    ptArray = []
    for i in range(len(ptsPerSeg)):
        ptArray.append(np.linspace(segPts[i],segPts[i+1],int(ptsPerSeg[i])))

    return np.hstack(ptArray).reshape(-1,1)

class FeedforwardBase(nn.Module):
    '''
    Base Class to Define all feedforward networks. used to deploy training step with progress bar printed to terminal and plotting loss as a function of epochs

    Created: 4/15/2022
    Author: Hunter Quebedeaux
    '''
    def __init__(self):
        super(FeedforwardBase, self).__init__()
    
    def trainForwardLag(self, epochs, optimizer, criterion, trainingDataInput, system, trialSolution):
        '''
        forward training function: begins training based on criteron, optmizer, epochs and training data

        Inputs: epochs - number training epochs, optimizer - optimizer for training: defined based on torch.nn module,
        criterion - loss criterion for training: defined based on torch.nn module, 
        trainingDataInput - input data tensor for training, trainingDataOutput - output data tensor for training

        Created: 1/17/2023
        Author: Hunter Quebedeaux
        '''
        percent = epochs/10
        self.lossVect = []
        for epoch in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            # Forward pass
            y_pred = self(trainingDataInput)
            sum_ypred = y_pred.sum(0)
            y_predBackL = []
            # Differentiate forward pass wrt input data
            for i in range(len(sum_ypred)):
                y_predBackL.append(autograd.grad(sum_ypred[i].sum(),trainingDataInput,retain_graph=True)[0]) # needs to be computed automagically
            y_predBack = cat(y_predBackL,1)
            # Compute Loss
            loss = criterion(y_pred+trainingDataInput*y_predBack, system(trainingDataInput, trialSolution(y_pred,trainingDataInput))) # need to change this per problem. hardcoded right now as the solution for ex 1 lagaris
            print(end="\r                                                   ")
            print(end="\rEpoch {}: train loss: {}".format(epoch, loss.item()))
            if epoch % percent == 0 or epoch % (percent/2) == 0:
                print("")
            self.lossVect = np.append(self.lossVect, loss.item())
            # Backward pass
            loss.backward()
            optimizer.step()
        self.lossVect = np.delete(self.lossVect, 0)

class FeedforwardShallow(FeedforwardBase):
    '''
    Class to Define shallow feedforward network.

    Inputs: input_size - number of inputs to network, hidden_size: number of hidden nodes for single hidden layer, output_size - number of outputs to network

    Created: 4/15/2022
    Author: Hunter Quebedeaux
    '''
    def __init__(self, input_size, hidden_size,output_size):
        super(FeedforwardShallow, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size,bias=True) #linear
        self.nonlin = nn.Tanh() #nonlinear
        self.nonlinSig = nn.Sigmoid() #nonlinear
        self.fc2 = nn.Linear(self.hidden_size, self.output_size,bias=True) #linear
    def forward(self, x,return_dict=False):
        output = self.fc1(x)
        output = self.nonlinSig(output)
        output = self.fc2(output)
        return output
    def sigDiff(self,x):
        return exp(-x)/(1+exp(-x))**2
    def forwardDiff(self,x):
        output = self.fc1(x)
        output = (self.sigDiff(output) * self.fc1.weight[:,0] * self.fc2.weight).sum(1)
        return output
    
class FeedforwardSin(FeedforwardBase):
    '''
    Class to Define shallow feedforward network.

    Inputs: input_size - number of inputs to network, hidden_size: number of hidden nodes for single hidden layer, output_size - number of outputs to network

    Created: 4/15/2022
    Author: Hunter Quebedeaux
    '''
    def __init__(self, input_size, hidden_size,output_size):
        super(FeedforwardSin, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size,bias=True) #linear
        self.nonlinSig = sin  # nonlinear
        self.fc2 = nn.Linear(self.hidden_size, self.output_size,bias=True) #linear
    def forward(self, x,return_dict=False):
        output = self.fc1(x)
        output = self.nonlinSig(output)
        output = self.fc2(output)
        return output

    def forwardDiff(self,x):
        output = self.fc1(x)
        output = (cos(output) * self.fc1.weight[:,0] * self.fc2.weight).sum(1)
        return output

class FeedforwardCos(FeedforwardBase):
    '''
    Class to Define shallow feedforward network.

    Inputs: input_size - number of inputs to network, hidden_size: number of hidden nodes for single hidden layer, output_size - number of outputs to network

    Created: 4/15/2022
    Author: Hunter Quebedeaux
    '''
    def __init__(self, input_size, hidden_size,output_size):
        super(FeedforwardCos, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(self.input_size, self.hidden_size,bias=True) #linear
        self.nonlinSig = cos  # nonlinear
        self.fc2 = nn.Linear(self.hidden_size, self.output_size,bias=True) #linear
    def forward(self, x,return_dict=False):
        output = self.fc1(x)
        output = self.nonlinSig(output)
        output = self.fc2(output)
        return output

    def forwardDiff(self,x):
        output = self.fc1(x)
        output = (-sin(output) * self.fc1.weight[:,0] * self.fc2.weight).sum(1)
        return output

def trainForwardLagAuto(nets, epochs, optimizers, criterion, trainingDataInput, system, trialSolution):
    '''
    forward training function: begins training based on criteron, optmizer, epochs and training data

    Inputs: epochs - number training epochs, optimizer - optimizer for training: defined based on torch.nn module,
    criterion - loss criterion for training: defined based on torch.nn module, 
    trainingDataInput - input data tensor for training, trainingDataOutput - output data tensor for training

    Created: 1/26/2023
    Author: Hunter Quebedeaux
    '''
    percent = epochs/10
    lossVect = []
    for epoch in range(epochs):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        # Forward pass
        y_predL = []
        for i in range(len(nets)):
            y_pred = nets[i](trainingDataInput)
            y_predL.append(y_pred)
        y_pred = cat(y_predL,1)

        sum_ypred = y_pred.sum(0)
        y_predBackL = []
        # Differentiate forward pass wrt input data
        for i in range(len(sum_ypred)):
            y_predBackL.append(autograd.grad(sum_ypred[i].sum(),trainingDataInput,retain_graph=True)[0]) # needs to be computed automagically
        y_predBack = cat(y_predBackL,1)

        # Compute Loss
        dPhidt = trialSolution.timeDerivative(y_pred,y_predBack,trainingDataInput)
        loss = criterion(dPhidt, system(trainingDataInput, trialSolution(y_pred,trainingDataInput))) # need to change this per problem. hardcoded right now as the solution for ex 1 lagaris
        lossK = criterion(dPhidt[:,0],trialSolution(y_pred,trainingDataInput)[:,1])
        loss = loss + 10*lossK
        print(end="\r                                                   ")
        print(end="\rEpoch {}: train loss: {}".format(epoch, loss.item()))
        if epoch % percent == 0 or epoch % (percent/2) == 0:
            print("")
        lossVect = np.append(lossVect, loss.item())
        # Backward pass
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

    lossVect = np.delete(lossVect, 0)

def trainForwardLagAna(nets, epochs, optimizers, criterion, data, sys, trialSolution,debug = None):
    '''
    forward training function: begins training based on criteron, optmizer, epochs and training data

    Inputs: epochs - number training epochs, optimizer - optimizer for training: defined based on torch.nn module,
    criterion - loss criterion for training: defined based on torch.nn module, 
    trainingDataInput - input data tensor for training, trainingDataOutput - output data tensor for training

    Created: 1/26/2023
    Author: Hunter Quebedeaux
    '''

    try:
        if trialSolution.isScaleSystem:
            def system(tau,x):
                a = trialSolution.a
                b = trialSolution.b 
                return (b - a) * sys(a + (b - a) * tau,x)
        else:
            system = sys
        if is_tensor(data):
            data = [data]
        if debug is not None:
            yTruth = trialSolution.yTruth

        timeToTrain = timer()

        percent = epochs/10
        lossVect = []
        for epoch in range(epochs):
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            for batch in data:
                # Forward pass
                trainingDataInput = batch
                trainingDataInput = normal(mean=trainingDataInput, std=0.01)

                # if trialSolution.isScaleSystem: 
                #     t = trialSolution.scaleSystem(t)
                y_predL = []
                y_predBackL = []
                for i in range(len(nets)):
                    y_pred = nets[i](trainingDataInput)
                    y_predL.append(y_pred)
                    y_predBack = nets[i].forwardDiff(trainingDataInput)
                    y_predBackL.append(y_predBack)
                y_pred = cat(y_predL,1)
                y_predBack = stack(y_predBackL).T

                # Compute Loss
                if trainingDataInput.size(1) > 1:
                    IC = trainingDataInput[:, 1:trialSolution.problemDim+1]
                    t = trainingDataInput[:, 0:1]
                else:
                    IC = None
                    t = trainingDataInput

                phi = trialSolution(y_pred, t, IC)

                dPhidt = trialSolution.timeDerivative(y_pred,y_predBack,t)

                loss = criterion(dPhidt, system(t, phi)) # issue MUST be from this gradient,dphidt is in tau space, system takes input from t space, but phi is in tau space
                
                conservationLoss = trialSolution.conservationLoss(criterion, phi)
                if conservationLoss is None: pass
                else: loss = loss + 0.8*conservationLoss

                kinematicConsistencyLoss = trialSolution.KCLoss(criterion,phi,dPhidt)
                if kinematicConsistencyLoss is None: pass
                else: loss = loss + kinematicConsistencyLoss

                lossVect = np.append(lossVect, loss.item())
                # Backward pass
                loss.backward()
                for optimizer in optimizers:
                    optimizer.step()
            if epoch % percent == 0 or epoch % (percent/2) == 0 or epoch == epochs-1:
                print("")
                if debug is not None:
                    yTest = trialSolution.evaluate(nets)
                    trialSolution.plotPredictions(yTruth,yTest,epoch)
            print(end="\r                                                   ")
            print(end="\rEpoch {}: train loss: {}".format(epoch, loss.item()))

    except KeyboardInterrupt:
        print("\nQuitting at epoch {}".format(epoch))
        pass

    timeToTrain.toc()
    lossVect = np.delete(lossVect, 0)

