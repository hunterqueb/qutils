from torch import nn, cat, exp, autograd,stack, sin, cos, is_tensor,normal,optim, from_numpy,tensor
import numpy as np
from qutils.tictoc import timer
from qutils.ml.trialSolution import Sin
from qutils.integrators import ode45
from qutils.ml.extras import findDecAcc

def genTimeDomain(numSegments,pts,ptsPerSeg,tStart,tEnd):
    if int(pts) != int(np.sum(ptsPerSeg)):
        print("Warning! The Points are not equally divided!")
    numSegmentDiv = numSegments + 1
    segPts = np.linspace(tStart,tEnd,numSegmentDiv)
    ptArray = []
    for i in range(len(ptsPerSeg)):
        ptArray.append(np.linspace(segPts[i],segPts[i+1],int(ptsPerSeg[i])))

    return np.hstack(ptArray).reshape(-1,1)

class PINN():
    '''
    class definition for the unsupervised physics informed solution to a set of coupled first order differential equations

    inputs: problemDim - number of coupled first order differential equations
            numSegs - number of solution segments for the differnetial equations. 1 is the ususal number, more and you get a more accurate solution the more nonlinear your equations are
            device - pytorch training device
            netptr - a pytorch feedforward network
            netprtd - a pytorch feedforward network to approximate the derivative of the state that netptr approximates. enforcing a network that has an activation function that is the derivative of netptrs activtion function yields a better approximation
            
    '''
    def __init__(self,problemDim,device,netptr,netptrd) -> None:
        self.problemDim = problemDim
        self.device = device
        self.netptr = netptr
        self.netptrd = netptrd

    def __call__(self,printOutAcc = True):
        '''
        forward call of the class to return the time vector and propagated states generated from the neural network approximation for an entire trajectory. 
        must use self.setupNetwork,self.setupTrialSolution,self.setupConvervation, and self.train_ to use

        TODO - using numpy appending is slow, need to speed it up, probably output the torch tensor directly and not use np append

        inputs: printOutAcc - set to false to suppress the accuracy of each segment of the trajectory trained.
        '''

        if self.trained == False:
            print('Network is not trained')
            return

        yTest = np.empty((0, self.problemDim))

        for i in range(self.numSegs):
            y_predL = []
            for j in range(self.problemDim):
                y_pred = self.nets[i][j](self.T[i])
                y_predL.append(y_pred)
            y_pred = cat(y_predL,1)
            yTestSeg = self.trialSolutions[i](y_pred, self.T[i]).cpu().squeeze().detach().numpy()
            yTest = np.append((yTest), (yTestSeg), axis=0)
            if printOutAcc:
                print('\nSection {}'.format(i+1))
                findDecAcc(self.yTruthSeg[i], yTestSeg)
        return cat(self.T).cpu().numpy(), yTest

    def testEvaulation(self,T):
        '''
        forward call of the class to return the time vector and propagated states generated at the final time from the neural network approximation. goal of this work is to perform these calculations immedietely 
        must use self.setupNetwork,self.setupTrialSolution,self.setupConvervation, and self.train_ to use

        TODO - automagically select the desired segment based on the time input given to the function

        inputs: T - n,1 tensor of values to test the evaluation speed of the neural network. n is the number of "numerical integrations" to perform
        '''

        if self.trained == False:
            print('Network is not trained')
            return

        yTest = np.empty((0, self.problemDim))
        y_predL = []
        
        for j in range(self.problemDim):
            y_pred = self.nets[-1][j](T)
            y_predL.append(y_pred)
        y_pred = cat(y_predL,1)
        yTestSeg = self.trialSolutions[-1](y_pred, T).cpu().squeeze().detach().numpy()
        yTest = np.append((yTest), (yTestSeg), axis=0)
        return T, yTest



    def setupNetwork(self,input_size,hidden_size,output_size,learningRateList,dataSetSize,tParts):
        '''
        automatically sets up a number of networks depended on the number of segments first created when creating the PINN object

        inputs: input_size - number of parameters to set as the input for the NN
                hidden_size - number of hidden parameters to set in the hidden layer
                output_size - number of parameters to set as the output for the NN, should be 1 usually
                learningRateList - list of learning rates for each segment
                dataSetSize - number of training data points
                tParts - vector that breaks up the number of time domain into the desired number of segments 

        '''


        self.tParts = tParts
        self.numSegs = len(tParts) - 1

        self.nets = [[0 for _ in range(self.problemDim)] for _ in range(self.numSegs)]
        self.optimizers = [[0 for _ in range(self.problemDim)] for _ in range(self.numSegs)]
        self.trainingDataInputs = [0 for _ in range(self.numSegs)]

        for i in range(self.numSegs):
            for j in range(self.problemDim):
                # check which network you are constructing in the dimension to keep kinematic consistency
                if j < self.problemDim/2:
                    net = self.netptr(input_size, hidden_size,output_size).double()
                else:
                    net = self.netptrd(input_size, hidden_size,output_size).double()
                
                net.to(self.device)

                optimizer = optim.Adam(net.parameters(), lr=learningRateList[i])

                self.nets[i][j] = net
                self.optimizers[i][j] = optimizer

            self.trainingDataInputs[i] = np.reshape(np.linspace(tParts[i], tParts[i+1], dataSetSize),[dataSetSize,1])
            self.trainingDataInputs[i] = from_numpy(self.trainingDataInputs[i]).double().requires_grad_(True)
            self.trainingDataInputs[i] = self.trainingDataInputs[i].to(self.device)

    def setupTrialSolution(self,system,y0,trialSolutionFunc=Sin,dt=0.001):
        '''
        automatically sets up trial solutions that are dependent on the NN created by self.setupNetwork()

        inputs: system - coupled first order differential equations
                y0 - initial conditions for your integration
                trialSolutionFunc - the arbitrary function that creates phi, Sin is the best performing. qutils.trialSolution has other options
                dt - time steps for the t segments
        '''

        self.dt=dt
        t = []
        T = []
        yTruth = np.empty((0, self.problemDim))
        self.yTruthSeg = [0 for _ in range(self.numSegs)]
        y0Seg = y0

        self.trialSolutions = [0 for _ in range(self.numSegs)]

        for i in range(self.numSegs):
            self.trialSolutions[i] = trialSolutionFunc(y0Seg,from_numpy(y0Seg).to(self.device).reshape(1,-1),self.tParts[i], self.tParts[i+1])

            tSeg = np.linspace(self.tParts[i], self.tParts[i+1], int((self.tParts[i+1]-self.tParts[i])/self.dt))

            tSeg, self.yTruthSeg[i] = ode45(system, (self.tParts[i], self.tParts[i+1]), y0Seg, rtol=1e-8,atol=1e-10,t_eval=tSeg)
            
            T.append(tensor(np.reshape(tSeg, [len(tSeg), 1])).to(self.device))

            y0Seg = self.yTruthSeg[i][-1,:]

            yTruth = np.append((yTruth), (self.yTruthSeg[i]), axis=0)
            t = np.append((t),(tSeg))

            self.trialSolutions[i].setTime(T[i], tSeg)
            self.trialSolutions[i].setTruth(self.yTruthSeg[i])
        self.yTruth = yTruth
        self.T = T

    def setupConservation(self,doKinematicConsistency = True, orbitEnergy=None,jacobiConst=None):
        '''
        sets the conservation principles for the trial solution, kinematic consistency set on by default
        '''

        for i in range(self.numSegs):
            self.trialSolutions[i].setKC(doKinematicConsistency)
            if orbitEnergy is not None:
                self.trialSolutions[i].setOrbitEnergy(orbitEnergy)
            if jacobiConst is not None:
                self.trialSolutions[i].setOrbitEnergy(orbitEnergy)

    def setPlot(self,plotFunc):
        '''
        sets the output plot
        '''
        for i in range(self.numSegs):
            self.trialSolutions[i].setPlot(plotFunc)


    def trainAnalytical(self,systemTensor,criterion,epochList):
        '''
        train the pinn using analytical method. the shallow network selected should have a forwardDiff() method that is the time derivative of the network.

        inputs: systemTensor - coupled first order differential equations that output a pytorch tensor.
                criterion - loss function selected by user
                epochList - list of epochs for each segment to be trained
        '''
        for i in range(self.numSegs):
            print('\nTraining Segment {}'.format(i+1))
            trainForwardLagAna(self.nets[i], epochList[i], self.optimizers[i], criterion, self.trainingDataInputs[i], systemTensor, self.trialSolutions[i])
        self.trained = True

    def trainAuto(self,systemTensor,criterion,epochList,debugList):
        '''
        train the pinn using auto differentiation, not as accurate and kind of broken. needs to be updated to have the same support as the analytical version
        '''
        for i in range(self.numSegs):
            print('\nTraining Segment {}'.format(i+1))
            trainForwardLagAuto(self.nets[i], epochList[i], self.optimizers[i], criterion, self.trainingDataInputs[i], systemTensor, self.trialSolutions[i],debug=debugList[i])
        self.trained = True


    def setToTrain(self):
        '''
        sets the networks to train mode
        '''
        for segs in self.nets:
            for element in segs:
                element.train()

    def setToEvaluate(self):
        '''
        sets the networks to evaluation mode
        '''
        for segs in self.nets:
            for element in segs:
                element.eval()

    def getTrueSolution(self):
        return self.yTruth
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

