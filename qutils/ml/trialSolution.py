from numpy import exp as nexp
from torch import exp as texp
from numpy import sin as nsin
from torch import sin as tsin
from torch import cos as tcos
from numpy import pi as npi
from torch import norm, cat
from numpy import zeros
from torch import cross, sqrt
from torch import zeros as tzeros
from torch import ones as tones


class phi():
    """
    base class for the trial solution phi. this class holds all function information, convervation calculations, and forward evalations. 
    should not be used by itself and instead used by child  
    """
    def __init__(self,y0,y0Tensor,orbitEnergy, pendEnergy, jacobiConst, scaleSystem):
        self.y0 = y0
        self.y0Tensor = y0Tensor
        self.problemDim = len(y0)
        self.orbitEnergy = orbitEnergy
        self.pendEnergy = pendEnergy
        self.jacobiConst = jacobiConst
        self.isScaleSystem = scaleSystem
        self.KC = True

    def setOrbitEnergy(self,orbitEnergy):
        self.orbitEnergy = orbitEnergy
    def setJacobiConst(self,jacobiConst):
        self.jacobiConst = jacobiConst
    
    def setTruth(self, yTruth):
        self.yTruth = yTruth

    def setPlot(self, plotfuncptr):
        self.plot = plotfuncptr

    def plotPredictions(self, yTruth, yTest, epoch):
        self.plot(yTruth, yTest, epoch, self.t)

    def setKC(self,KC):
        self.KC = KC

    def KCLoss(self,criterion,estState,estDerivative):
        if self.KC == True:
            if self.problemDim == 2:
                lossK = criterion(estState[:,1],estDerivative[:,0])
                # lossK = criterion(dPhidt[:,0],phi[:,1])
            elif self.problemDim == 4:
                lossK = criterion(estState[:,2],estDerivative[:,0]) + criterion(estState[:,3],estDerivative[:,1])
                # lossK = criterion(dPhidt[:,0],phi[:,2]) + criterion(dPhidt[:,1],phi[:,3])
            elif self.problemDim == 6:
                lossK = criterion(estState[:,3],estDerivative[:,0]) + criterion(estState[:,4],estDerivative[:,1]) + criterion(estState[:,5],estDerivative[:,2])
            else: lossK = None
        else: lossK = None

        return lossK

    def conservationLoss(self, criterion, y):
        if self.jacobiConst is not None:
            return self.lossCR3BP(criterion, y)
        elif self.pendEnergy is not None:
            return self.lossPendEnergy(criterion,y)
        elif self.orbitEnergy is not None:
            return self.lossOrbitCons(criterion,y)
        else: return None
    def jacobiConstant(self,Y):
        m_1 = 5.974E24  # kg
        m_2 = 7.348E22 # kg
        mu = m_2/(m_1 + m_2)

        x = Y[:,0].reshape(-1,1)
        y = Y[:,1].reshape(-1,1)
        xdot = Y[:,2].reshape(-1,1)
        ydot = Y[:,3].reshape(-1,1)

        vSquared = (xdot**2 + ydot**2)
        xn1 = -mu
        xn2 = 1-mu
        rho1 = sqrt((x-xn1)**2+y**2)
        rho2 = sqrt((x-xn2)**2+y**2)

        C = (x**2 + y**2) + 2*(1-mu)/rho1 + 2*mu/rho2 - vSquared

        return C

    def penEnergyCalc(self,y):
        return (0.5 * y[:,0]**2 + 0.5 * y[:,1]**2 + 0.5 * y[:,0]**4).reshape(-1,1)

    def energyCalc(self, y):
        R = norm(y[:, 0:2], dim=1).reshape(1, -1).t()
        V = norm(y[:, 2:5], dim=1).reshape(1, -1).t()
        return pow(V, 2)/2 - (1/R)

    def angMomCalc(self,y):
        device = y.device
        zeros_column = tzeros((y.size(0), 1)).to(device)

        R = cat((y[:, 0:2],zeros_column),dim=1)
        V = cat((y[:, 2:5],zeros_column),dim=1)
        H_pred = cross(R,V, dim=1)

        return H_pred[:,-1].reshape(-1,1)

    def eccCalc(self,y):
        device = y.device
        zeros_column = tzeros((y.size(0), 1)).to(device)

        R = cat((y[:, 0:2],zeros_column),dim=1)
        V = cat((y[:, 2:5],zeros_column),dim=1)
        H_pred = cross(R,V, dim=1)

        e = cross(V,H_pred) - R / norm(R, dim=1).reshape(-1, 1)
        return norm(e,dim=1).reshape(-1,1)

    def aCalc(self,y):

        return -1/(2*self.energyCalc(y))

    def lossPendEnergy(self,criterion,y):
        device = y.device
        E_pred = self.penEnergyCalc(y)
        E = self.pendEnergy
        return criterion(E_pred, E*tones((y.size(0), 1)).to(device).double())

    def lossCR3BP(self,criterion,y):
        device = y.device

        C_pred = self.jacobiConstant(y)
        C = self.jacobiConst

        return criterion(C*tones((C_pred.size(0), 1)).to(device).double(), C_pred)

    def lossOrbitCons(self,criterion,y):
        device = y.device
        E = self.orbitEnergy[0]
        H = self.orbitEnergy[1]
        e = self.orbitEnergy[2]
        a = self.orbitEnergy[3]
        E_pred = self.energyCalc(y)
        H_pred = self.angMomCalc(y)
        e_pred = self.eccCalc(y)
        a_pred = self.aCalc(y)

        ELoss = criterion(E_pred, E*tones((y.size(0), 1)).to(device).double())
        HLoss = criterion(H_pred, H*tones((y.size(0), 1)).to(device).double())
        eLoss = criterion(e_pred, e*tones((y.size(0), 1)).to(device).double())
        aLoss = criterion(a_pred, a*tones((y.size(0), 1)).to(device).double())
        return ELoss


    # forward passes

    def evaluate(self,nets,T = None):
        if T is None:
            T = self.T
        problemDim = self.problemDim
        nSamples = T.size(0)
        yTest = zeros((nSamples, problemDim))

        y_predL = []
        for i in range(problemDim):
            y_pred = nets[i](T)
            y_predL.append(y_pred)
        y_pred = cat(y_predL,1)
        yTest = self(y_pred,T).cpu().squeeze().detach().numpy()
        return yTest
    
    def evaluateICs(self,nets,T = None):
        if T is None:
            T = self.T
        problemDim = self.problemDim
        nSamples = T.size(0)
        yTest = zeros((nSamples, problemDim))

        y_predL = []
        for i in range(problemDim):
            y_pred = nets[i](T)
            y_predL.append(y_pred)
        y_pred = cat(y_predL,1)
        yTest = self(y_pred,T[:, 0].reshape(-1,1),T[0, 1::]).cpu().squeeze().detach().numpy()
        return yTest
    
    # time scaling

    def setTime(self,T,t=None):
        self.T = T
        if self.isScaleSystem:
            self.T = self.scaleSystem(T)
        self.t = t
        if self.isScaleSystem and t is not None:
            self.t = self.scaleSystem(t)

    def scaleSystem(self,T):
        return (T - self.a)/(self.b-self.a)
    
    # DO NOT USE -- GROUND WORK FOR AN AGNOSTIC QUANITITY CONSERVATION METHOD, BUT ACCURACY IS AWFUL
        # IDK IF IT IS CAUSED BY THE FLOAT INACCURACIES FROM STORING IN A PYTHON LIST, BUT ITS BAD
        # USE LIKE THIS - in trialSolution instansiation, call method
        # -------
        # trialSolutions[i].conserve([jacobiConstant],[C0])
        # -------
        ## then replace conservation loss function with one below and it
        # GOAL IS TO NOT HAVE THESE FUNCTIONS DEFINED IN TRIAL SOLUTION CLASS
        
    # def conserve(self,funcArr:list,constArr:list):
    #     if len(funcArr) == len(constArr):
    #         self.funcArr = funcArr
    #         self.constArr = constArr
    #     else:
    #         print('Conserving functions are not equal to number of constants! Will not train with conservation loss!')
    #         self.funcArr == None
    # def conservationLoss(self, criterion, y):
    #     if self.funcArr is not None:
    #         for i in range(len(self.funcArr)):
    #             device = y.device
    #             conservingQuantinity_pred = self.funcArr[i](y)
    #             conservingQuantinityLoss = criterion(self.constArr[i]*tones((conservingQuantinity_pred.size(0), 1)).to(device).double(), conservingQuantinity_pred)
    #             return conservingQuantinityLoss
    #     else: return None



class Linear(phi):
    def __init__(self, y0, y0Tensor, tStart, tEnd, orbitEnergy:list=None, pendEnergy:float = None, jacobiConst:float = None, scaleSystem=False):
        super(Linear, self).__init__(y0, y0Tensor, orbitEnergy, pendEnergy, jacobiConst, scaleSystem)
        self.a = tStart
        self.b = tEnd
    def __call__(self,net,t,j):
        return net*(t-self.a) + self.y0[j]

    def __call__(self,net,t):
        return net*(t-self.a) + self.y0Tensor

    def __call__(self,net,t,IC=None):
        if IC is None:
            y0Tensor = self.y0Tensor
        else:
            y0Tensor = IC

        return net*(t-self.a) + y0Tensor

    def timeDerivative(self,N,dN,t):
        return N+t*dN
    
class Exp(phi):
    def __init__(self, y0, y0Tensor, tStart, tEnd, orbitEnergy:list =None, pendEnergy:float = None, jacobiConst:float = None, scaleSystem=False):
        super(Exp, self).__init__(y0, y0Tensor, orbitEnergy, pendEnergy, jacobiConst, scaleSystem)
        self.a = tStart
        self.b = tEnd
    def __call__(self,net,t,j):
        return net*(1-nexp(-(t-self.a))) + self.y0[j]

    def __call__(self,net,t,IC=None):
        if IC is None:
            y0Tensor = self.y0Tensor
        else:
            y0Tensor = IC
        return net*(1-texp(-(t-self.a))) + y0Tensor

    def timeDerivative(self,N,dN,t):
        return texp(-(t-self.a)) * N + (1-texp(-(t-self.a))) * dN

class Sin(phi):
    def __init__(self, y0, y0Tensor, tStart, tEnd, orbitEnergy:list =None, pendEnergy:float = None, jacobiConst:float = None, scaleSystem=False):
        super(Sin, self).__init__(y0, y0Tensor, orbitEnergy, pendEnergy, jacobiConst, scaleSystem)
        self.a = tStart
        self.b = tEnd
        self.tEnd = tEnd
    def __call__(self,net,t,j:int):
        return net*nsin(npi*t/(2*self.tEnd) - npi*self.a/(2*self.tEnd)) + self.y0[j]

    def __call__(self,net,t,IC=None):
        if IC is None:
            y0Tensor = self.y0Tensor
        else:
            y0Tensor = IC

        return net*tsin(npi*t/(2*self.tEnd) - npi*self.a/(2*self.tEnd)) + y0Tensor
    
    def timeDerivative(self,N,dN,t):
        return npi/(2*self.tEnd)*tcos(npi*t/(2*self.tEnd) - npi*self.a/(2*self.tEnd)) * N + tsin(npi*t/(2*self.tEnd) - npi*self.a/(2*self.tEnd)) * dN

