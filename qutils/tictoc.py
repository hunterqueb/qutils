import time

class timer():
    def __init__(self):
        self.ticTime = time.perf_counter()
    def tic(self):
        self.ticTime = time.perf_counter()
    def toc(self): 
        finalTime = time.perf_counter()
        print("\n\tElapsed time is {:.4f} seconds.".format(finalTime - self.ticTime))
        return finalTime - self.ticTime
    def tocVal(self):
        self.tocTime = time.perf_counter() - self.ticTime
        return self.tocTime
    def getTocTime(self):
        return self.tocTime
    def tocStr(self,str):
        finalTime = time.perf_counter()
        print()
        print(str + "\tElapsed time is {:.4f} seconds.".format(finalTime - self.ticTime))
        return finalTime - self.ticTime
