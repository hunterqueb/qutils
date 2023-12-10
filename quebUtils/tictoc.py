import time

class timer():
    def __init__(self):
        self.ticTime = time.time()
    def tic(self):
        self.ticTime = time.time()
    def toc(self): 
        print("\n\tElapsed time is {:.4f} seconds.".format(time.time() - self.ticTime))
    def tocVal(self):
        return time.time() - self.ticTime
