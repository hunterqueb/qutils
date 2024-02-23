import time

class timer():
    def __init__(self):
        self.ticTime = time.perf_counter()
    def tic(self):
        self.ticTime = time.perf_counter()
    def toc(self): 
        print("\n\tElapsed time is {:.4f} seconds.".format(time.perf_counter() - self.ticTime))
    def tocVal(self):
        return time.perf_counter() - self.ticTime