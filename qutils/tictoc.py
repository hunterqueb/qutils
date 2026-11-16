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


class jetson_timer(timer):
    '''
    timer that also drives a GPIO pin on a Jetson board to mark a measured section
    for an external power-reading device. tic() sets the pin HIGH, any toc() sets it LOW.
    pin numbering uses GPIO.BOARD (physical header pin numbers).

    if cuda_sync is True (default) and torch reports CUDA available, torch.cuda.synchronize()
    is called before every pin change, so queued GPU work lands in the correct section.
    if torch is missing or CUDA is unavailable, syncing is silently skipped.


    example usage:
    from qutils.tictoc import jetson_timer

    with jetson_timer() as t:          # or jetson_timer(pin=11, cuda_sync=False)
        run_inference(model, warmup_batch)
        t.tic()                        # sync GPU, pin HIGH
        run_inference(model, data)
        t.toc()                        # sync GPU, pin LOW
    # leaving the block sets the pin LOW and releases it, even on an exception

    without `with`, wrap the measured section in try/finally so cleanup() always runs:
    t = jetson_timer()
    try:
        run_inference(model, warmup_batch)
        t.tic()
        run_inference(model, data)
        t.toc()
    finally:
        t.cleanup()                    # pin LOW + release, even if run_inference raises
    del t                              # optional: cleanup() already ran, so __del__ does nothing

    do not rely on `del t` or garbage collection alone; __del__ is a best-effort backup
    and may not run at interpreter shutdown.
    '''
    def __init__(self, pin=7, cuda_sync=True):
        import Jetson.GPIO as GPIO  # imported here so qutils.tictoc still imports off-Jetson
        self.GPIO = GPIO
        self.pin = pin
        self._torch = None
        if cuda_sync:
            try:
                import torch
                if torch.cuda.is_available():
                    self._torch = torch
            except ImportError:
                pass
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.LOW)
        self._cleaned = False  # set only after setup succeeds, so __del__ skips a failed init
        super().__init__()
    def _sync(self):
        if self._torch is not None:
            self._torch.cuda.synchronize()
    def tic(self):
        self._sync()
        self.GPIO.output(self.pin, self.GPIO.HIGH)
        super().tic()
    def toc(self):
        self._sync()
        self.GPIO.output(self.pin, self.GPIO.LOW)
        return super().toc()
    def tocVal(self):
        self._sync()
        self.GPIO.output(self.pin, self.GPIO.LOW)
        return super().tocVal()
    def tocStr(self,str):
        self._sync()
        self.GPIO.output(self.pin, self.GPIO.LOW)
        return super().tocStr(str)
    def cleanup(self):
        if self._cleaned:
            return
        # force the end mark first so an unfinished tic() can't leave the meter recording
        self.GPIO.output(self.pin, self.GPIO.LOW)
        self.GPIO.cleanup(self.pin)
        self._cleaned = True
    def __del__(self):
        # best effort only: not guaranteed to run at interpreter shutdown, prefer `with` or cleanup()
        if getattr(self, '_cleaned', True):
            return
        try:
            self.cleanup()
        except Exception:
            pass
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, tb):
        self.cleanup()
