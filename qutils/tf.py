from matplotlib.mlab import psd, csd

def tfe(x, y, *args, **kwargs):
   """estimate transfer function from x to y, see csd for calling convention"""
   return csd(y, x, *args, **kwargs) / psd(x, *args, **kwargs)
