import numpy as np

from map_module_f2py import map_module

def mapc2p(xc, yc):
    # Check if input are numpy arrays
    if not (isinstance(xc, np.ndarray) and isinstance(yc, np.ndarray)):
        # Then assume they are scalars
        xp, yp = map_module.mapc2p(xc, yc)
    else:
        # They are meshgrid arrays
        xp, yp = map_module.vectorized_mapc2p(xc, yc)
    return xp, yp