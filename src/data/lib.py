import numpy as np
from numpy.random import RandomState


def get_random(seed):
    if seed is not None:
        return RandomState(seed)
    else:
        return np.random
