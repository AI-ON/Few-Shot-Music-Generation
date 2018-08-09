import numpy as np


class Sampler(object):

    def __init__(self, _type='random'):
        self._type = _type

    def sample(self, p):
        if self._type == 'random':
            return np.random.choice(len(p), p=p)
        elif self._type == 'argmax':
            return np.argmax(p)
        else:
            raise ValueError('Sample type %s not recognized' % self._type)
