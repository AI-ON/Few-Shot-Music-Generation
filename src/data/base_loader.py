#!/usr/bin/python3
"""A module for the parent loader class.
"""
import os
import numpy as np
import logging


log = logging.getLogger("few-shot")


class Loader(object):
    """A class for turning data into a sequence of tokens.
    """
    def __init__(self, max_len, dtype=np.int32, persist=True):
        self.max_len = max_len
        self.dtype = dtype
        self.persist = persist

    def is_song(self, filepath):
        raise NotImplementedError

    def read(self, filepath):
        raise NotImplementedError

    def tokenize(self, data):
        raise NotImplementedError

    def detokenize(self, numpy_data):
        raise NotImplementedError

    def get_num_tokens(self):
        raise NotImplementedError

    def validate(self, filepath):
        try:
            self.load(filepath)
            return True
        except OSError:
            return False
        except KeyError:
            return False
        except EOFError:
            return False
        except IndexError:
            return False
        except ValueError:
            return False
        except IOError:
            return False

    def load(self, filepath):
        npfile = '%s.%s.npy' % (filepath, self.max_len)
        if self.persist and os.path.isfile(npfile):
            return np.load(npfile).astype(self.dtype)
        else:
            data = self.read(filepath)
            tokens = self.tokenize(data)
            numpy_tokens = np.zeros(self.max_len, dtype=self.dtype)
            for token_index in range(min(self.max_len, len(tokens))):
                numpy_tokens[token_index] = tokens[token_index]
            if self.persist:
                np.save(npfile, numpy_tokens)
            return numpy_tokens
