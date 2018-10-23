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
    def __init__(self, min_len, max_len, max_unk_pecent,
                 dtype=np.int32, persist=True):
        self.min_len = min_len
        self.max_len = max_len
        self.max_unk_percent = max_unk_pecent
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
            # Must have at least one valid stanza for whole song to be valid
            np_tokens, _ = self.load(filepath)
            if np.shape(np_tokens)[0] > 0:
                return True
            else:
                return False
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
        npfile = '%s.%s.npz' % (filepath, self.max_len)
        if self.persist and os.path.isfile(npfile):
            data = np.load(npfile)
            numpy_tokens = data['tokens'].astype(self.dtype)
            numpy_seq_lens = data['seq_lens'].astype(self.dtype)
        else:
            data = self.read(filepath)
            all_tokens = self.tokenize(data)
            # Filter stanzas
            # Keep all stanzas that are >= min length in length
            all_tokens = list(
                filter(lambda x: len(x) >= self.min_len, all_tokens))
            # Keep all stanzas that are <= max length in length
            all_tokens = list(
                filter(lambda x: len(x) <= self.max_len, all_tokens))
            # Keep all stanzas that have < max unk% of tokens
            all_tokens = list(filter(
                lambda x: self.get_unk_percent(x) < self.max_unk_percent,
                all_tokens))

            n_stanzas = len(all_tokens)
            numpy_tokens = np.zeros(
                (n_stanzas, self.max_len), dtype=self.dtype)
            numpy_seq_lens = np.array(list(
                map(lambda x: len(x), all_tokens)
            ))
            for i, stanza_tokens in enumerate(all_tokens):
                for j in range(min(self.max_len, len(stanza_tokens))):
                    numpy_tokens[i][j] = all_tokens[i][j]

            if self.persist:
                np.savez(npfile, tokens=numpy_tokens, seq_lens=numpy_seq_lens)

        return numpy_tokens, numpy_seq_lens
