#!/usr/bin/python3
"""A module for lyrics dataset loader.
"""
import logging
import string
import codecs

import nltk
import numpy as np

from data.base_loader import Loader


log = logging.getLogger("few-shot")


class LyricsLoader(Loader):
    """Objects of this class parse lyrics files and persist word IDs.

    Arguments:
        max_len (int): maximum length of sequence of words
        metadata (Metadata): a Metadata object
        tokenizer (callable): a callable which takes a file name and returns a
            list of words. Defaults to nltk's `word_tokenize`, which requires
            the punkt tokenizer models. You can download the models with
            `nltk.download('punkt')`
        persist (bool): if true, the tokenizer will persist the IDs of each word
            to a file. If the file already exists, the tokenizer will bootstrap
            from the file.
    """
    def __init__(self, max_len, metadata, tokenizer=nltk.word_tokenize,
            persist=True, dtype=np.int32):
        super(LyricsLoader, self).__init__(max_len, dtype=dtype)
        self.tokenizer = tokenizer
        self.metadata = metadata
        self.word_to_id = {}
        self.id_to_word = {}
        self.highest_word_id = -1
        # read persisted word ids
        if persist:
            log.info('Loading lyrics metadata...')
            for line in self.metadata.lines('word_ids.csv'):
                row = line.rstrip('\n').split(',', 1)
                word_id = int(row[0])
                self.word_to_id[row[1]] = word_id
                self.id_to_word[word_id] = row[1]
                if word_id > self.highest_word_id:
                    self.highest_word_id = word_id

    def is_song(self, filepath):
        return filepath.endswith('.txt')

    def read(self, filepath):
        """Read a file.

        Arguments:
            filepath (str): path to the lyrics file. e.g.
                "/home/user/lyrics_data/tool/lateralus.txt"
        """
        return ''.join(codecs.open(filepath, 'r', errors='ignore').readlines())

    def get_num_tokens(self):
        return self.highest_word_id + 1

    def tokenize(self, raw_lyrics):
        """Turns a string of lyrics data into a numpy array of int "word" IDs.

        Arguments:
            raw_lyrics (str): Stringified lyrics data
        """
        tokens = []
        for token in self.tokenizer(raw_lyrics):
            if token not in self.word_to_id:
                self.highest_word_id += 1
                self.word_to_id[token] = self.highest_word_id
                self.id_to_word[self.highest_word_id] = token
                if self.persist:
                    self.metadata.write(
                        'word_ids.csv',
                        '%s,%s\n' % (self.highest_word_id, token)
                    )
            tokens.append(self.word_to_id[token])
        return tokens

    def detokenize(self, numpy_data):
        ret = ''
        for token in numpy_data:
            word = self.id_to_word[token]
            if word == "n't":
                ret += word
            elif word not in string.punctuation and not word.startswith("'"):
                ret += " " + word
            else:
                ret += word
        return "".join(ret).strip()
