#!/usr/bin/python3
"""A module for lyrics dataset loader.
"""
import logging
import string
import codecs

import nltk
import numpy as np
import re

from data.base_loader import Loader


log = logging.getLogger("few-shot")


def custom_tokenizer(str):
    """Return list of stanzas where each stanza contains list of tokens."""
    str = str.lower()
    # Remove lines indicating anything in brackets, like "[chorus:]"
    str = re.sub("[\[].*?[\]]", "", str)
    # Handle windows new spaces
    str = str.replace("\r", "\n")
    # Remove apostrophe's at end and beginning of words, like "fallin'"
    str = re.sub(r"'([^A-Za-z])", r"\1", str)
    str = re.sub(r"([^A-Za-z])'", r"\1", str)

    tokens = []
    for stanza in str.split("\n\n"):
        t = []
        for sent in stanza.split("\n"):
            s = nltk.word_tokenize(sent)
            if len(s) > 0:
                if s[-1] in [',', '.']:
                    del s[-1]

                t.extend(s)
                t.append("\n".encode('unicode_escape'))

        if len(t) > 0:
            tokens.append(t)

    return tokens


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

    def __init__(self, min_len, max_len, max_unk_percent,
                 metadata, tokenizer=custom_tokenizer,
                 persist=True, dtype=np.int32):
        super(LyricsLoader, self).__init__(
            min_len, max_len, max_unk_percent, dtype=dtype)
        self.tokenizer = tokenizer
        self.metadata = metadata
        self.persist = persist
        self.pruned = False

        self.word_to_id = {}
        self.word_to_cnt = {}
        self.id_to_word = {}
        self.highest_word_id = -1

        # read persisted word ids (if they exist)
        self.read_vocab()

    def is_song(self, filepath):
        return filepath.endswith('.txt')

    def prune(self, n):
        log.info('Vocab size before pruning: %d' % len(self.word_to_cnt.keys()))
        # Delete words that have less than n frequency
        for word in self.word_to_cnt.keys():
            if self.word_to_cnt[word] < n:
                del self.word_to_cnt[word]

        # Recreate word_to_id and id_to_word based on pruned dictionary
        self.word_to_id = {}
        self.id_to_word = {}
        for i, word in enumerate(self.word_to_cnt.keys()):
            self.word_to_id[word] = i
            self.id_to_word[i] = word

        self.highest_word_id = len(self.word_to_cnt.keys()) - 1
        self.pruned = True

        log.info('Vocab size after pruning: %d' % len(self.word_to_cnt.keys()))
        self.write_vocab()

    def read_vocab(self):
        if self.metadata.exists('word_ids.csv'):
            log.info('Loading lyrics metadata...')
            for line in self.metadata.lines('word_ids.csv'):
                row = line.rstrip('\n').split(',', 1)
                word_id = int(row[0])
                self.word_to_id[row[1]] = word_id
                self.id_to_word[word_id] = row[1]
                if word_id > self.highest_word_id:
                    self.highest_word_id = word_id

            self.pruned = True
            return True
        else:
            return False

    def write_vocab(self):
        log.info('writing lyrics metadata...')
        for word in self.word_to_id:
            self.metadata.write(
                'word_ids.csv',
                '%s,%s\n' % (self.word_to_id[word], word)
            )

    def read(self, filepath):
        """Read a file.

        Arguments:
            filepath (str): path to the lyrics file. e.g.
                "/home/user/lyrics_data/tool/lateralus.txt"
        """
        return ''.join(codecs.open(filepath, 'U', errors='ignore').readlines())

    def get_num_tokens(self):
        # +3 because:
        # (self.highest_word_id + 1) is unknown token
        # (self.highest_word_id + 2) is stop token
        # (self.highest_word_id + 3) is start token (but we don't include this
        # because we never predict start token)
        return self.highest_word_id + 3

    def get_unk_token(self):
        # unk token is self.highest_word_id + 1
        return self.highest_word_id + 1

    def get_stop_token(self):
        # stop token is self.highest_word_id + 2
        return self.highest_word_id + 2

    def get_start_token(self):
        # start token is self.highest_word_id + 3
        return self.highest_word_id + 3

    def get_unk_percent(self, list_of_tokens):
        unk_token = self.get_unk_token()
        num_unk = len(list(filter(lambda x: x == unk_token, list_of_tokens)))
        return float(num_unk) / len(list_of_tokens)

    def tokenize(self, raw_lyrics):
        """Turn a string of lyrics data into a numpy array of int "word" IDs.

        Arguments:
            raw_lyrics (str): Stringified lyrics data
        """
        all_tokens = []
        for stanza in self.tokenizer(raw_lyrics):

            stanza_tokens = []
            for token in stanza:
                if not self.pruned:
                    if token not in self.word_to_id:
                        self.highest_word_id += 1
                        self.word_to_id[token] = self.highest_word_id
                        self.word_to_cnt[token] = 0
                        self.id_to_word[self.highest_word_id] = token
                    else:
                        self.word_to_cnt[token] += 1

                    token_value = self.word_to_id[token]
                else:
                    if token not in self.word_to_id:
                        token_value = self.get_unk_token()
                    else:
                        token_value = self.word_to_id[token]

                stanza_tokens.append(token_value)

            all_tokens.append(stanza_tokens)

        return all_tokens

    def detokenize(self, numpy_data):
        ret = ''
        for token in numpy_data:
            if token == self.get_stop_token():
                ret += " [stop]"
                break
            elif token == self.get_start_token():
                ret += "[start]"
            elif token == self.get_unk_token():
                ret += " [unk]"
            else:
                word = self.id_to_word[token]
                if word == "n't":
                    ret += word
                elif word not in string.punctuation and not word.startswith("'"):
                    ret += " " + word
                elif word == "\n".encode('unicode_escape'):
                    ret += "\n"
                else:
                    ret += word
        return "".join(ret).strip()
