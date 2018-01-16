#!/usr/bin/python3
"""A module for lyrics and MIDI dataset loaders."""
import os 
import numpy as np


def tokenize_lyrics_file(filepath, max_len):
    """Turns a file into a list of "words"

    Arguments:
        filepath (str): path to the lyrics file. e.g.
            "/home/user/lyrics_data/tool/lateralus.txt"
        max_len (int): the maximum length
    """
    token_count = 0
    for line in open(filepath, 'r', errors='ignore'):
        for token in line.split():
            yield token
            token_count += 1
            if token_count >= max_len:
                return


def tokenize_midi_file(filepath):
    """Turns a MIDI file into a list of event IDs.

    Arguments:
        filepath (str): path to the lyrics file. e.g.
            "/home/user/freemidi_data/Tool/lateralus.mid"
    """
    midi_file = pretty_midi.PrettyMIDI(filepath)
    return None


class LyricsLoader(object):
    """A class which loads and parses lyrics files into a list of word IDs.

    Arguments:
        length (int): maximum length of tokens, after witch to truncate. Songs
            shorter than `length` are zero padded.
        tokenize (function): a function which takes a single string argument
            and returns a list of integer word IDs.
        persist_file_name: (None or str): if not None, causes LyricsLoader to
            load word IDs from and persist word IDs in the specified file.
    """
    def __init__(self, length, tokenize=tokenize_lyrics_file,
            persist_file_name=None, dtype=np.int32):
        self.length = length
        self.tokenize = tokenize
        self.word_ids = {}
        self.highest_word_id = -1
        self.dtype = dtype

        # read persisted word ids
        if persist_file_name is not None:
            for line in open(persist_file_name, 'r'):
                row = line.rstrip('\n').split(',', 1)
                word_id = int(row[0])
                self.word_ids[row[1]] = word_id
                if word_id > self.highest_word_id:
                    self.highest_word_id = word_id

        if persist_file_name is not None:
            self.persist_file = open(persist_file_name, 'w')
        else:
            self.persist_file = None

    def __call__(self, filepath):
        """This method takes some lyrics data and returns a list of integers
        word IDs for that lyrics data.

        Arguments:
            filepath (str): specifies the path to the file to load.
        """
        tokens = self.tokenize(filepath, self.length)
        word_ids = np.zeros(self.length, dtype=self.dtype)
        for token_index, token in enumerate(tokens):
            if token not in self.word_ids:
                self.highest_word_id += 1
                self.word_ids[token] = self.highest_word_id
                if self.persist_file is not None:
                    self.persist_file.write(
                        '%s,%s\n' % (self.highest_word_id, token))
            word_ids[token_index] = self.word_ids[token]
        return word_ids

    def close(self):
        self.persist_file.close()


class MIDILoader(object):
    """A class which loads and parses MIDI files into a list of event IDs.

    Arguments:
        length (int): The length of the return value of `load`. If the MIDI
            file is shorter than `length`, the remainder is zero padded.
        tokenize (function): takes a MIDI file path and returns a list of
            `length` length.
        dtype (numpy data type): the numpy data type of the array returned by
            `load`.
    """
    def __init__(self, length, tokenize=tokenize_midi_file, dtype=np.int32):
        self.length = length
        self.tokenize = tokenize
        self.dtype = dtype

    def __call__(self, filepath):
        """This method takes a MIDI file path and returns a list of integer
        event IDs.
        """
        raise NotImplementedError
