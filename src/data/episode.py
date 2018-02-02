#!/usr/bin/python3
import os
import time
import logging

import numpy as np


class Episode(object):
    def __init__(self, support, query):
        self.support = support
        self.query = query


class SQSampler(object):
    """A sampler for randomly sampling support/query sets.

    Arguments:
        root (str): the root of the data directory
        support_size (int): number of songs in the support set
        query_size (int): number of songs in the query set
        cache (bool): caches the song names instead of hitting the FS
    """
    def __init__(self, root, support_size, query_size, cache=True):
        self.root = root
        self.support_size = support_size
        self.query_size = query_size
        self.cache = cache
        if cache:
            self.songs = {}

    def sample(self, artist):
        if self.cache and artist in self.songs:
            songs = self.songs[artist]
        else:
            artist_path = os.path.join(self.root, artist)
            songs = os.listdir(artist_path)
            if self.cache:
                self.songs[artist] = songs
        sample = np.random.choice(
            songs,
            size=self.support_size+self.query_size,
            replace=False)
        query = sample[:self.query_size]
        support = sample[self.query_size:]
        return query, support


class EpisodeSampler(object):
    def __init__(self, dataset, batch_size, support_size, query_size, max_len):
        self.dataset = dataset
        self.batch_size = batch_size
        self.support_size = support_size
        self.query_size = query_size
        self.max_len = max_len
        self.sq_sampler = SQSampler(dataset.root, support_size, query_size)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return 'EpisodeSampler("%s", "%s")' % (self.root, self.split)

    def get_episode(self):
        support = np.zeros((self.batch_size, self.support_size, self.max_len))
        query = np.zeros((self.batch_size, self.query_size, self.max_len))
        artists = np.random.choice(self.dataset, size=self.batch_size, replace=False)
        for batch_index, artist in enumerate(artists):
            support_songs, query_songs = self.sq_sampler.sample(artist)
            for support_index, song in enumerate(support_songs):
                parsed_song = self.dataset.load(artist, song)
                support[batch_index,support_index,:] = parsed_song
            for query_index, song in enumerate(query_songs):
                parsed_song = self.dataset.load(artist, song)
                query[batch_index,query_index,:] = parsed_song
        return Episode(support, query)
