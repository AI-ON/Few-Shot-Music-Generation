#!/usr/bin/python3
"""A dataset class lyrics and MIDI data for the few-shot-music-gen project
"""
import os
import logging
import time

import numpy as np


log = logging.getLogger('few-shot')


class Dataset(object):
    """A class for train/val/test sets.

    This class is initialized with the following arguments:
    Arguments:
        root (str): the root directory of the dataset
        split ("train", "val", or "test"): the split of the dataset which
            this object represents.
        loader (LyricsLoader or MIDILoader): the object used for reading and
            parsing
        split_proportions (tuple of three numbers): the unnormalized
            (train, val, test) split.
        persist (bool): persists the train/val/test split information in csv
            files in `root`, so future runs will use the same splits. If those
            csvs already exist, the sampler uses the splits from those files.
        cache (bool): if true, caches the loaded/parsed songs in memory.
            Otherwise it loads and parses songs on every episode.
        min_songs (int): the minimum number of songs which an artist must have.
            If they don't have `min_songs` songs, they will not be present in
            the dataset.
    """
    def __init__(self, root, split, loader, split_proportions=(8,1,1),
            persist=True, cache=True, min_songs=0):
        self.root = root
        self.cache = cache
        self.cache_data = {}
        self.loader = loader
        split_csv_path = os.path.join(root, '%s.csv' % split)
        if persist and os.path.exists(split_csv_path):
            split_csv = open(split_csv_path, 'r')
            self.artists = [line.strip() for line in split_csv.readlines()]
            split_csv.close()
        else:
            dirs = []
            for artist in os.listdir(root):
                if os.path.isdir(os.path.join(root, artist)):
                    dirs.append(artist)
            artists = []
            skipped_count = 0
            num_dirs = len(dirs)
            last_log = 0
            for artist_index, artist in enumerate(dirs):
                # log progress every second
                if time.time() - last_log >= 1:
                    log.info("Preprocessing data. %s%%" % int(100*artist_index/num_dirs))
                    last_log = time.time()
                songs = os.listdir(os.path.join(root, artist))
                if len(songs) >= min_songs:
                    artists.append(artist)
                else:
                    skipped_count += 1
            if skipped_count > 0:
                log.info("%s artists don't have K+K'=%s songs. Using %s artists" % (
                    skipped_count, min_songs, len(artists)))
            train_count = int(float(split_proportions[0]) / sum(split_proportions) * len(artists))
            val_count = int(float(split_proportions[1]) / sum(split_proportions) * len(artists))
            np.random.shuffle(artists)
            if persist:
                train_csv = open(os.path.join(root, 'train.csv'), 'w')
                val_csv = open(os.path.join(root, 'val.csv'), 'w')
                test_csv = open(os.path.join(root, 'test.csv'), 'w')
                train_csv.write('\n'.join(artists[:train_count]))
                val_csv.write('\n'.join(artists[train_count:train_count+val_count]))
                test_csv.write('\n'.join(artists[train_count+val_count:]))
                train_csv.close()
                val_csv.close()
                test_csv.close()
            if split == 'train':
                self.artists = artists[:train_count]
            elif split == 'val':
                self.artists = artists[train_count:train_count+val_count]
            else:
                self.artists = artists[train_count+val_count:]

    def load(self, song, artist):
        """Read and parse `song` by `artist`.

        Arguments:
            song (str): the name of the song file. e.g. `"lateralus.txt"`
            artist (str): the name of the artist directory. e.g. `"tool"`
        """
        if self.cache and (song, artist) in self.cache_data:
            return self.cache_data[(song, artist)]
        else:
            data = self.loader(os.path.join(self.root, song, artist))
            self.cache_data[(song, artist)] = data
            return data

    def __len__(self):
        return len(self.artists)

    def __getitem__(self, index):
        return self.artists[index]
