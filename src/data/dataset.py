#!/usr/bin/python3
"""A dataset class lyrics and MIDI data for the few-shot-music-gen project
"""
import os
import logging
import time
import multiprocessing
import itertools
try:
    from urllib import quote, unquote # python 2
except ImportError:
    from urllib.parse import quote, unquote # python 3

import numpy as np


log = logging.getLogger('few-shot')
logging.basicConfig(level=logging.INFO)


class Metadata(object):
    """An object for tracking the metadata associated with a configuration of
    the sampler.
    """
    def __init__(self, root, name):
        self.dir = os.path.join(root, name)
        self.open_files = {}
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

    def exists(self, filename):
        return os.path.exists(os.path.join(self.dir, filename))

    def lines(self, filename):
        if self.exists(os.path.join(self.dir, filename)):
            for line in open(os.path.join(self.dir, filename), 'r'):
                yield line

    def write(self, filename, line):
        if filename not in self.open_files:
            self.open_files[filename] = open(os.path.join(self.dir, filename), 'a')
        self.open_files[filename].write(line)

    def close(self):
        for filename in self.open_files:
            self.open_files[filename].close()


class Dataset(object):
    """A class for train/val/test sets.

    NOTE/TODO: This object only creates disjoint subsets of the full dataset
    when persistence is enabled. It might be good to support disjoint subsets
    when persistence is disabled.

    This class is initialized with the following arguments:
    Arguments:
        root (str): the root directory of the dataset
        split ("train", "val", or "test"): the split of the dataset which
            this object represents.
        loader (Loader): the object used for reading and parsing
        metadata (Metadata): the object used for persisting metadata
        split_proportions (tuple of three numbers): the unnormalized
            (train, val, test) split. Specifying the split for all three subsets
            (despite the Dataset object only being initialized for a single
            subset) is necessary because the dataset will be split and persisted
            on the first instantiation of the Dataset object.
        persist (bool): persists the train/val/test split information in csv
            files in `root`, so future runs will use the same splits. If those
            csvs already exist, the sampler uses the splits from those files.
        cache (bool): if true, caches the loaded/parsed songs in memory.
            Otherwise it loads and parses songs on every episode.
        validate (bool): if true, validates every song at initialization. If
            the song doesn't pass validation, it is removed from the dataset.
            If persist is also set to true, the validation info will be
            persisted.
        min_songs (int): the minimum number of songs which an artist must have.
            If they don't have `min_songs` songs, they will not be present in
            the dataset.
        valid_songs_file (str): the name file which contains persists a list
            of valid songs.
        seed (int or None): the random seed which is used for shuffling the
            artists.
    """
    def __init__(self, root, split, loader, metadata, split_proportions=(8,1,1),
            persist=True, cache=True, validate=True, min_songs=0, parallel=False,
            valid_songs_file='valid_songs.csv', seed=None):
        self.root = root
        self.cache = cache
        self.cache_data = {}
        self.loader = loader
        self.metadata = metadata
        self.artists = []
        self.valid_songs_file = valid_songs_file
        valid_songs = {}
        artist_in_split = []

        # If we're both validating and using persistence, load any validation
        # data from disk. The format of the validation file is just a CSV
        # with two entries: artist and song. The artist is the name of the
        # artist (i.e. the directory (e.g. 'K_s Choice')) and the song is
        # the song file name (e.g. 'ironflowers.mid').
        if validate and persist:
            for line in self.metadata.lines(valid_songs_file):
                artist, song = line.rstrip('\n').split(',', 1)
                artist = unquote(artist)
                song = unquote(song)
                if artist not in valid_songs:
                    valid_songs[artist] = set()
                valid_songs[artist].add(song)

        if persist and self.metadata.exists('%s.csv' % split):
            artists_in_split = []
            for line in self.metadata.lines('%s.csv' % split):
                artists_in_split.append(line.rstrip('\n'))
        else:
            dirs = []
            all_artists = []
            skipped_count = 0
            pool = multiprocessing.Pool(multiprocessing.cpu_count())

            for artist in os.listdir(root):
                if os.path.isdir(os.path.join(root, artist)):
                    songs = os.listdir(os.path.join(root, artist))
                    songs = [song for song in songs if loader.is_song(song)]
                    if len(songs) > 0:
                        dirs.append(artist)

            num_dirs = len(dirs)
            progress_logger = ProgressLogger(num_dirs)

            for artist_index, artist in enumerate(dirs):
                songs = os.listdir(os.path.join(root, artist))
                # We only want .txt and .mid files. Filter all others.
                songs = [song for song in songs if loader.is_song(song)]
                # populate `valid_songs[artist]`
                if validate:
                    progress_logger.maybe_log(artist_index)
                    if artist not in valid_songs:
                        valid_songs[artist] = set()
                    songs_to_validate = [song for song in songs if song not in valid_songs[artist]]
                    song_files = [os.path.join(root, artist, song) for song in songs_to_validate]
                    if parallel:
                        mapped = pool.map(loader.validate, song_files)
                    else:
                        mapped = map(loader.validate, song_files)
                    validated = itertools.compress(songs_to_validate, mapped)
                    for song in validated:
                        song_file = os.path.join(root, artist, song)
                        if persist:
                            line = '%s,%s\n' % (quote(artist), quote(song))
                            self.metadata.write(self.valid_songs_file, line)
                        valid_songs[artist].add(song)
                else:
                    valid_songs[artist] = set(songs)

                if len(valid_songs[artist]) >= min_songs:
                    all_artists.append(artist)
                else:
                    skipped_count += 1
            pool.close()
            pool.join()
            if skipped_count > 0:
                log.info("%s artists don't have K+K'=%s songs. Using %s artists" % (
                    skipped_count, min_songs, len(all_artists)))
            train_count = int(float(split_proportions[0]) / sum(split_proportions) * len(all_artists))
            val_count = int(float(split_proportions[1]) / sum(split_proportions) * len(all_artists))
            # Use RandomState(seed) so that shuffles with the same set of
            # artists will result in the same shuffle on different computers.
            np.random.RandomState(seed).shuffle(all_artists)
            if persist:
                self.metadata.write('train.csv', '\n'.join(all_artists[:train_count]))
                self.metadata.write('val.csv', '\n'.join(all_artists[train_count:train_count+val_count]))
                self.metadata.write('test.csv', '\n'.join(all_artists[train_count+val_count:]))
            if split == 'train':
                artists_in_split = all_artists[:train_count]
            elif split == 'val':
                artists_in_split = all_artists[train_count:train_count+val_count]
            else:
                artists_in_split = all_artists[train_count+val_count:]

        self.metadata.close()

        for artist in artists_in_split:
            self.artists.append(ArtistDataset(artist, list(valid_songs[artist])))

    def load(self, artist, song):
        """Read and parse `song` by `artist`.

        Arguments:
            song (str): the name of the song file. e.g. `"lateralus.txt"`
            artist (str): the name of the artist directory. e.g. `"tool"`
        """
        if self.cache and (artist, song) in self.cache_data:
            return self.cache_data[(artist, song)]
        else:
            data = self.loader.load(os.path.join(self.root, artist, song))
            self.cache_data[(artist, song)] = data
            return data

    def __len__(self):
        return len(self.artists)

    def __getitem__(self, index):
        return self.artists[index]


class ArtistDataset(object):
    def __init__(self, artist, songs):
        self.name = artist
        self.songs = songs

    def __len__(self):
        return len(self.songs)

    def __getitem__(self, index):
        return self.songs[index]


class ProgressLogger(object):
    def __init__(self, num_dirs):
        self.last_log = 0
        self.last_log_percent = None
        self.num_dirs = num_dirs

    def maybe_log(self, index):
        # log progress at most every second
        if time.time() - self.last_log >= 1:
            if self.last_log_percent != '%.2f' % (100*index/self.num_dirs):
                self.last_log_percent = '%.2f' % (100*index/self.num_dirs)
                log.info("Preprocessing data. %s%%" % self.last_log_percent)
                self.last_log = time.time()
