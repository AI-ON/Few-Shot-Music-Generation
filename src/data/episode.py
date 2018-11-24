#!/usr/bin/python3
import os
import logging
import yaml
import numpy as np

from data.midi_loader import MIDILoader
from data.lyrics_loader import LyricsLoader
from data.dataset import Dataset, Metadata
from data.lib import get_random

log = logging.getLogger('few-shot')
logging.basicConfig(level=logging.INFO)


class Episode(object):
    def __init__(self, support, support_seq_len, query, query_seq_len,
                 other_query=None, other_query_seq_len=None,
                 metadata_support=None, metadata_query=None):
        self.support = support
        self.support_seq_len = support_seq_len
        self.query = query
        self.query_seq_len = query_seq_len
        self.other_query = other_query
        self.other_query_seq_len = other_query_seq_len
        self.metadata_support = metadata_support
        self.metadata_query = metadata_query


class SQSampler(object):
    """A sampler for randomly sampling support/query sets.

    Arguments:
        support_size (int): number of songs in the support set
        query_size (int): number of songs in the query set
        random (RandomState): random generator to use
    """
    def __init__(self, support_size, query_size, random):
        self.support_size = support_size
        self.query_size = query_size
        self.random = random

    def sample(self, artist):
        sample = self.random.choice(
            artist,
            size=self.support_size + self.query_size,
            replace=False)
        query = sample[:self.query_size]
        support = sample[self.query_size:]
        return query, support

    def sample_from_artists(self, artists, n):
        ret = []
        for artist in artists:
            sample = self.random.choice(
                artist,
                size=n,
                replace=False)
            ret += sample.tolist()

        return ret


class EpisodeSampler(object):
    def __init__(self, dataset, batch_size, support_size, query_size, max_len,
                 dtype=np.int32, seed=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.support_size = support_size
        self.query_size = query_size
        self.max_len = max_len
        self.dtype = dtype
        self.seed = seed
        self.random = get_random(self.seed)
        self.sq_sampler = SQSampler(support_size, query_size, self.random)
        self.dataset.random = self.random

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return 'EpisodeSampler("%s", "%s")' % (self.root, self.split)

    def reset_seed(self):
        self.random = get_random(self.seed)
        self.sq_sampler.random = self.random
        self.dataset.random = self.random

    def get_artists_episode(self, artists):
        batch_size = len(artists)
        support = np.zeros(
            (batch_size, self.support_size, self.max_len), dtype=self.dtype)
        support_seq_len = np.zeros(
            (batch_size, self.support_size), dtype=self.dtype)
        query = np.zeros(
            (batch_size, self.query_size, self.max_len), dtype=self.dtype)
        query_seq_len = np.zeros(
            (batch_size, self.query_size), dtype=self.dtype)

        metadata_support = {}
        metadata_query = {}
        for batch_index, artist in enumerate(artists):
            query_songs, support_songs = self.sq_sampler.sample(artist)
            metadata_support[artist.name] = support_songs.tolist()
            metadata_query[artist.name] = query_songs.tolist()

            for support_index, song in enumerate(support_songs):
                parsed_song, parsed_len = self.dataset.load(artist.name, song)
                support[batch_index, support_index, :] = parsed_song
                support_seq_len[batch_index, support_index] = parsed_len
            for query_index, song in enumerate(query_songs):
                parsed_song, parsed_len = self.dataset.load(artist.name, song)
                query[batch_index, query_index, :] = parsed_song
                query_seq_len[batch_index, query_index] = parsed_len

        return Episode(support, support_seq_len, query, query_seq_len,
                       metadata_support=metadata_support,
                       metadata_query=metadata_query)

    def get_episode(self):
        support = np.zeros(
            (self.batch_size, self.support_size, self.max_len),
            dtype=self.dtype)
        support_seq_len = np.zeros(
            (self.batch_size, self.support_size), dtype=self.dtype)
        query = np.zeros(
            (self.batch_size, self.query_size, self.max_len), dtype=self.dtype)
        query_seq_len = np.zeros(
            (self.batch_size, self.query_size), dtype=self.dtype)
        artists = self.random.choice(
            self.dataset, size=self.batch_size, replace=False)

        metadata_support = {}
        metadata_query = {}
        for batch_index, artist in enumerate(artists):
            query_songs, support_songs = self.sq_sampler.sample(artist)
            metadata_support[artist.name] = support_songs.tolist()
            metadata_query[artist.name] = query_songs.tolist()

            for support_index, song in enumerate(support_songs):
                parsed_song, parsed_len = self.dataset.load(artist.name, song)
                support[batch_index, support_index, :] = parsed_song
                support_seq_len[batch_index, support_index] = parsed_len
            for query_index, song in enumerate(query_songs):
                parsed_song, parsed_len = self.dataset.load(artist.name, song)
                query[batch_index, query_index, :] = parsed_song
                query_seq_len[batch_index, query_index] = parsed_len

        return Episode(support, support_seq_len, query, query_seq_len,
                       metadata_support=metadata_support,
                       metadata_query=metadata_query)

    def get_episode_with_other_artists(self):
        support = np.zeros(
            (self.batch_size, self.support_size, self.max_len),
            dtype=self.dtype)
        support_seq_len = np.zeros(
            (self.batch_size, self.support_size), dtype=self.dtype)
        query = np.zeros(
            (self.batch_size, self.query_size, self.max_len), dtype=self.dtype)
        query_seq_len = np.zeros(
            (self.batch_size, self.query_size), dtype=self.dtype)
        other_query = np.zeros(
            (self.batch_size, self.query_size, self.max_len), dtype=self.dtype)
        other_query_seq_len = np.zeros(
            (self.batch_size, self.query_size), dtype=self.dtype)
        artists = self.random.choice(
            self.dataset, size=self.batch_size, replace=False)

        metadata_support = {}
        metadata_query = {}
        for batch_index, artist in enumerate(artists):
            query_songs, support_songs = self.sq_sampler.sample(artist)
            metadata_support[artist.name] = support_songs.tolist()
            metadata_query[artist.name] = query_songs.tolist()

            for support_index, song in enumerate(support_songs):
                parsed_song, parsed_len = self.dataset.load(artist.name, song)
                support[batch_index, support_index, :] = parsed_song
                support_seq_len[batch_index, support_index] = parsed_len
            for query_index, song in enumerate(query_songs):
                parsed_song, parsed_len = self.dataset.load(artist.name, song)
                query[batch_index, query_index, :] = parsed_song
                query_seq_len[batch_index, query_index] = parsed_len

            other_artists = self.random.choice(
                self.dataset, size=self.query_size + 1, replace=False)
            other_artists = other_artists.tolist()
            if artist in other_artists:
                other_artists.remove(artist)

            other_artists = other_artists[:self.query_size]
            other_songs = self.sq_sampler.sample_from_artists(other_artists, 1)

            other_artists_and_songs = zip(other_artists, other_songs)
            for index, (other_artist, song) in enumerate(other_artists_and_songs):
                parsed_song, parsed_len = self.dataset.load(
                    other_artist.name, song)
                other_query[batch_index, index, :] = parsed_song
                other_query_seq_len[batch_index, index] = parsed_len

        return Episode(support, support_seq_len, query, query_seq_len,
                       other_query, other_query_seq_len,
                       metadata_support, metadata_query)

    def get_num_unique_words(self):
        return self.dataset.loader.get_num_tokens()

    def get_unk_token(self):
        return self.dataset.loader.get_unk_token()

    def get_start_token(self):
        return self.dataset.loader.get_start_token()

    def get_stop_token(self):
        return self.dataset.loader.get_stop_token()

    def detokenize(self, numpy_data):
        return self.dataset.loader.detokenize(numpy_data)


def create_split(root, loader, metadata, seed,
                 split_proportions=(8, 1, 1), persist=True):
    train_exists = metadata.exists('train.csv')
    val_exists = metadata.exists('test.csv')
    test_exists = metadata.exists('val.csv')

    if train_exists and val_exists and test_exists:
        artist_splits = {}
        for split in ['train', 'test', 'val']:
            artist_splits[split] = []
            for line in metadata.lines('%s.csv' % split):
                artist_splits[split].append(line.rstrip('\n'))

        return artist_splits

    all_artists = []
    for artist in os.listdir(root):
        if os.path.isdir(os.path.join(root, artist)):
            songs = os.listdir(os.path.join(root, artist))
            songs = [s for s in songs if loader.is_song(s)]
            if len(songs) > 0:
                all_artists.append(artist)

    train_count = int(float(split_proportions[0]) /
                      sum(split_proportions) * len(all_artists))
    val_count = int(float(split_proportions[1]) /
                    sum(split_proportions) * len(all_artists))

    # Use RandomState(seed) so that shuffles with the same set of
    # artists will result in the same shuffle on different computers.
    np.random.RandomState(seed).shuffle(all_artists)
    split_train = all_artists[:train_count]
    split_val = all_artists[train_count:train_count + val_count]
    split_test = all_artists[train_count + val_count:]

    return {
        'train': split_train,
        'val': split_val,
        'test': split_test
    }


def load_all_samplers_from_config(config):
    if isinstance(config, str):
        config = yaml.load(open(config, 'r'))
    elif isinstance(config, dict):
        config = config
    else:
        config = yaml.load(config)
    required_keys = [
        'dataset_path',
        'query_size',
        'support_size',
        'batch_size',
        'min_len',
        'max_len',
        'dataset',
    ]
    for key in required_keys:
        if key not in config:
            raise RuntimeError('required config key "%s" not found' % key)

    root = config['dataset_path']
    if not os.path.isdir(root):
        raise RuntimeError('required data directory %s does not exist' % root)

    metadata_dir = 'few_shot_metadata_%s_%s' % (config['dataset'],
                                                config['max_len'])
    metadata = Metadata(root, metadata_dir)

    if config['dataset'] == 'lyrics':
        loader = LyricsLoader(
            config['min_len'], config['max_len'], config['max_unk_percent'],
            metadata=metadata, persist=False)
    elif config['dataset'] == 'midi':
        loader = MIDILoader(config['max_len'])
    else:
        raise RuntimeError('unknown dataset "%s"' % config['dataset'])
    artist_splits = create_split(
        root, loader, metadata, seed=config.get('dataset_seed', 0))

    if not loader.read_vocab():
        log.info("Building vocabulary using train")
        config['split'] = 'train'
        config['validate'] = True
        config['persist'] = False
        config['cache'] = False
        load_sampler_from_config(config, metadata, artist_splits, loader)
        loader.prune(config['word_min_times'])
        log.info("Vocabulary pruned!")

    loader.persist = True
    episode_sampler = {}
    for split in config['splits']:
        log.info("Encoding %s split using pruned vocabulary" % split)
        config['split'] = split
        config['validate'] = True
        config['persist'] = True
        config['cache'] = True
        episode_sampler[split] = load_sampler_from_config(
            config, metadata, artist_splits, loader)

    return episode_sampler


def load_sampler_from_config(config, metadata, artist_splits, loader=None):
    """Create an EpisodeSampler from a yaml config."""
    # Force batch_size of 1 for evaluation
    if config['split'] in ['val', 'test']:
        config['batch_size'] = 1

    root = config['dataset_path']
    if not os.path.isdir(root):
        raise RuntimeError('required data directory %s does not exist' % root)

    if loader is None:
        if config['dataset'] == 'lyrics':
            loader = LyricsLoader(config['max_len'], metadata=metadata)
        elif config['dataset'] == 'midi':
            loader = MIDILoader(config['max_len'])
        else:
            raise RuntimeError('unknown dataset "%s"' % config['dataset'])

    dataset = Dataset(
        root,
        loader,
        metadata,
        artist_splits[config['split']],
        seed=config.get('seed', None),
        cache=config.get('cache', True),
        persist=config.get('persist', True),
        validate=config.get('validate', True),
        min_songs=config['support_size'] + config['query_size'],
        artists_file='%s.csv' % config['split'],
        valid_songs_file='valid_songs_%s.csv' % config['split']
    )
    return EpisodeSampler(
        dataset,
        config['batch_size'],
        config['support_size'],
        config['query_size'],
        config['max_len'],
        seed=config.get('seed', None))
