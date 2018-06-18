import numpy as np


class BaseModel(object):

    def __init__(self, config):
        self._config = config

    @property
    def name(self):
        return self._config['name']

    def train(self, episode):
        """Train model on episode.

        Args:
            episode: Episode object containing support and query set.
        """
        raise NotImplementedError()

    def eval(self, episode):
        """Evaluate model on episode.

        Args:
            episode: Episode object containing support and query set.
        """
        raise NotImplementedError()

    def sample(self, support_set, num):
        """Sample a sequence of size num conditioned on support_set.

        Args:
            support_set (numpy array): support set to condition the sample.
            num: size of sequence to sample.
        """
        raise NotImplementedError()

    def save(self, checkpt_path):
        """Save model's current parameters at checkpt_path.

        Args:
            checkpt_path (string): path where to save parameters.
        """
        raise NotImplementedError()

    def recover_or_init(self, init_path):
        """Recover or initialize model based on init_path.

        If init_path has appropriate model parameters, load them; otherwise,
        initialize parameters randomly.
        Args:
            init_path (string): path from where to load parameters.
        """
        raise NotImplementedError()


def flatten_first_two_dims(token_array):
    """Convert shape from [B,S,N] => [BxS,N]."""
    shape = token_array.shape
    return np.reshape(token_array, (shape[0] * shape[1], shape[2]))


def convert_tokens_to_input_and_target(token_array, start_word=None):
    """Convert token_array to input and target to use for model for
    sequence generation.

    If start_word is given, add to start of each sequence of tokens.
    Input is token_array without last item; Target is token_array without first item.

    Arguments:
        token_array (numpy int array): tokens array of size [B,S,N] where
            B is batch_size, S is number of songs, N is size of the song
        start_word (int): token to use for start word
    """
    X = flatten_first_two_dims(token_array)

    if start_word is None:
        Y = np.copy(X[:, 1:])
        X_new = X[:, :-1]
    else:
        Y = np.copy(X)
        start_word_column = np.full(
            shape=[np.shape(X)[0], 1], fill_value=start_word)
        X_new = np.concatenate([start_word_column, X[:, :-1]], axis=1)

    return X_new, Y
