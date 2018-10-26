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


def convert_tokens_to_input_and_target(token_array, seq_len_array,
                                       start_word, end_word,
                                       flatten_batch=True, x_and_y=True):
    if token_array.ndim == 3 and flatten_batch:
        X = flatten_first_two_dims(token_array)
    else:
        X = token_array

    if x_and_y:
        if X.ndim == 2:
            # Y is X + end_word but have to add an extra column to Y
            # make sure end_word correctly ends each sequence
            # in case of maximal sequence length
            Y = np.copy(X)
            n_rows = np.shape(X)[0]
            extra_column = np.zeros((n_rows, 1))
            Y = np.concatenate([Y, extra_column], axis=1)
            Y[range(n_rows), seq_len_array.flatten()] = end_word

            # X_new is start_word + X
            start_word_column = np.full(
                shape=[n_rows, 1], fill_value=start_word)
            X_new = np.concatenate([start_word_column, X], axis=1)
        elif X.ndim == 3:
            Y = np.copy(X)
            n_rows, n_cols = np.shape(X)[0], np.shape(X)[1]
            extra_column = np.zeros((n_rows, n_cols, 1))
            Y = np.concatenate([Y, extra_column], axis=2)
            indices_arr = np.indices((n_rows, n_cols))
            seq_len_array = seq_len_array.flatten()
            idx1_flat = indices_arr[0].flatten()
            idx2_flat = indices_arr[1].flatten()
            Y[idx1_flat, idx2_flat, seq_len_array] = end_word

            start_word_column = np.full(
                shape=[n_rows, n_cols, 1], fill_value=start_word)
            X_new = np.concatenate([start_word_column, X], axis=2)

        return X_new, Y
    """
    else:
        if X.ndim == 3:
            n_rows, n_cols = np.shape(X)[0], np.shape(X)[1]
            start_word_column = np.full(
                shape=[n_rows, n_cols, 1], fill_value=start_word)
            X_new = np.concatenate([start_word_column, X], axis=2)

            Y = np.copy(X_new)
            x_last_column = np.expand_dims(X[:, :, -1], 2)
            # print(Y.shape)
            # print(x_last_column.shape)
            Y = np.concatenate([Y, x_last_column], axis=2)

            n_rows, n_cols = np.shape(X)[0], np.shape(X)[1]
            indices_arr = np.indices((n_rows, n_cols))
            seq_len_array = (seq_len_array + 1).flatten()
            idx1_flat = indices_arr[0].flatten()
            idx2_flat = indices_arr[1].flatten()
            Y[idx1_flat, idx2_flat, seq_len_array] = end_word

            return Y
    """

"""
def convert_tokens_to_input_and_target(token_array, start_word=None,
                                       flatten_batch=True):
    Convert token_array to input and target to use for model for
    sequence generation.

    If start_word is given, add to start of each sequence of tokens.
    Input is token_array without last item; Target is token_array without first item.

    Arguments:
        token_array (numpy int array): tokens array of size [B,S,N] or [S, N]
            where B is batch_size, S is number of songs, N is size of the song
        start_word (int): token to use for start word
        flatten_batch (boolean): whether to flatten along the batch dimension
    Returns:
        X_new (numpy int array): input tokens of size either
            a. [B*S,N] if token_array is [B,S,N] and flatten_batch=True
            b. [B,S,N] if token_array is [B,S,N] and flatten_batch=False
            c. [S,N] if token_array is [S,N]
        Y (numpy int array): output tokens of same size as X_new

    if token_array.ndim == 3 and flatten_batch:
        X = flatten_first_two_dims(token_array)
    else:
        X = token_array

    if X.ndim == 2:
        if start_word is None:
            Y = np.copy(X[:, 1:])
            X_new = X[:, :-1]
        else:
            Y = np.copy(X)
            start_word_column = np.full(
                shape=[np.shape(X)[0], 1], fill_value=start_word)
            X_new = np.concatenate([start_word_column, X[:, :-1]], axis=1)
    elif X.ndim == 3:
        if start_word is None:
            Y = np.copy(X[:, :, 1:])
            X_new = X[:, :, :-1]
        else:
            Y = np.copy(X)
            start_word_column = np.full(
                shape=[np.shape(X)[0], np.shape(X)[1], 1], fill_value=start_word)
            X_new = np.concatenate([start_word_column, X[:, :, :-1]], axis=2)

    return X_new, Y
"""
