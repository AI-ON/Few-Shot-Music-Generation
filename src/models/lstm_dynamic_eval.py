import numpy as np
import tensorflow as tf

from models.fast_weights_lstm import FastWeightsLSTM
from models.base_model import convert_tokens_to_input_and_target


def convert_to_subsequences(sequences, subseq_len):
    """Break sequence into bunch of subsequences based on desired subseq_len.

    Arguments:
        sequences: (numpy int array) of size [S,N], where S is number of
            sequences and N is size of each sequence
        subseq_len: (int) size of subsequences desired to break sequence into
    Returns:
        sequences: (numpy int array) of size [S,n_subseq,subseq_len], where S is
            number of sequences, n_subseq is number of subsequences that make
            up whole sequence, and subseq_len is the size of each subsequence
    """
    n_songs, n_dim = np.shape(sequences)[0], np.shape(sequences)[1]

    n_subseq = n_dim // subseq_len
    sequences = np.reshape(
        sequences, [n_songs, n_subseq, subseq_len])
    return sequences


class LSTMDynamicEval(FastWeightsLSTM):

    def __init__(self, config):
        super(LSTMDynamicEval, self).__init__(config)

    def _define_placeholders(self):
        # Overwrite time steps as we are operating on subsequences
        assert_msg = 'Sequence length %d is not divisible by subsequence Length %d'\
            % (self._config['max_len'], self._config['subseq_len'])
        assert self._config['max_len'] % self._time_steps == 0, assert_msg
        self._time_steps = self._config['subseq_len']
        self._n_subseq = self._config['max_len'] // self._time_steps

        self._embd_size = self._config['embedding_size']
        self._hidden_size = self._config['hidden_size']
        self._n_layers = self._config['n_layers']
        self._lr = self._config['lr']
        self._max_grad_norm = self._config['max_grad_norm']
        self._stop_grad = True
        self._embedding_var_name = 'embedding'

        self._seq_length = tf.placeholder(tf.int32, [None])
        self._words = tf.placeholder(
            tf.int32, [None, None, self._time_steps])
        self._target = tf.placeholder(
            tf.int32, [None, None, self._time_steps])

    def _build_graph(self):
        self.weights = self._build_weights()
        elems = (self._words, self._target)
        self._avg_neg_log = tf.map_fn(
            self._dynamic_eval, elems=elems, dtype=tf.float32)
        self._avg_neg_log = tf.reduce_mean(self._avg_neg_log)

    def _dynamic_eval(self, _input):
        """Perform dynamic evaluation on a single sequence.

        Iterate through each subsequence and update parameters on loss of
        each subsequence. Loss is computed on each subsequence before update
        step.
        Arguments:
            _input: tuple containing words & target, where
            words: (tensor) of size (n_subseq, subseq_len)
            target: (tensor) of size (n_subseq, subseq_len)
        Returns:
            tensor of losses on each subsequence

        """
        words, target = _input
        train_losses = []

        fast_weights = self.weights
        final_state = None
        for i in range(self._n_subseq):
            X = tf.slice(words, begin=[i, 0], size=[1, -1])
            Y = tf.slice(target, begin=[i, 0], size=[1, -1])

            logits, _, final_state = self._model(
                fast_weights, X, 1, self._seq_length, final_state)
            loss_train = self._loss_fxn(logits, Y)
            train_losses.append(loss_train)

            grads = self._get_grads(loss_train, fast_weights)
            fast_weights = self._get_update(fast_weights, grads, self._lr)

        return tf.stack(loss_train)

    def train(self, episode):
        raise NotImplementedError()

    def eval(self, episode):
        """Ignore support set and perform dynamic evaluation on query set."""
        X, Y = convert_tokens_to_input_and_target(
            episode.query, self._start_word, flatten_batch=True)
        subseq_len = self._config['subseq_len']
        X = convert_to_subsequences(X, subseq_len)
        Y = convert_to_subsequences(Y, subseq_len)

        feed_dict = {}
        feed_dict[self._words] = X
        feed_dict[self._target] = Y
        feed_dict[self._seq_length] = [np.shape(X)[2]]
        avg_neg_log = self._sess.run(
            self._avg_neg_log, feed_dict=feed_dict)

        return avg_neg_log
