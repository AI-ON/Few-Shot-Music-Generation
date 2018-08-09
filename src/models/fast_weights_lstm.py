import tensorflow as tf
import numpy as np

from models.tf_model import TFModel
from models.lstm_cell import LSTMCell, sep


class FastWeightsLSTM(TFModel):
    """Defines functions for lstm-models that work via a fast weights mechanism."""

    def _define_placeholders(self):
        self._support_batch_size = tf.placeholder(tf.int32, shape=())
        self._query_batch_size = tf.placeholder(tf.int32, shape=())
        self._support_seq_length = tf.placeholder(tf.int32, [None])
        self._query_seq_length = tf.placeholder(tf.int32, [None])
        self._supportX = tf.placeholder(
            tf.int32, [None, None, self._time_steps])
        self._supportY = tf.placeholder(
            tf.int32, [None, None, self._time_steps])
        self._queryX = tf.placeholder(
            tf.int32, [None, None, self._time_steps])
        self._queryY = tf.placeholder(
            tf.int32, [None, None, self._time_steps])

    def _build_weights(self):
        """Contruct and return all weights for LSTM."""
        weights = {}

        embedding = tf.get_variable(
            self._embedding_var_name, [self._input_size, self._embd_size])
        weights[self._embedding_var_name] = embedding

        for i in range(self._n_layers):
            weights.update(self._build_cell_weights(i))

        softmax_w = tf.get_variable(
            'softmax_w', [self._hidden_size, self._input_size])
        softmax_b = tf.get_variable('softmax_b', [self._input_size])
        weights['softmax_w'] = softmax_w
        weights['softmax_b'] = softmax_b
        return weights

    def _build_cell_weights(self, n):
        """Construct and return all weights for single LSTM cell."""
        weights = {}
        n = str(n)

        vocabulary_size = self._embd_size
        n_units = self._hidden_size
        weights[sep(n, 'kernel')] = tf.get_variable(
            sep(n, 'kernel'), shape=[vocabulary_size + n_units, 4 * n_units])
        weights[sep(n, 'bias')] = tf.get_variable(
            sep(n, 'bias'),
            shape=[4 * n_units],
            initializer=tf.constant_initializer(0.0))

        return weights

    def _model(self, weights, X, batch_size, seq_length, initial_state=None):
        """LSTM model that accepts dynamic weights.

        Arguments:
            weights: (tensor) weights for LSTM
            X: input sequences for LSTM of size [B, N] where B is batch_size
                and N is length of sequences
            batch_size: batch_size of input
            seq_length: list of size batch_size indicating sequence length of
                each sequence
            initial_state: (optional) intial state to begin LSTM processing
        Returns:
            logits: (tensor) output logits of size [B, N, C] where C is number
                of outputs classes
            hidden_states: (tensor) hidden states of LSTM of size [B, N, H]
                where H is hidden state size
            final_state: (tensor) final hidden state of LSTM of size [B, H]
        """
        def make_cell(n):
            return LSTMCell(
                n, self._embd_size, self._hidden_size, weights=weights)

        embedding = weights[self._embedding_var_name]
        inputs = tf.nn.embedding_lookup(embedding, X)

        cell = tf.contrib.rnn.MultiRNNCell(
            [make_cell(i) for i in range(self._n_layers)])

        if initial_state is None:
            initial_state = cell.zero_state(batch_size, dtype=tf.float32)

        # tf.nn.static_rnn not working so dynamic_rnn
        hidden_states, final_state = tf.nn.dynamic_rnn(
            cell, inputs,
            initial_state=initial_state, sequence_length=seq_length
        )
        output = tf.reshape(hidden_states, [-1, self._hidden_size])

        # Reshape logits to be a 3-D tensor for sequence loss
        softmax_w = weights['softmax_w']
        softmax_b = weights['softmax_b']
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        logits = tf.reshape(
            logits, [batch_size, self._time_steps, self._input_size])

        return logits, hidden_states, final_state

    def _loss_fxn(self, logits, Y):
        """Sequence loss function for logits and target Y."""
        return tf.contrib.seq2seq.sequence_loss(
            logits,
            Y,
            tf.ones_like(Y, dtype=tf.float32),
            average_across_timesteps=True,
            average_across_batch=True)

    def _get_grads(self, loss, weights):
        """Get gradient for loss w/r/t weights."""
        grads = tf.gradients(loss, list(weights.values()))
        grads, _ = tf.clip_by_global_norm(grads, self._max_grad_norm)
        if self._stop_grad:
            grads = [tf.stop_gradient(grad) for grad in grads]

        return dict(zip(weights.keys(), grads))

    def _get_update(self, weights, gradients, lr):
        """Update weights using gradients and lr."""
        weight_updates = [weights[key] - lr * gradients[key]
                          for key in weights.keys()]
        return dict(zip(weights.keys(), weight_updates))

    def train(self, episode):
        raise NotImplementedError()

    def eval(self, episode):
        raise NotImplementedError()
