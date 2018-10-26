import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import fully_connected
from models.nn_lib import LSTM, make_cell, get_sentinel_prob, num_stable_log,\
    seq_loss

from models.tf_model import TFModel
from models.base_model import convert_tokens_to_input_and_target


def word_dropout(tokens, unknown_token, dropout_p):
    """With probability dropout_p, replace tokens with unknown token.

    Args:
        tokens: np array of size [B,N,S]
        unknown_token: int
        dropout_p: float
    """
    if dropout_p > 0:
        original_shape = tokens.shape
        temp = tokens.flatten()
        bernoulli_sample = np.random.binomial(n=1, p=dropout_p, size=temp.size)
        idxs = np.where(bernoulli_sample == 1)
        temp[idxs] = unknown_token
        temp = temp.reshape(original_shape)
        return temp
    else:
        return tokens


class LSTMEncDec(TFModel):

    def __init__(self, config):
        super(LSTMEncDec, self).__init__(config)

    def _define_placeholders(self):
        self._embd_size = self._config['embedding_size']
        self._hidden_size = self._config['hidden_size']
        self._n_layers = self._config['n_layers']
        self._lr = self._config['lr']
        self._max_grad_norm = self._config['max_grad_norm']
        self._embedding_var_name = 'embedding'
        self._enc_size = self._config['enc_size']
        self._dropout_p = 0.2

        self._support_size = tf.placeholder(tf.int32, shape=())
        self._query_size = tf.placeholder(tf.int32, shape=())
        self._support_seq_length = tf.placeholder(tf.int32, [None, None])
        self._query_seq_length = tf.placeholder(tf.int32, [None, None])

        self._supportX = tf.placeholder(
            tf.int32, [None, None, self._time_steps])
        self._supportY = tf.placeholder(
            tf.int32, [None, None, self._time_steps])
        self._queryX = tf.placeholder(
            tf.int32, [None, None, self._time_steps])
        self._queryY = tf.placeholder(
            tf.int32, [None, None, self._time_steps])
        self._is_training = tf.placeholder_with_default(
            True, shape=(), name='is_training')

    def _build_graph(self):
        elems = (self._supportX, self._supportY, self._support_seq_length,
                 self._queryX, self._queryY, self._query_seq_length)
        self._all_avg_neg_log, self._test_avg_neg_log, self._prob, self._enc \
            = tf.map_fn(self._train_episode, elems=elems,
                        dtype=(tf.float32, tf.float32, tf.float32, tf.float32))
        self._all_avg_neg_log = tf.reduce_mean(self._all_avg_neg_log)
        self._test_avg_neg_log = tf.reduce_mean(self._test_avg_neg_log)
        optimizer = tf.train.AdamOptimizer(self._lr)
        self._gvs = gvs = optimizer.compute_gradients(self._all_avg_neg_log)
        self._train_op = optimizer.apply_gradients(gvs, self._global_step)

    def _train_episode(self, _input):
        supportX, supportY, support_seq_length,\
            queryX, queryY, query_seq_length = _input

        enc = self._encode(
            supportY,
            self._support_size, support_seq_length)

        """
        X = tf.concat([supportX, queryX], axis=0)
        Y = tf.concat([supportY, queryY], axis=0)
        logits_all = self._decode(
            X,
            self._support_size + self._query_size,
            tf.concat([support_seq_length, query_seq_length], axis=0),
            enc
        )
        loss_all = tf.contrib.seq2seq.sequence_loss(
            logits_all,
            Y,
            tf.sequence_mask(
                tf.concat([support_seq_length, query_seq_length], axis=0),
                self._time_steps, dtype=tf.float32),
            average_across_timesteps=True,
            average_across_batch=True)

        logits_test = tf.slice(
            logits_all, [self._support_size, 0, 0], [-1, -1, -1])
        loss_test = tf.contrib.seq2seq.sequence_loss(
            logits_test,
            queryY,
            tf.sequence_mask(query_seq_length,
                             self._time_steps, dtype=tf.float32),
            average_across_timesteps=True,
            average_across_batch=True)
        """
        logits_all = self._decode(
            queryX,
            self._query_size,
            query_seq_length,
            enc
        )
        logits_test = logits_all
        loss_all = seq_loss(
            logits_all, queryY, query_seq_length, self._time_steps)
        """
        loss_all = tf.contrib.seq2seq.sequence_loss(
            logits_all,
            queryY,
            tf.sequence_mask(
                query_seq_length,
                self._time_steps, dtype=tf.float32),
            average_across_timesteps=True,
            average_across_batch=True)
        """
        loss_test = loss_all

        prob_test = tf.nn.softmax(logits_test)
        return loss_all, loss_test, prob_test, enc

    def _encode(self, X, size, seq_length):
        with tf.variable_scope('shared'):
            embedding = tf.get_variable(
                self._embedding_var_name, [self._input_size, self._embd_size])

        with tf.variable_scope('lstm1'):
            cell = tf.contrib.rnn.MultiRNNCell(
                [make_cell(0, self._embd_size, self._hidden_size)])
        initial_state = cell.zero_state(
            self._support_size, dtype=tf.float32)

        # [support_size, time_steps, hidden_size]
        hidden_states, final_state = LSTM(
            cell, X, embedding,
            seq_length, size, initial_state, scope='lstm1'
        )
        """
        # gather last hidden state using seq_length
        # [support_size, hidden_size]
        final_state = tf.gather(hidden_states, seq_length, axis=1)
        print('gather')
        print(final_state)
        """

        """
        inputs = tf.unstack(inputs, axis=1)  # remove for dynamic
        initial_state = cell.zero_state(batch_size, dtype=tf.float32)
        hidden_states, final_state = self._rnn(cell, inputs, initial_state)
        """

        final_state = final_state[0]
        # [support_size, 2*hidden_size]
        hidden_concat = tf.concat([final_state.c, final_state.h], axis=-1)
        # [1, 2*hidden_size]
        mean_hidden = tf.reduce_mean(hidden_concat, axis=0, keep_dims=True)
        # [1, enc_size]
        enc = fully_connected(
            mean_hidden, self._enc_size, activation_fn=tf.nn.tanh)
        # enc = tf.reduce_mean(enc, axis=0, keepdims=True)
        return enc

    def _decode(self, X, size, seq_length, enc):
        enc = tf.tile(enc, [size, 1])
        initial_state_h = fully_connected(
            enc, self._hidden_size, activation_fn=tf.nn.tanh)
        initial_state_c = fully_connected(
            enc, self._hidden_size, activation_fn=tf.nn.tanh)
        initial_state = (tf.contrib.rnn.LSTMStateTuple(
            initial_state_c, initial_state_h),)
        # concat_state = tf.concat([initial_state_h, initial_state_c], axis=-1)
        # enc = tf.tile(
        #    tf.expand_dims(concat_state, axis=1), [1, self._time_steps, 1])
        enc = tf.tile(tf.expand_dims(enc, axis=1), [1, self._time_steps, 1])
        """
        enc_shape = enc.get_shape().as_list()
        noise_shape = [enc_shape[0], enc_shape[1]] + [1]
        enc = tf.layers.dropout(
            enc, training=self._is_training, rate=self._dropout_p,
            noise_shape=noise_shape)
        """
        with tf.variable_scope('shared', reuse=True):
            embedding = tf.get_variable(
                self._embedding_var_name, [self._input_size, self._embd_size])

        with tf.variable_scope('lstm2'):
            cell = tf.contrib.rnn.MultiRNNCell(
                [make_cell(0, self._embd_size + self._enc_size,
                           self._hidden_size)])

        # [n_query, time_step, n_hidden]
        hidden_states, _ = LSTM(
            cell, X, embedding,
            seq_length, size, initial_state, enc=enc, scope='lstm2'
        )

        # [n_query*time_step, n_hidden]
        hidden_states = tf.reshape(
            hidden_states, [-1, self._hidden_size])

        """
        softmax_w = tf.get_variable(
            'softmax_w', [self._hidden_size, self._input_size])
        softmax_b = tf.get_variable('softmax_b', [self._input_size])
        logits = tf.nn.xw_plus_b(hidden_states, softmax_w, softmax_b)
        """
        logits = tf.matmul(hidden_states, embedding, transpose_b=True)
        logits = tf.reshape(
            logits, [size, self._time_steps, self._input_size])
        prob_vocab = tf.nn.softmax(logits)

        g, prob_cache = get_sentinel_prob(
            X, hidden_states, size,
            self._time_steps, self._hidden_size, self._input_size)
        return num_stable_log(tf.multiply(g, prob_vocab) +
                              tf.multiply((1. - g), prob_cache))

    """
    def _build_weights(self):
        Contruct and return all weights for LSTM.
        enc_weights = {}
        dec_weights = {}

        embedding = tf.get_variable(
            self._embedding_var_name, [self._input_size, self._embd_size])
        enc_weights[self._embedding_var_name] = embedding
        dec_weights[self._embedding_var_name] = enc_weights[self._embedding_var_name]

        for i in range(self._n_layers):
            enc_weights.update(self._build_cell_weights(i, prefix='enc'))
            dec_weights.update(self._build_cell_weights(i, prefix='dec'))

        softmax_w = tf.get_variable(
            'softmax_w', [self._hidden_size, self._input_size])
        softmax_b = tf.get_variable('softmax_b', [self._input_size])
        dec_weights['softmax_w'] = softmax_w
        dec_weights['softmax_b'] = softmax_b

        return enc_weights, dec_weights

    def _build_cell_weights(self, n, prefix=""):
        Construct and return all weights for single LSTM cell.
        weights = {}
        n = str(n)

        vocabulary_size = self._embd_size
        n_units = self._hidden_size
        weights[sep(n, 'kernel')] = tf.get_variable(
            sep(n, prefix + '/kernel'),
            shape=[vocabulary_size + n_units, 4 * n_units])
        weights[sep(n, 'bias')] = tf.get_variable(
            sep(n, prefix + '/bias'),
            shape=[4 * n_units],
            initializer=tf.constant_initializer(0.0))

        return weights

    """

    def train(self, episode):
        """MAML training objective involves input of support and query sets."""
        feed_dict = {}
        support_size = np.shape(episode.support)[1]
        query_size = np.shape(episode.query)[1]
        support_seq_len = episode.support_seq_len
        query_seq_len = episode.query_seq_len

        supportX, supportY = convert_tokens_to_input_and_target(
            episode.support, episode.support_seq_len,
            self._start_word, self._end_word,
            flatten_batch=False)
        queryX, queryY = convert_tokens_to_input_and_target(
            episode.query, episode.query_seq_len,
            self._start_word, self._end_word,
            flatten_batch=False)

        feed_dict[self._supportX] = supportX
        feed_dict[self._supportY] = supportY
        feed_dict[self._queryX] = queryX
        feed_dict[self._queryY] = queryY
        feed_dict[self._support_size] = support_size
        feed_dict[self._query_size] = query_size
        feed_dict[self._support_seq_length] = support_seq_len + 1
        feed_dict[self._query_seq_length] = query_seq_len + 1

        _, loss = self._sess.run(
            [self._train_op, self._test_avg_neg_log], feed_dict=feed_dict)

        if self._summary_writer:
            summary = tf.Summary(value=[
                tf.Summary.Value(tag='Train/loss',
                                 simple_value=loss)])
            self._summary_writer.add_summary(summary, self._train_calls)
            self._train_calls += 1

        return loss

    def eval(self, episode):
        """Perform gradients steps on support set and evaluate on query set."""
        feed_dict = {}
        support_size = np.shape(episode.support)[1]
        query_size = np.shape(episode.query)[1]
        support_seq_len = episode.support_seq_len
        query_seq_len = episode.query_seq_len

        supportX, supportY = convert_tokens_to_input_and_target(
            episode.support, episode.support_seq_len,
            self._start_word, self._end_word,
            flatten_batch=False)
        queryX, queryY = convert_tokens_to_input_and_target(
            episode.query, episode.query_seq_len,
            self._start_word, self._end_word,
            flatten_batch=False)

        feed_dict[self._supportX] = supportX
        feed_dict[self._supportY] = supportY
        feed_dict[self._queryX] = queryX
        feed_dict[self._queryY] = queryY
        feed_dict[self._support_size] = support_size
        feed_dict[self._query_size] = query_size
        feed_dict[self._support_seq_length] = support_seq_len + 1
        feed_dict[self._query_seq_length] = query_seq_len + 1
        feed_dict[self._is_training] = False

        avg_neg_log = self._sess.run(
            self._test_avg_neg_log, feed_dict=feed_dict)

        if self._summary_writer:
            summary = tf.Summary(value=[
                tf.Summary.Value(tag='Eval/Avg_NLL',
                                 simple_value=avg_neg_log)])
            self._summary_writer.add_summary(summary, self._eval_calls)
            self._eval_calls += 1

        return avg_neg_log
