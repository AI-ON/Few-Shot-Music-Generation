import numpy as np
import tensorflow as tf

from models.fast_weights_lstm import FastWeightsLSTM
from models.base_model import convert_tokens_to_input_and_target


def _log(probs, eps=1e-7):
    _epsilon = eps
    return tf.log(tf.clip_by_value(probs, _epsilon, 1. - _epsilon))


class LSTMContCache(FastWeightsLSTM):

    def __init__(self, config):
        super(LSTMContCache, self).__init__(config)

    def _define_placeholders(self):
        self._embd_size = self._config['embedding_size']
        self._hidden_size = self._config['hidden_size']
        self._n_layers = self._config['n_layers']
        self._lr = self._config['lr']
        self._max_grad_norm = self._config['max_grad_norm']
        self._embedding_var_name = 'embedding'

        # Hyperparameters Lambda and Theta as defined in continuous cache paper
        self._lambda = self._config['lambda']
        self._theta = self._config['theta']

        super(LSTMContCache, self)._define_placeholders()

    def _build_graph(self):
        self.weights = self._build_weights()
        elems = (self._supportX, self._supportY, self._queryX, self._queryY)
        self._test_avg_neg_log = tf.map_fn(
            self._use_cont_cache, elems=elems, dtype=tf.float32)
        self._test_avg_neg_log = tf.reduce_mean(self._test_avg_neg_log)
        lr = self._config['lr']
        optimizer = tf.train.AdamOptimizer(lr)
        self._gvs = gvs = optimizer.compute_gradients(self._test_avg_neg_log)
        self._train_op = optimizer.apply_gradients(gvs, self._global_step)

    def _use_cont_cache(self, _input):
        """Use continuous cache computed on support set for query set output.

        Compute cache on hidden states of LSTM processed on support set
        and use cache to produce output distribution for query set.
        Arguments:
            supportX: support set input of size [S,N] where S is number of songs
                and N is size of songs
            supportY: support set target of size [S,N]
            queryX: query set input of size [S',N] where S' is number of songs
                and N is size of songs
            queryY: query set target of size [S',N]
        Returns:
            loss on query set using output distribution that is mixture of
            regular LSTM distribution and cache distribution

        """
        supportX, supportY, queryX, queryY = _input

        _, all_hidden_train, _ = self._model(
            self.weights, supportX,
            self._support_batch_size, self._support_seq_length)
        # convert train hidden states from
        # [batch_size, time_step, n_hidden]
        # => [n_hidden, support_batch_size * time_step]
        all_hidden_train = tf.reshape(all_hidden_train, [-1, self._hidden_size])
        all_hidden_train = tf.transpose(all_hidden_train)
        # convert from [support_batch_size, time_step]
        # => [support_batch_size * time_step, input_size]
        supportY_one_hot = tf.one_hot(
            tf.reshape(supportY, [-1]), self._input_size)

        logits, all_hidden_test, _ = self._model(
            self.weights, queryX,
            self._query_batch_size, self._query_seq_length)
        # distribution according to LSTM
        lstm_prob = tf.nn.softmax(logits)
        # test hidden states: [query_batch_size * time_step, n_hidden]
        all_hidden_test = tf.reshape(all_hidden_test, [-1, self._hidden_size])

        # [query_batch_size * time_step, support_batch_size * time_step]
        sim_scores = tf.matmul(
            all_hidden_test, all_hidden_train)
        # [query_batch_size * time_step, support_batch_size * time_step]
        p = tf.nn.softmax(self._theta * sim_scores)
        # [query_batch_size * time_step, input_size]
        p = tf.matmul(p, supportY_one_hot)
        # [query_batch_size, time_step, input_size]
        cache_prob = tf.reshape(
            p, [self._query_batch_size, self._time_steps, self._input_size])

        # final distribution is mixture of LSTM and cache distributions
        prob = (1. - self._lambda) * lstm_prob + self._lambda * cache_prob

        # convert prob distribution to logits for loss function
        return self._loss_fxn(_log(prob), queryY)
        """
        return tf.contrib.seq2seq.sequence_loss(
            _log(prob),
            queryY,
            tf.ones_like(queryY, dtype=tf.float32),
            average_across_timesteps=True,
            average_across_batch=True)
        """

    def train(self, episode):
        """Use support set to compute cache and train on loss on query set."""
        feed_dict = {}
        support_batch_size = np.shape(episode.support)[1]
        query_batch_size = np.shape(episode.query)[1]

        supportX, supportY = convert_tokens_to_input_and_target(
            episode.support, self._start_word, flatten_batch=False)
        queryX, queryY = convert_tokens_to_input_and_target(
            episode.query, self._start_word, flatten_batch=False)
        feed_dict[self._supportX] = supportX
        feed_dict[self._supportY] = supportY
        feed_dict[self._queryX] = queryX
        feed_dict[self._queryY] = queryY
        feed_dict[self._support_batch_size] = support_batch_size
        feed_dict[self._query_batch_size] = query_batch_size
        feed_dict[self._support_seq_length] = [np.shape(supportX)[2]] * np.shape(supportX)[1]
        feed_dict[self._query_seq_length] = [np.shape(queryX)[2]] * np.shape(queryX)[1]

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
        """Use support set to compute cache and evaluate on query set."""
        feed_dict = {}
        support_batch_size = np.shape(episode.support)[1]
        query_batch_size = np.shape(episode.query)[1]

        supportX, supportY = convert_tokens_to_input_and_target(
            episode.support, self._start_word, flatten_batch=False)
        queryX, queryY = convert_tokens_to_input_and_target(
            episode.query, self._start_word, flatten_batch=False)
        feed_dict[self._supportX] = supportX
        feed_dict[self._supportY] = supportY
        feed_dict[self._queryX] = queryX
        feed_dict[self._queryY] = queryY
        feed_dict[self._support_batch_size] = support_batch_size
        feed_dict[self._query_batch_size] = query_batch_size
        feed_dict[self._support_seq_length] = [np.shape(supportX)[2]] * np.shape(supportX)[1]
        feed_dict[self._query_seq_length] = [np.shape(queryX)[2]] * np.shape(queryX)[1]

        avg_neg_log = self._sess.run(
            self._test_avg_neg_log, feed_dict=feed_dict)

        if self._summary_writer:
            summary = tf.Summary(value=[
                tf.Summary.Value(tag='Eval/Avg_NLL',
                                 simple_value=avg_neg_log)])
            self._summary_writer.add_summary(summary, self._eval_calls)
            self._eval_calls += 1

        return avg_neg_log
