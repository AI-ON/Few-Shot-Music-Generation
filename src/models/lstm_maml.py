import tensorflow as tf
import numpy as np

from models.fast_weights_lstm import FastWeightsLSTM
from models.base_model import convert_tokens_to_input_and_target


class LSTMMAML(FastWeightsLSTM):

    def __init__(self, config):
        super(LSTMMAML, self).__init__(config)

    def _define_placeholders(self):
        self._embd_size = self._config['embedding_size']
        self._hidden_size = self._config['hidden_size']
        self._n_layers = self._config['n_layers']
        self._lr = self._config['lr']
        self._meta_lr = self._config['meta_lr']
        self._max_grad_norm = self._config['max_grad_norm']
        self._stop_grad = self._config['stop_grad']
        self._n_update = self._config['n_update']
        self._embedding_var_name = 'embedding'

        super(LSTMMAML, self)._define_placeholders()

    def _build_graph(self):
        self.weights = self._build_weights()
        elems = (self._supportX, self._supportY, self._queryX, self._queryY)
        self._test_avg_neg_log = tf.map_fn(
            self._train_episode, elems=elems, dtype=tf.float32)
        self._test_avg_neg_log = tf.reduce_mean(self._test_avg_neg_log)
        optimizer = tf.train.AdamOptimizer(self._meta_lr)
        self._gvs = gvs = optimizer.compute_gradients(self._test_avg_neg_log)
        self._train_op = optimizer.apply_gradients(gvs, self._global_step)

    def _train_episode(self, _input):
        supportX, supportY, queryX, queryY = _input
        train_losses = []

        fast_weights = self.weights
        for i in range(self._n_update):
            logits, _, _ = self._model(
                fast_weights, supportX,
                self._support_batch_size, self._support_seq_length
            )
            loss_train = self._loss_fxn(logits, supportY)
            train_losses.append(loss_train)

            grads = self._get_grads(loss_train, fast_weights)
            fast_weights = self._get_update(fast_weights, grads, self._lr)

        logits, _, _ = self._model(
            fast_weights, supportX,
            self._support_batch_size, self._support_seq_length)
        loss_train = self._loss_fxn(logits, supportY)
        train_losses.append(loss_train)

        logits, _, _ = self._model(
            fast_weights, queryX,
            self._query_batch_size, self._query_seq_length)
        test_loss = self._loss_fxn(logits, queryY)

        return test_loss

    def train(self, episode):
        """MAML training objective involves input of support and query sets."""
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
        """Perform gradients steps on support set and evaluate on query set."""
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
