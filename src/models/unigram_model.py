import numpy as np
import tensorflow as tf

from models.tf_model import TFModel
from models.base_model import convert_tokens_to_input_and_target


class UnigramModel(TFModel):
    """Unigram model that uses word frequencies to compute word probabilities.

    Use meta-training set to approximate word frequencies. During
    evaluation, ignore each episode's support set and evaluate only on query set.
    """

    def __init__(self, config):
        super(UnigramModel, self).__init__(config)

    def _define_placedholders(self):
        self._input_size = self._config['input_size']
        self._time_steps = self._config['max_len']

        self._words = tf.placeholder(
            tf.int32, [None, self._time_steps - 1])
        self._alpha = 1

    def _build_graph(self):
        word_count = tf.get_variable(
            'word_count', [self._input_size],
            initializer=tf.constant_initializer(self._alpha),
            trainable=False)
        flatten_words = tf.reshape(self._words, [-1])
        ones = tf.ones_like(flatten_words, dtype=tf.float32)
        self._train_op = tf.scatter_add(word_count, flatten_words, ones)

        sum_ = tf.reduce_sum(word_count)
        self._prob = tf.gather(word_count, flatten_words) / sum_
        self._avg_neg_log = -tf.reduce_mean(tf.log(self._prob))

        self._prob_all = word_count / sum_

    def train(self, episode):
        """Concatenate query and support sets to train."""
        X, Y = convert_tokens_to_input_and_target(
            episode.support)
        X2, Y2 = convert_tokens_to_input_and_target(
            episode.query)
        X = np.concatenate([X, X2])

        feed_dict = {}
        feed_dict[self._words] = X

        _, loss = self._sess.run([self._train_op, self._avg_neg_log],
                                 feed_dict=feed_dict)

        return loss

    def eval(self, episode):
        """Ignore support set and evaluate only on query set."""
        query_set = episode.query
        X, Y = convert_tokens_to_input_and_target(query_set)

        feed_dict = {}
        feed_dict[self._words] = Y
        avg_neg_log = self._sess.run(self._avg_neg_log,
                                     feed_dict=feed_dict)

        return avg_neg_log

    def sample(self, support_set, num):
        """Ignore support set for sampling."""
        pred_words = []

        for i in range(num):
            prob = self._sess.run(self._prob_all)
            word = np.argmax(prob)
            pred_words.append(word)

        return pred_words
