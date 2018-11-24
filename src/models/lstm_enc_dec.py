import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import fully_connected
from models.nn_lib import LSTM, make_cell, get_sentinel_prob, num_stable_log,\
    seq_loss, get_ndcg, make_cell_film, LSTMFilm

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
    """LSTM language model which conditions on support set to evaluate query set.

    Trained on episodes from the meta-training set. During evaluation,
    use each episode's support set to produce encoding that is then
    used to condition when evaluating on query set.
    """

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
        # self._max_token_len = tf.placeholder(tf.int32, shape=())

    def _build_graph(self):
        elems = (self._supportX, self._supportY, self._support_seq_length,
                 self._queryX, self._queryY, self._query_seq_length)
        self._all_neg_log, self._query_neg_log, self._prob, self._enc \
            = tf.map_fn(self._train_episode, elems=elems,
                        dtype=(tf.float32, tf.float32, tf.float32, tf.float32))
        self._all_avg_neg_log = tf.reduce_mean(self._all_neg_log)
        self._query_avg_neg_log = tf.reduce_mean(self._query_neg_log)
        optimizer = tf.train.AdamOptimizer(self._lr)
        self._gvs = gvs = optimizer.compute_gradients(self._all_avg_neg_log)
        self._train_op = optimizer.apply_gradients(gvs, self._global_step)

    def _train_episode(self, _input):
        supportX, supportY, support_seq_length,\
            queryX, queryY, query_seq_length = _input

        enc = self._encode(
            supportY,
            self._support_size, support_seq_length)

        # Option of whether to decode only query OR support & query using
        # support encoding
        if self._config['decode_support']:
            X = tf.concat([supportX, queryX], axis=0)
            Y = tf.concat([supportY, queryY], axis=0)
            size = self._support_size + self._query_size
            seq_length = tf.concat(
                [support_seq_length, query_seq_length], axis=0)

            logits_both = self._decode(
                X,
                Y,
                size,
                seq_length,
                enc
            )
            loss_both = seq_loss(
                logits_both, Y,
                seq_length, self._time_steps,
                avg_batch=False)

            logits_query = tf.slice(
                logits_both, [self._support_size, 0, 0], [-1, -1, -1])
            loss_query = tf.contrib.seq2seq.sequence_loss(
                logits_query,
                queryY,
                query_seq_length,
                average_across_timesteps=True,
                average_across_batch=True)
            prob_query = tf.nn.softmax(logits_query)

            return loss_both, loss_query, prob_query, enc
        else:
            X = queryX
            Y = queryY
            size = self._query_size
            seq_length = query_seq_length

            logits = self._decode(
                X,
                Y,
                size,
                seq_length,
                enc
            )
            """
            max_token_len = tf.tile([self._max_token_len], [self._query_size])
            loss = seq_loss(
                logits, Y,
                tf.minimum(max_token_len, query_seq_length), self._time_steps,
                avg_batch=False)
            """
            loss = seq_loss(
                logits, Y,
                seq_length, self._time_steps,
                avg_batch=False)
            prob_query = tf.nn.softmax(logits)
            return loss, loss, prob_query, enc

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
        _, final_state = LSTM(
            cell, X, embedding,
            seq_length, size, initial_state, scope='lstm1'
        )
        final_state = final_state[0]
        # [support_size, 2*hidden_size]
        hidden_concat = tf.concat([final_state.c, final_state.h], axis=-1)
        pool_fxn = tf.reduce_mean

        """
        # Pool + FC
        # [1, 2*hidden_size]
        # mean_hidden = tf.reduce_mean(hidden_concat, axis=0, keep_dims=True)
        pool_hidden = pool_fxn(hidden_concat, axis=0, keep_dims=True)
        # [1, enc_size]
        # enc = fully_connected(
        #    mean_hidden, self._enc_size, activation_fn=tf.nn.tanh)
        enc = fully_connected(
            pool_hidden, self._enc_size, activation_fn=tf.nn.tanh)
        """

        # FC + Pool (seems to work better)
        # [support_size, enc_size]
        enc = fully_connected(
            hidden_concat, self._enc_size)
        # [1, enc_size]
        enc = pool_fxn(enc, axis=0, keepdims=True)

        return enc

    def _decode(self, X, Y, size, seq_length, enc):
        enc = tf.tile(enc, [size, 1])
        initial_state_h = fully_connected(
            enc, self._hidden_size, activation_fn=tf.nn.tanh)
        initial_state_c = fully_connected(
            enc, self._hidden_size, activation_fn=tf.nn.tanh)
        initial_state = (tf.contrib.rnn.LSTMStateTuple(
            initial_state_c, initial_state_h),)
        enc = tf.tile(tf.expand_dims(enc, axis=1), [1, self._time_steps, 1])

        with tf.variable_scope('shared', reuse=True):
            embedding = tf.get_variable(
                self._embedding_var_name, [self._input_size, self._embd_size])

        LSTMclass = None
        with tf.variable_scope('lstm2'):
            if self._config['use_film']:
                list_of_cells = [make_cell_film(
                    0, self._embd_size, self._enc_size, self._hidden_size)]
                LSTMclass = LSTMFilm
            else:
                list_of_cells = [make_cell(
                    0, self._embd_size + self._enc_size, self._hidden_size)]
                LSTMclass = LSTM

            cell = tf.contrib.rnn.MultiRNNCell(list_of_cells)

        # [n_query, time_step, n_hidden]
        hidden_states, _ = LSTMclass(
            cell, X, embedding,
            seq_length, size, initial_state, enc, scope='lstm2'
        )

        # [n_query*time_step, n_hidden]
        hidden_states = tf.reshape(
            hidden_states, [-1, self._hidden_size])
        logits = tf.matmul(hidden_states, embedding, transpose_b=True)
        logits = tf.reshape(
            logits, [size, self._time_steps, self._input_size])
        if not self._config['use_sentinel']:
            return logits

        prob_vocab = tf.nn.softmax(logits)
        g, prob_cache = get_sentinel_prob(
            Y, hidden_states, size,
            self._time_steps, self._hidden_size, self._input_size)
        prob = tf.multiply(g, prob_vocab) + prob_cache
        return num_stable_log(prob)

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
        # feed_dict[self._max_token_len] = self._config['max_len']

        _, loss = self._sess.run(
            [self._train_op, self._test_avg_neg_log], feed_dict=feed_dict)

        if self._summary_writer:
            summary = tf.Summary(value=[
                tf.Summary.Value(tag='Train/loss',
                                 simple_value=loss)])
            self._summary_writer.add_summary(summary, self._train_calls)
            self._train_calls += 1

        return loss

    def eval_ndcg(self, episode):
        # Evaluate NDCG ranking metric

        if np.shape(episode.support)[0] > 1:
            episode.support = episode.support[0:1, :, :]
            episode.query = episode.query[0:1, :, :]
            episode.other_query = episode.other_query[0:1, :, :]
            episode.support_seq_len = episode.support_seq_len[0:1, :]
            episode.query_seq_len = episode.query_seq_len[0:1, :]
            episode.other_query_seq_len = episode.other_query_seq_len[0:1, :]

        feed_dict = {}
        support_size = np.shape(episode.support)[1]
        query_size = np.shape(episode.query)[1]
        support_seq_len = episode.support_seq_len
        query_seq_len = episode.query_seq_len
        other_query_seq_len = episode.other_query_seq_len

        supportX, supportY = convert_tokens_to_input_and_target(
            episode.support, episode.support_seq_len,
            self._start_word, self._end_word,
            flatten_batch=False)
        queryX, queryY = convert_tokens_to_input_and_target(
            episode.query, episode.query_seq_len,
            self._start_word, self._end_word,
            flatten_batch=False)
        queryX_other, queryY_other = convert_tokens_to_input_and_target(
            episode.other_query, episode.other_query_seq_len,
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
        # feed_dict[self._max_token_len] = self._config['eval_len']

        nll, avg_nll = self._sess.run(
            [self._query_neg_log, self._query_avg_neg_log], feed_dict=feed_dict)

        feed_dict[self._supportX] = supportX
        feed_dict[self._supportY] = supportY
        feed_dict[self._queryX] = queryX_other
        feed_dict[self._queryY] = queryY_other
        feed_dict[self._support_size] = support_size
        feed_dict[self._query_size] = query_size
        feed_dict[self._support_seq_length] = support_seq_len + 1
        feed_dict[self._query_seq_length] = other_query_seq_len + 1
        feed_dict[self._is_training] = False
        # feed_dict[self._max_token_len] = self._config['eval_len']

        nll_other, _ = self._sess.run(
            [self._query_neg_log, self._query_avg_neg_log], feed_dict=feed_dict)

        nll = nll.flatten()
        nll_other = nll_other.flatten()
        rel_scores = np.ones(shape=np.shape(nll))
        rel_scores_neg_songs = np.zeros(shape=np.shape(nll_other))

        ndcg = get_ndcg(
            np.concatenate([rel_scores, rel_scores_neg_songs]),
            np.concatenate([nll, nll_other]),
            rank_position=np.shape(nll)[0])

        return ndcg

    def eval(self, episode):
        # Use support set to produce encoding that is then used to condition
        # LSTM when evaluating corresponding support set
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
        # feed_dict[self._max_token_len] = self._config['max_len']

        avg_neg_log = self._sess.run(
            self._query_avg_neg_log, feed_dict=feed_dict)

        if self._summary_writer:
            summary = tf.Summary(value=[
                tf.Summary.Value(tag='Eval/Avg_NLL',
                                 simple_value=avg_neg_log)])
            self._summary_writer.add_summary(summary, self._eval_calls)
            self._eval_calls += 1

        return avg_neg_log

    def sample(self, support_set, num):
        raise NotImplementedError()
