import tensorflow as tf
import numpy as np
from models.lstm_cell import LSTMCell
from models.lstm_film_cell import LSTMFilmCell


def num_stable_log(probs, eps=1e-7):
    _epsilon = eps
    return tf.log(tf.clip_by_value(probs, _epsilon, 1. - _epsilon))


def make_cell(n, embd_size, hidden_size):
    """Get LSTM cell."""
    """
    # Use Tensorflow Cell
    return tf.contrib.rnn.BasicLSTMCell(
        self._hidden_size, forget_bias=1., state_is_tuple=True)
    """
    return LSTMCell(n, embd_size, hidden_size)


def make_cell_film(n, embd_size, cond_size, hidden_size):
    """Get LSTM FILM-based cell."""
    return LSTMFilmCell(n, embd_size, cond_size, hidden_size)


def seq_loss(logits, Y, seq_length, time_steps,
             avg_timesteps=True, avg_batch=True):
    return tf.contrib.seq2seq.sequence_loss(
        logits,
        Y,
        tf.sequence_mask(seq_length, time_steps, dtype=tf.float32),
        average_across_timesteps=avg_timesteps,
        average_across_batch=avg_batch)


def LSTM(cell,
         onehot_X,
         embedding,
         seq_length,
         batch_size,
         initial_state,
         enc=None,
         scope="",
         reuse=False):

    # [batch_size, time_step, embd_size]
    inputs = tf.nn.embedding_lookup(embedding, onehot_X)

    if enc is not None:
        inputs = tf.concat([inputs, enc], axis=-1)

    # outputs: [batch_size, time_step, hidden_size]
    # state: [batch_size, hidden_size]
    with tf.variable_scope(scope, reuse=reuse):
        outputs, state = tf.nn.dynamic_rnn(
            cell, inputs, initial_state=initial_state,
            sequence_length=seq_length
        )

    return outputs, state


def LSTMFilm(cell,
             onehot_X,
             embedding,
             seq_length,
             batch_size,
             initial_state,
             cond,
             scope="",
             reuse=False):

    # [batch_size, time_step, embd_size]
    inputs = tf.nn.embedding_lookup(embedding, onehot_X)

    # outputs: [batch_size, time_step, hidden_size]
    # state: [batch_size, hidden_size]
    with tf.variable_scope(scope, reuse=reuse):
        outputs, state = tf.nn.dynamic_rnn(
            cell, (inputs, cond), initial_state=initial_state,
            sequence_length=seq_length
        )

    return outputs, state


def get_logits(hidden_states, emb_matrix, hidden_size):
    hidden_states = tf.reshape(hidden_states, [-1, hidden_size])
    return tf.matmul(hidden_states, emb_matrix, transpose_b=True)


def masked_softmax(logits, mask):
    """Masked softmax over dim 1.

    Args:
        logits: (N, L)
        mask: (N, L)
    Returns:
        probabilities (N, L)
    """
    indices = tf.where(mask)
    values = tf.gather_nd(logits, indices)
    denseShape = tf.cast(tf.shape(logits), tf.int64)
    sparseResult = tf.sparse_softmax(
        tf.SparseTensor(indices, values, denseShape))
    result = tf.scatter_nd(
        sparseResult.indices, sparseResult.values, sparseResult.dense_shape)
    result.set_shape(logits.shape)

    return result


def get_seq_mask(size, time_steps):
    """Get mask that doesn't allow items at curr time to be dependent on future.

    Args:
        size: 1st dimension of requested output
        time_steps: number of time steps being considered
    Returns:
        M: [size, time_steps, time_steps] tensor where for each of size
        matrices, the t^th row is only True for the previous (t-1) entries
    """
    # [time_steps]
    lengths = tf.range(time_steps, dtype=tf.int32)
    # [time_steps, time_steps]
    M = tf.sequence_mask(lengths, time_steps)
    # [size, time_steps, time_steps]
    M = tf.tile(tf.expand_dims(M, 0), [size, 1, 1])

    # [size*time_steps, time_steps]
    M = tf.reshape(M, [-1, time_steps])
    return M


def get_sentinel_prob(X, hidden_states, size,
                      time_steps, hidden_size, input_size,
                      embd_size=None):
    """Get probability according to pointer-sentinel mixture model.

    Args:
        X: [size, time_steps] tensor
        hidden_states: [size*time_steps, hidden_size] tensor
        size: size of 1st dimension of X
        time_steps: size of 2nd dimension of X
        hidden_size: size of hidden units
        input_size: size of output units
        cond: [size, embd_size]
    Returns:
        prob_b: [size, time_steps, 1] tensor
        prob_sentinel: [size, time_steps, input_size] tensor
    Reference:
        https://arxiv.org/abs/1609.07843
    """
    queryW = tf.get_variable(
        'queryW', [hidden_size, hidden_size])
    queryb = tf.get_variable(
        'queryb', [hidden_size])
    queryS = tf.get_variable(
        'queryS', [hidden_size, 1])

    # [size, time_steps, hidden_size]
    hidden_states_r = tf.reshape(
        hidden_states, [size, time_steps, hidden_size])
    # [size, time_steps, hidden_size]
    queryW = tf.tile(tf.expand_dims(queryW, 0), [size, 1, 1])
    query = tf.nn.tanh(tf.matmul(hidden_states_r, queryW) + queryb)
    # [size, time_steps, hidden_size] * [size, hidden_size, time_steps]
    # = [size, time_steps, time_steps]
    alpha = tf.matmul(query, tf.transpose(hidden_states_r, [0, 2, 1]))
    # [size*time_steps, time_steps]
    alpha = tf.reshape(alpha, [-1, time_steps])
    # [size*time_steps, hidden_size]
    query = tf.reshape(query, [-1, hidden_size])

    # [size*time_steps, 1]
    g = tf.matmul(query, queryS)
    # [size*time_steps, time_steps + 1]
    alpha_with_g = tf.concat([alpha, g], axis=-1)

    # [size*time_steps, time_steps]
    mask = get_seq_mask(size, time_steps)
    # [size*time_steps, time_steps + 1]
    mask_with_g = tf.concat(
        [mask, tf.ones([size * time_steps, 1], tf.bool)],
        axis=-1)
    # [size*time_steps, time_steps + 1]
    prob_with_g = masked_softmax(
        alpha_with_g,
        mask_with_g
    )

    # [size*time_steps, time_steps]
    prob_ptr = tf.slice(
        prob_with_g, [0, 0], [-1, time_steps])
    # [size*time_steps, 1]
    prob_g = tf.slice(
        prob_with_g, [0, time_steps], [-1, 1])
    prob_g = tf.reshape(prob_g, [size, time_steps, 1])
    prob_ptr = tf.reshape(
        prob_ptr, [size, time_steps, time_steps])

    # [size, time_steps, time_steps] * [size, time_steps, input_size]
    # [size, time_steps, input_size]
    onehot_X = tf.one_hot(X, input_size)
    prob_ptr = tf.matmul(prob_ptr, onehot_X)

    return prob_g, prob_ptr


def get_ndcg(rel_scores, neg_logs, rank_position):
    """Compute NDCG metric for neg log rankings of songs by a model.

    Given relevancy scores of songs, model negative log likelihoods of songs,
    and the rank position at which to evaluate, compute NDCG.
    [https://en.wikipedia.org/wiki/Discounted_cumulative_gain]
    """
    p = rank_position
    _, sorted_rel_scores = (
        list(t) for t in zip(*sorted(zip(neg_logs, rel_scores))))

    idxs = np.array(range(1, len(sorted_rel_scores) + 1))
    dcg = np.sum(sorted_rel_scores[:p] / np.log2(idxs[:p] + 1))

    ideal_scores = sorted(rel_scores, reverse=True)
    idcg = np.sum(ideal_scores[:p] / np.log2(idxs[:p] + 1))

    return dcg / idcg
