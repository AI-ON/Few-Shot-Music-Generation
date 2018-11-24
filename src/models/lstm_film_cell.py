import tensorflow as tf

from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple
from models.lstm_cell import LSTMCell


def sep(*args):
    return '/'.join(args)


class LSTMFilmCell(LSTMCell):
    """Compute LSTM cell with FILM conditioning.

    Reference:
        https://arxiv.org/abs/1709.07871
    """

    def __init__(self, n, input_size, cond_size, num_units, weights=None,
                 state_is_tuple=True, forget_bias=1):
        super(LSTMFilmCell, self).__init__(
            n, input_size, num_units, weights, state_is_tuple, forget_bias)
        self._cond_scale_kernel = tf.get_variable(
            sep(self._n, "cond_scale"),
            shape=[cond_size, 4 * num_units])
        self._cond_shift_kernel = tf.get_variable(
            sep(self._n, "cond_shift"),
            shape=[cond_size, 4 * num_units])

    def __call__(self, inputs, state):
        """Modified from
        https://github.com/tensorflow/tensorflow/blob/r1.9/tensorflow/python/ops/rnn_cell_impl.py#L614"""

        if self._state_is_tuple:
            c, h = state
        else:
            c, h = tf.split(value=state, num_or_size_splits=2, axis=1)

        # Note that using `add` and `multiply` instead of `+` and `*` gives a
        # performance improvement. So using those at the cost of readability.
        add = tf.add
        multiply = tf.multiply
        sigmoid = tf.sigmoid
        activation = tf.tanh

        i, cond = inputs
        o = h
        gate_inputs = tf.matmul(
            tf.concat([i, o], 1), self._kernel)
        gate_inputs = tf.nn.bias_add(gate_inputs, self._bias)
        cond_scale = tf.matmul(
            cond, self._cond_scale_kernel) + 1
        cond_shift = tf.matmul(
            cond, self._cond_shift_kernel)
        gate_inputs = add(multiply(cond_scale, gate_inputs), cond_shift)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = tf.split(
            value=gate_inputs, num_or_size_splits=4, axis=1)

        new_c = add(multiply(c, sigmoid(add(f, self._forget_bias_tensor))),
                    multiply(sigmoid(i), activation(j)))
        new_h = multiply(activation(new_c), sigmoid(o))

        if self._state_is_tuple:
            new_state = LSTMStateTuple(new_c, new_h)
        else:
            new_state = tf.concat([new_c, new_h], 1)

        return new_h, new_state
