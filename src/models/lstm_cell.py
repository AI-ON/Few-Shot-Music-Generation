import tensorflow as tf

from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple


def sep(*args):
    return '/'.join(args)


class LSTMCell(tf.contrib.rnn.BasicLSTMCell):

    def __init__(self, n, input_size, num_units, weights=None,
                 state_is_tuple=True, forget_bias=1):
        self._n = str(n)
        self._num_units = num_units = num_units
        self._state_is_tuple = state_is_tuple
        self._forget_bias = forget_bias

        if weights is None:
            self._kernel = tf.get_variable(
                sep(self._n, "kernel"),
                shape=[input_size + num_units, 4 * num_units])
            self._bias = tf.get_variable(
                sep(self._n, "bias"),
                shape=[4 * num_units],
                initializer=tf.constant_initializer(0.0))
        else:
            self._kernel = weights[sep(self._n, "kernel")]
            self._bias = weights[sep(self._n, "bias")]

        self._forget_bias_tensor = tf.constant(
            self._forget_bias, dtype=tf.float32)

    def __call__(self, inputs, state):
        """Borrowed from
        https://github.com/tensorflow/tensorflow/blob/r1.9/tensorflow/python/ops/rnn_cell_impl.py#L614"""

        if self._state_is_tuple:
            c, h = state
        else:
            c, h = tf.split(value=state, num_or_size_splits=2, axis=1)

        i = inputs
        o = h
        gate_inputs = tf.matmul(
            tf.concat([i, o], 1), self._kernel)
        gate_inputs = tf.nn.bias_add(gate_inputs, self._bias)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = tf.split(
            value=gate_inputs, num_or_size_splits=4, axis=1)

        # Note that using `add` and `multiply` instead of `+` and `*` gives a
        # performance improvement. So using those at the cost of readability.
        add = tf.add
        multiply = tf.multiply
        sigmoid = tf.sigmoid
        activation = tf.tanh
        new_c = add(multiply(c, sigmoid(add(f, self._forget_bias_tensor))),
                    multiply(sigmoid(i), activation(j)))
        new_h = multiply(activation(new_c), sigmoid(o))

        if self._state_is_tuple:
            new_state = LSTMStateTuple(new_c, new_h)
        else:
            new_state = tf.concat([new_c, new_h], 1)

        return new_h, new_state
